#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import torch
import torch.multiprocessing as mp
from datasets import Audio, load_dataset
from tqdm import tqdm

from irodori_tts.codec import DACVAECodec
from irodori_tts.text_normalization import normalize_text


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value)
    return str(value)


def _sanitize_id_component(value: Any, *, fallback: str) -> str:
    raw = _coerce_text(value).strip()
    if not raw:
        return fallback
    # Keep Unicode letters/digits (e.g. non-ASCII speaker IDs), while removing
    # separators/control chars that can break parsing or downstream tooling.
    s = unicodedata.normalize("NFKC", raw)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[:/\\\\]+", "-", s)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    s = re.sub(r"[^\w.\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-_.")
    if not s:
        s = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    if len(s) > 96:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        s = f"{s[:80]}-{digest}"
    return s


def _resolve_speaker_namespace(args: argparse.Namespace) -> str:
    base = args.speaker_id_prefix if args.speaker_id_prefix else args.dataset
    if args.config:
        base = f"{base}:{args.config}"
    return _sanitize_id_component(base, fallback="dataset")


def _coerce_audio(audio_value: Any) -> tuple[torch.Tensor, int]:
    wav: torch.Tensor
    sr: int

    if isinstance(audio_value, dict):
        if "array" not in audio_value or "sampling_rate" not in audio_value:
            raise ValueError("Audio dict must include keys: 'array', 'sampling_rate'")
        wav = torch.as_tensor(audio_value["array"]).float()
        sr = int(audio_value["sampling_rate"])
    elif hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        wav = torch.as_tensor(samples.data).float()
        sr = int(samples.sample_rate)
    elif hasattr(audio_value, "data") and hasattr(audio_value, "sample_rate"):
        wav = torch.as_tensor(audio_value.data).float()
        sr = int(audio_value.sample_rate)
    else:
        raise TypeError(f"Unsupported audio value type: {type(audio_value)}")

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        if wav.shape[1] <= 8 and wav.shape[0] > wav.shape[1]:
            wav = wav.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"Unsupported decoded audio shape: {tuple(wav.shape)}")

    if wav.numel() == 0:
        raise ValueError("Decoded audio is empty")

    return wav, sr


def parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected float or one of [none, null, off, disable, disabled], got: {value}"
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"normalize-db must be finite, got: {value}")
    return out


def _parse_data_files(items: list[str] | None) -> Any:
    if items is None:
        return None

    flat_items: list[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        flat_items.append(item)

    if not flat_items:
        return None

    if len(flat_items) == 1:
        raw = flat_items[0]
        if raw.startswith("{") or raw.startswith("["):
            return json.loads(raw)

    if any("=" in x for x in flat_items):
        out: dict[str, list[str]] = {}
        for item in flat_items:
            if "=" not in item:
                raise ValueError(
                    "When using split-qualified --data-files, all entries must be split=path."
                )
            split_name, path_spec = item.split("=", 1)
            split_name = split_name.strip()
            paths = [p.strip() for p in path_spec.split(",") if p.strip()]
            if not split_name or not paths:
                raise ValueError(f"Invalid --data-files entry: {item}")
            out.setdefault(split_name, []).extend(paths)
        reduced: dict[str, Any] = {}
        for split_name, paths in out.items():
            reduced[split_name] = paths[0] if len(paths) == 1 else paths
        return reduced

    if len(flat_items) == 1 and "," in flat_items[0]:
        return [p.strip() for p in flat_items[0].split(",") if p.strip()]
    if len(flat_items) == 1:
        return flat_items[0]
    return flat_items


def _parse_speaker_columns(items: list[str] | None) -> list[str]:
    if items is None:
        return []

    out: list[str] = []
    for item in items:
        for column in str(item).split(","):
            column = column.strip()
            if column:
                out.append(column)
    return out


@dataclass
class _PreparedItem:
    idx: int
    status: str  # "ok", "skip", "error"
    text: str | None = None
    caption: str | None = None
    wav: torch.Tensor | None = None
    sample_rate: int | None = None
    speaker_id: str | None = None
    skip_reason: str | None = None
    error: str | None = None


_END = object()


def _prepare_example(
    idx: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> _PreparedItem:
    try:
        text = _coerce_text(sample.get(args.text_column, ""))
        if args.text_normalize:
            text = normalize_text(text)
        text = text.strip()
        caption = None
        if args.caption_column is not None:
            caption = _coerce_text(sample.get(args.caption_column, ""))
            caption = caption.strip() or None

        if not text:
            return _PreparedItem(idx=idx, status="skip", skip_reason="empty_text")

        speaker_id: str | None = None
        if args.speaker_columns:
            speaker_components: list[str] = []
            for speaker_column in args.speaker_columns:
                speaker_raw = sample.get(speaker_column, None)
                speaker_component = _sanitize_id_component(speaker_raw, fallback="")
                if speaker_component:
                    speaker_components.append(speaker_component)

            # If speaker columns are configured but all values are empty for this row,
            # keep the sample and simply omit speaker_id. Training handles this path.
            if speaker_components:
                speaker_component = (
                    speaker_components[0]
                    if len(speaker_components) == 1
                    else "__".join(speaker_components)
                )
                speaker_id = f"{args.speaker_id_namespace}:{speaker_component}"

        try:
            wav, sr = _coerce_audio(sample[args.audio_column])
        except Exception as exc:
            return _PreparedItem(
                idx=idx,
                status="skip",
                skip_reason="audio_decode",
                error=str(exc),
            )

        if args.min_sample_rate > 0 and sr < args.min_sample_rate:
            return _PreparedItem(idx=idx, status="skip", skip_reason="low_sample_rate")

        if args.max_seconds is not None:
            wav = wav[:, : int(args.max_seconds * sr)]
            if wav.numel() == 0:
                return _PreparedItem(idx=idx, status="skip", skip_reason="trimmed_empty")

        return _PreparedItem(
            idx=idx,
            status="ok",
            text=text,
            caption=caption,
            wav=wav,
            sample_rate=sr,
            speaker_id=speaker_id,
        )
    except Exception as exc:
        return _PreparedItem(
            idx=idx,
            status="error",
            skip_reason="prepare_error",
            error=str(exc),
        )


def _start_prefetch(
    iterator: Iterable[_PreparedItem | tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[Queue, Event, Thread]:
    queue: Queue = Queue(maxsize=max(1, args.prefetch))
    stop_event = Event()
    worker_count = max(1, int(getattr(args, "prefetch_workers", 1)))
    if worker_count == 1:

        def _worker() -> None:
            try:
                for entry in iterator:
                    if stop_event.is_set():
                        break
                    if isinstance(entry, _PreparedItem):
                        queue.put(entry)
                        continue
                    idx, sample = entry
                    queue.put(_prepare_example(idx, sample, args))
            except Exception as exc:
                queue.put(
                    _PreparedItem(
                        idx=-1,
                        status="error",
                        skip_reason="dataset_iter_error",
                        error=str(exc),
                    )
                )
            finally:
                queue.put(_END)

        thread = Thread(target=_worker, daemon=True)
        thread.start()
        return queue, stop_event, thread

    raw_queue: Queue = Queue(maxsize=max(1, args.prefetch * worker_count))

    def _reader() -> None:
        try:
            for entry in iterator:
                if stop_event.is_set():
                    break
                if isinstance(entry, _PreparedItem):
                    queue.put(entry)
                    continue
                raw_queue.put(entry)
        except Exception as exc:
            queue.put(
                _PreparedItem(
                    idx=-1,
                    status="error",
                    skip_reason="dataset_iter_error",
                    error=str(exc),
                )
            )
        finally:
            for _ in range(worker_count):
                raw_queue.put(_END)

    def _worker() -> None:
        while True:
            item = raw_queue.get()
            if item is _END:
                break
            idx, sample = item
            if stop_event.is_set():
                continue
            queue.put(_prepare_example(idx, sample, args))
        queue.put(_END)

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()
    for _ in range(worker_count):
        Thread(target=_worker, daemon=True).start()
    return queue, stop_event, reader_thread


def _first_index_for_rank(start: int, rank: int, world_size: int) -> int:
    return start + ((rank - start) % world_size)


def _count_rank_items(start: int, end: int, rank: int, world_size: int) -> int:
    if end <= start:
        return 0
    first = _first_index_for_rank(start, rank, world_size)
    if first >= end:
        return 0
    return ((end - 1 - first) // world_size) + 1


def _count_rank_items_contiguous(start: int, end: int, rank: int, world_size: int) -> int:
    if end <= start:
        return 0
    total = end - start
    per_rank = int(math.ceil(total / world_size))
    shard_start = start + (rank * per_rank)
    shard_end = min(end, shard_start + per_rank)
    return max(0, shard_end - shard_start)


def _is_map_style_dataset(dataset, args: argparse.Namespace) -> bool:
    return not args.streaming and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _resolve_shard_strategy(args: argparse.Namespace, *, is_map_style: bool) -> str:
    strategy = args.shard_strategy
    if strategy == "auto":
        return "contiguous" if is_map_style else "dataset"
    return strategy


def _iter_rank_examples(
    dataset,
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Iterable[_PreparedItem | tuple[int, dict[str, Any]]]:
    start = max(0, int(args.skip_samples))
    is_map_style = _is_map_style_dataset(dataset, args)
    shard_strategy = _resolve_shard_strategy(args, is_map_style=is_map_style)
    if is_map_style:
        ds_len = len(dataset)
        if ds_len <= start:
            return iter(())

        def _iter_map(
            indices: Iterable[int],
        ) -> Iterable[_PreparedItem | tuple[int, dict[str, Any]]]:
            for idx in indices:
                try:
                    sample = dataset[int(idx)]
                except Exception as exc:
                    yield _PreparedItem(
                        idx=int(idx),
                        status="error",
                        skip_reason="dataset_iter_error",
                        error=str(exc),
                    )
                    continue
                yield int(idx), sample

        if shard_strategy == "contiguous":
            total = ds_len - start
            per_rank = int(math.ceil(total / world_size))
            shard_start = start + (rank * per_rank)
            shard_end = min(ds_len, shard_start + per_rank)
            return _iter_map(range(shard_start, shard_end))
        first = _first_index_for_rank(start, rank, world_size)
        return _iter_map(range(first, ds_len, world_size))

    def _iter_stream() -> Iterable[_PreparedItem | tuple[int, dict[str, Any]]]:
        if shard_strategy in ("dataset", "contiguous") and hasattr(dataset, "shard"):
            try:
                sharded = dataset.shard(num_shards=world_size, index=rank)
                for idx, sample in enumerate(sharded):
                    if idx < start:
                        continue
                    yield idx, sample
                return
            except Exception:
                pass
        for idx, sample in enumerate(dataset):
            if idx < start:
                continue
            if idx % world_size != rank:
                continue
            yield idx, sample

    return _iter_stream()


def _ranked_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return path
    width = max(2, len(str(world_size - 1)))
    suffix = f".rank{rank:0{width}d}"
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _merge_shards(base_path: Path, world_size: int, *, keep_shards: bool) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with base_path.open("w", encoding="utf-8") as out_f:
        for rank in range(world_size):
            shard_path = _ranked_path(base_path, rank, world_size)
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
    if not keep_shards:
        for rank in range(world_size):
            shard_path = _ranked_path(base_path, rank, world_size)
            if shard_path.exists():
                shard_path.unlink()


def _resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _run_worker(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    if world_size > 1 and not str(args.device).startswith("cuda"):
        raise RuntimeError("Multi-GPU mode requires CUDA/ROCm backend; use --device cuda.")

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but not available.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)

    output_manifest = _ranked_path(
        Path(args.output_manifest).expanduser().resolve(), rank, world_size
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_base = output_manifest.parent

    latent_dir = Path(args.latent_dir).expanduser().resolve()
    latent_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        data_files=_parse_data_files(args.data_files),
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        streaming=args.streaming,
    )

    if args.audio_column not in ds.column_names:
        raise ValueError(f"audio column '{args.audio_column}' not found: {ds.column_names}")
    if args.text_column not in ds.column_names:
        raise ValueError(f"text column '{args.text_column}' not found: {ds.column_names}")
    if args.caption_column is not None and args.caption_column not in ds.column_names:
        raise ValueError(f"caption column '{args.caption_column}' not found: {ds.column_names}")
    if args.speaker_columns:
        missing_speaker_columns = [c for c in args.speaker_columns if c not in ds.column_names]
        if missing_speaker_columns:
            raise ValueError(
                f"speaker column(s) not found: {missing_speaker_columns}; available={ds.column_names}"
            )

    if args.target_sample_rate is not None:
        ds = ds.cast_column(args.audio_column, Audio(sampling_rate=args.target_sample_rate))
    else:
        ds = ds.cast_column(args.audio_column, Audio())

    if args.normalize_db is not None:
        try:
            from audiotools import AudioSignal

            del AudioSignal
        except Exception as exc:
            raise RuntimeError(
                "--normalize-db requires audiotools. Install audiotools or set --normalize-db none."
            ) from exc

    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=str(device),
        deterministic_encode=bool(args.codec_deterministic_encode),
        deterministic_decode=bool(args.codec_deterministic_decode),
        normalize_db=args.normalize_db,
    )

    start = max(0, int(args.skip_samples))
    total: int | None = None
    is_map_style = _is_map_style_dataset(ds, args)
    shard_strategy = _resolve_shard_strategy(args, is_map_style=is_map_style)
    if is_map_style:
        total = len(ds) - start
        if total < 0:
            total = 0
        if world_size > 1:
            if shard_strategy == "contiguous":
                total = _count_rank_items_contiguous(start, len(ds), rank, world_size)
            else:
                total = _count_rank_items(start, len(ds), rank, world_size)

    written = 0
    seen = 0
    skip_counts: dict[str, int] = {}
    rank_prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""
    rank_width = max(2, len(str(world_size - 1)))
    show_progress = bool(args.progress) and (world_size == 1 or args.progress_all or rank == 0)
    desc = "Precompute latents" if world_size == 1 else f"Precompute [rank {rank}/{world_size}]"
    pbar = tqdm(
        total=total,
        desc=desc,
        unit="utt",
        disable=not show_progress,
        position=rank if args.progress_all else 0,
        dynamic_ncols=True,
    )

    def _inc_skip(reason: str | None) -> None:
        key = reason or "unknown"
        skip_counts[key] = skip_counts.get(key, 0) + 1

    def _log_progress() -> None:
        if args.log_every <= 0:
            return
        if seen <= 0 or seen % args.log_every != 0:
            return
        skipped_empty = skip_counts.get("empty_text", 0)
        skipped_speaker = skip_counts.get("missing_speaker", 0)
        skipped_low_sr = skip_counts.get("low_sample_rate", 0)
        skipped_audio = sum(
            skip_counts.get(k, 0)
            for k in (
                "audio_decode",
                "trimmed_empty",
                "prepare_error",
                "encode_error",
                "dataset_iter_error",
            )
        )
        skipped_max = skip_counts.get("max_samples_limit", 0)
        total_msg = f"/{total}" if total is not None else ""
        message = (
            f"{rank_prefix}seen={seen}{total_msg} written={written} "
            f"skipped_empty={skipped_empty} "
            f"skipped_speaker={skipped_speaker} "
            f"skipped_audio={skipped_audio} skipped_low_sr={skipped_low_sr} "
            f"skipped_max={skipped_max}"
        )
        if show_progress:
            pbar.set_postfix(
                {
                    "written": written,
                    "skip": (
                        skipped_empty
                        + skipped_speaker
                        + skipped_audio
                        + skipped_low_sr
                        + skipped_max
                    ),
                },
                refresh=False,
            )
        else:
            print(message)

    def _handle_item(item: _PreparedItem, *, stop_requested: bool, out_f) -> None:
        nonlocal seen, written
        seen += 1
        if show_progress:
            pbar.update(1)

        if item.status == "skip":
            _inc_skip(item.skip_reason)
            _log_progress()
            return
        if item.status == "error":
            _inc_skip(item.skip_reason or "prepare_error")
            _log_progress()
            return
        if stop_requested:
            _inc_skip("max_samples_limit")
            _log_progress()
            return

        wav = item.wav
        sr = item.sample_rate
        text = item.text
        caption = item.caption
        speaker_id = item.speaker_id
        if wav is None or sr is None or text is None:
            _inc_skip("prepare_error")
            _log_progress()
            return

        try:
            with torch.inference_mode():
                latent = codec.encode_waveform(wav, sample_rate=sr)[0].cpu()
        except Exception:
            _inc_skip("encode_error")
            _log_progress()
            return

        if world_size > 1:
            latent_name = f"rank{rank:0{rank_width}d}_{written:08d}_{item.idx:08d}.pt"
        else:
            latent_name = f"{written:08d}_{item.idx:08d}.pt"
        latent_path = (latent_dir / latent_name).resolve()
        torch.save(latent, latent_path)
        latent_rel = os.path.relpath(latent_path, start=manifest_base)
        payload = {
            "text": text,
            "latent_path": latent_rel,
            "num_frames": int(latent.shape[0]),
        }
        if caption is not None:
            payload["caption"] = caption
        if speaker_id is not None:
            payload["speaker_id"] = speaker_id
        out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        written += 1
        if args.flush_every > 0 and written % args.flush_every == 0:
            out_f.flush()
        _log_progress()

    iter_items = _iter_rank_examples(ds, args=args, rank=rank, world_size=world_size)
    with output_manifest.open("w", encoding="utf-8") as out_f:
        try:
            if args.prefetch > 0:
                queue, stop_event, thread = _start_prefetch(iter_items, args)
                stop_requested = False
                end_needed = max(1, int(getattr(args, "prefetch_workers", 1)))
                while True:
                    queued = queue.get()
                    if queued is _END:
                        end_needed -= 1
                        if end_needed <= 0:
                            break
                        continue

                    if args.max_samples is not None and written >= args.max_samples:
                        stop_requested = True
                        stop_event.set()
                    _handle_item(queued, stop_requested=stop_requested, out_f=out_f)
                thread.join()
            else:
                for entry in iter_items:
                    if args.max_samples is not None and written >= args.max_samples:
                        break
                    if isinstance(entry, _PreparedItem):
                        _handle_item(entry, stop_requested=False, out_f=out_f)
                        continue
                    idx, sample = entry
                    item = _prepare_example(idx, sample, args)
                    _handle_item(item, stop_requested=False, out_f=out_f)
        finally:
            out_f.flush()
            pbar.close()

    skipped_empty = skip_counts.get("empty_text", 0)
    skipped_speaker = skip_counts.get("missing_speaker", 0)
    skipped_low_sr = skip_counts.get("low_sample_rate", 0)
    skipped_max = skip_counts.get("max_samples_limit", 0)
    skipped_audio = sum(
        skip_counts.get(k, 0)
        for k in (
            "audio_decode",
            "trimmed_empty",
            "prepare_error",
            "encode_error",
            "dataset_iter_error",
        )
    )
    print(
        f"{rank_prefix}done. seen={seen} written={written} "
        f"skipped_empty={skipped_empty} "
        f"skipped_speaker={skipped_speaker} "
        f"skipped_audio={skipped_audio} skipped_low_sr={skipped_low_sr} "
        f"skipped_max={skipped_max} manifest={output_manifest}"
    )
    if skip_counts:
        print(f"{rank_prefix}skip breakdown:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"{rank_prefix}  {reason}: {count}")


def _spawn_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    _run_worker(args, rank=rank, world_size=world_size, local_rank=rank)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DACVAE latents directly from a Hugging Face dataset "
            "(without saving intermediate audio files)."
        )
    )
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. myorg/my_dataset")
    parser.add_argument("--config", default=None, help="HF dataset config/subset")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--data-files",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Optional data_files for load_dataset. "
            "Accepts paths/globs or split-qualified entries like train=data/train.jsonl."
        ),
    )
    parser.add_argument("--audio-column", required=True, help="Audio column name")
    parser.add_argument("--text-column", required=True, help="Text column name")
    parser.add_argument(
        "--text-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply irodori_tts text normalization before writing manifest text. "
            "Use --no-text-normalize to keep raw text."
        ),
    )
    parser.add_argument(
        "--caption-column",
        default=None,
        help="Optional caption/style-control text column name. Output manifest key is always 'caption'.",
    )
    parser.add_argument(
        "--speaker-column",
        action="append",
        default=None,
        help=(
            "Optional speaker/source column name. Can be specified multiple times "
            "or as a comma-separated list (e.g. --speaker-column speaker,source). "
            "If set, output manifest will include speaker_id built from dataset namespace + column value(s)."
        ),
    )
    parser.add_argument(
        "--speaker-id-prefix",
        default=None,
        help=(
            "Optional namespace prefix for speaker_id. Default is dataset name (+ config when set)."
        ),
    )
    parser.add_argument(
        "--output-manifest", required=True, help="Output JSONL path for latent manifest"
    )
    parser.add_argument("--latent-dir", required=True, help="Directory to write latent .pt files")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument(
        "--codec-deterministic-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE encode path (default: enabled).",
    )
    parser.add_argument(
        "--codec-deterministic-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE decode path (default: enabled).",
    )
    parser.add_argument(
        "--normalize-db",
        type=parse_optional_float,
        default=-16.0,
        help=(
            "Target loudness normalization in dB before encode (DAC-like). "
            "Set to 'none' to disable. Default: -16."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for local multiprocessing (spawns one process per GPU).",
    )
    parser.add_argument(
        "--shard-strategy",
        type=str,
        default="auto",
        choices=("auto", "stride", "contiguous", "dataset"),
        help=(
            "How to split samples across ranks. "
            "'auto' uses contiguous shards for map-style datasets and "
            "dataset.shard() for iterable datasets; "
            "'stride' keeps modulo-based split."
        ),
    )
    parser.add_argument(
        "--merge-output",
        action="store_true",
        help="Merge per-rank output manifests after a --num-gpus run.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep per-rank manifest shards when --merge-output is used.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Load dataset in streaming mode.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Optional decode sample rate",
    )
    parser.add_argument(
        "--min-sample-rate",
        type=int,
        default=0,
        help=(
            "Skip samples whose decoded sample rate is below this threshold. "
            "Default: 0 (disabled). Set to e.g. 16000 or 44100 to enable."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional trim duration before encode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max accepted samples to write (per-rank in multi-GPU mode)",
    )
    parser.add_argument("--skip-samples", type=int, default=0, help="Skip first N source samples")
    parser.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help="Prefetch queue size for CPU-side preparation (0 disables).",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=1,
        help="Number of CPU worker threads for prefetch preparation.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help="Flush manifest output every N written records (0 disables periodic flush).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all",
        action="store_true",
        help="Show tqdm bars for all ranks in multi-GPU mode (default: rank0 only).",
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code to datasets.load_dataset",
    )
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    args = parser.parse_args()
    if args.flush_every < 0:
        raise ValueError("--flush-every must be >= 0.")
    args.speaker_columns = _parse_speaker_columns(args.speaker_column)
    args.speaker_id_namespace = _resolve_speaker_namespace(args)

    flat_data_files: list[str] | None = None
    if args.data_files:
        flat_data_files = []
        for group in args.data_files:
            flat_data_files.extend(group)
    args.data_files = flat_data_files

    rank, world_size, local_rank = _resolve_dist_env()
    if world_size > 1:
        if args.merge_output and rank == 0:
            print("Note: --merge-output is ignored under torchrun. Merge shard manifests manually.")
        _run_worker(args, rank=rank, world_size=world_size, local_rank=local_rank)
        return

    num_gpus = int(args.num_gpus) if args.num_gpus is not None else 1
    if num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1.")
    if num_gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but not available.")
        available = torch.cuda.device_count()
        if num_gpus > available:
            raise ValueError(f"Requested {num_gpus} GPUs, but only {available} are available.")
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        mp.spawn(_spawn_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
        if args.merge_output:
            _merge_shards(
                Path(args.output_manifest).expanduser().resolve(),
                num_gpus,
                keep_shards=args.keep_shards,
            )
        return

    _run_worker(args, rank=0, world_size=1, local_rank=0)


if __name__ == "__main__":
    main()
