from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .codec import patchify_latent
from .duration import build_duration_features
from .tokenizer import PretrainedTextTokenizer

_MANIFEST_INDEX_CACHE_VERSION = 2


def _caption_candidates(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        candidates: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                candidates.append(text)
        return candidates
    text = str(raw).strip()
    return [text] if text else []


def _has_caption(raw: Any) -> bool:
    return bool(_caption_candidates(raw))


def _select_caption(raw: Any) -> str:
    candidates = _caption_candidates(raw)
    if not candidates:
        return ""
    return random.choice(candidates)


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """
    Normalize latent tensor to (T, D).
    Accepts common layouts: (T, D), (D, T), (1, T, D), (1, D, T).
    """
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")

    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


class LatentTextDataset(Dataset):
    """
    Manifest format (JSONL), one sample per line:
      {"text": "...", "latent_path": "path/to/latent.pt", "speaker_id": "..."}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        latent_dim: int,
        max_latent_steps: int | None = None,
        subset_indices: list[int] | None = None,
        enable_caption_condition: bool = False,
        enable_speaker_condition: bool = True,
        caption_key: str = "caption",
        manifest_index: _ManifestIndex | None = None,
        show_manifest_progress: bool = False,
        manifest_progress_desc: str | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.latent_dim = latent_dim
        self.max_latent_steps = max_latent_steps
        self.enable_caption_condition = bool(enable_caption_condition)
        self.enable_speaker_condition = bool(enable_speaker_condition)
        self.caption_key = str(caption_key)
        self._manifest_fp = None
        subset_index_set: set[int] | None = None
        if subset_indices is not None:
            subset_index_set = {int(x) for x in subset_indices}
            if not subset_index_set:
                raise ValueError("subset_indices must contain at least one index.")

        if manifest_index is None:
            manifest_index = _ManifestIndex.build(
                manifest_path=self.manifest_path,
                caption_key=self.caption_key,
                show_progress=show_manifest_progress,
                progress_desc=manifest_progress_desc,
            )
        elif manifest_index.caption_key != self.caption_key:
            raise ValueError(
                "manifest_index caption_key mismatch: "
                f"expected {self.caption_key!r}, got {manifest_index.caption_key!r}"
            )
        self.manifest_index = manifest_index

        if subset_index_set is None:
            self.sample_indices = list(range(len(self.manifest_index.offsets)))
        else:
            max_index = len(self.manifest_index.offsets) - 1
            invalid_indices = sorted(x for x in subset_index_set if x < 0 or x > max_index)
            if invalid_indices:
                raise ValueError(
                    f"subset_indices contain out-of-range values for manifest: {invalid_indices[:8]}"
                )
            self.sample_indices = sorted(subset_index_set)

        self.speaker_to_indices: dict[str, list[int]] = {}
        self.speaker_labeled_count = 0
        self.caption_labeled_count = 0
        for local_index, sample_index in enumerate(self.sample_indices):
            speaker_id = self.manifest_index.speaker_ids[sample_index]
            if speaker_id is not None:
                self.speaker_labeled_count += 1
                if self.enable_speaker_condition:
                    self.speaker_to_indices.setdefault(speaker_id, []).append(local_index)
            if self.manifest_index.has_caption[sample_index]:
                self.caption_labeled_count += 1

        if not self.sample_indices:
            raise ValueError(f"No valid samples in manifest: {self.manifest_path}")

    def _resolve_latent_path(self, latent_path_raw: str) -> Path:
        latent_path = Path(latent_path_raw).expanduser()
        if not latent_path.is_absolute():
            latent_path = (self.manifest_dir / latent_path).resolve()
        return latent_path

    def _load_latent(self, latent_path_raw: str) -> torch.Tensor:
        latent_path = self._resolve_latent_path(latent_path_raw)
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = _coerce_latent_shape(latent, self.latent_dim).float()
        if self.max_latent_steps is not None:
            latent = latent[: self.max_latent_steps]
        return latent

    def _manifest_file(self):
        if self._manifest_fp is None or self._manifest_fp.closed:
            self._manifest_fp = self.manifest_path.open("r", encoding="utf-8")
        return self._manifest_fp

    def _read_item(self, index: int) -> dict[str, Any]:
        sample_index = self.sample_indices[index]
        fp = self._manifest_file()
        fp.seek(self.manifest_index.offsets[sample_index])
        line = fp.readline()
        if line == "":
            raise ValueError(
                f"Unexpected EOF while reading manifest sample_index={sample_index}: {self.manifest_path}"
            )
        item = json.loads(line)
        if "text" not in item or "latent_path" not in item:
            raise ValueError(f"Invalid manifest line (needs text and latent_path): {line.rstrip()}")
        return item

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._read_item(index)
        latent = self._load_latent(item["latent_path"])

        ref_index = index
        has_speaker = False
        if self.enable_speaker_condition:
            speaker_id = self.manifest_index.speaker_ids[self.sample_indices[index]]
            candidates = self.speaker_to_indices.get(speaker_id, [])
            if len(candidates) > 1:
                alternatives = [i for i in candidates if i != index]
                if alternatives:
                    ref_index = random.choice(alternatives)
                    has_speaker = True

        if ref_index == index:
            ref_latent = latent
        else:
            ref_item = self._read_item(ref_index)
            ref_latent = self._load_latent(ref_item["latent_path"])
        manifest_num_frames = int(item.get("num_frames", latent.shape[0]))
        num_frames = min(manifest_num_frames, int(latent.shape[0]))
        caption = (
            _select_caption(item.get(self.caption_key)) if self.enable_caption_condition else ""
        )

        return {
            "text": item["text"],
            "caption": caption,
            "has_caption": bool(caption) if self.enable_caption_condition else False,
            "latent": latent,
            "num_frames": num_frames,
            "ref_latent": ref_latent,
            "has_speaker": has_speaker,
        }


@dataclass(frozen=True)
class _ManifestIndex:
    offsets: list[int]
    speaker_ids: list[str | None]
    has_caption: list[bool]
    caption_key: str

    @staticmethod
    def _cache_path(manifest_path: Path, caption_key: str) -> Path:
        suffix = ".irodori_index.pt"
        if caption_key != "caption":
            safe_caption_key = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in caption_key
            )
            suffix = f".{safe_caption_key}.irodori_index.pt"
        return manifest_path.with_name(manifest_path.name + suffix)

    @staticmethod
    def _load_cache(manifest_path: Path, caption_key: str) -> _ManifestIndex | None:
        cache_path = _ManifestIndex._cache_path(manifest_path, caption_key)
        if not cache_path.exists():
            return None
        stat = manifest_path.stat()
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _MANIFEST_INDEX_CACHE_VERSION:
            return None
        if payload.get("manifest_size") != stat.st_size:
            return None
        if payload.get("manifest_mtime_ns") != stat.st_mtime_ns:
            return None
        if payload.get("caption_key") != caption_key:
            return None

        offsets = payload.get("offsets")
        speaker_ids = payload.get("speaker_ids")
        has_caption = payload.get("has_caption")
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
            return None
        if not isinstance(has_caption, torch.Tensor) or has_caption.ndim != 1:
            return None
        if not isinstance(speaker_ids, list):
            return None
        if offsets.numel() != has_caption.numel() or offsets.numel() != len(speaker_ids):
            return None
        return _ManifestIndex(
            offsets=[int(x) for x in offsets.tolist()],
            speaker_ids=[None if x is None else str(x) for x in speaker_ids],
            has_caption=[bool(x) for x in has_caption.tolist()],
            caption_key=str(caption_key),
        )

    def _save_cache(self, manifest_path: Path) -> None:
        cache_path = self._cache_path(manifest_path, self.caption_key)
        stat = manifest_path.stat()
        payload = {
            "version": _MANIFEST_INDEX_CACHE_VERSION,
            "manifest_size": stat.st_size,
            "manifest_mtime_ns": stat.st_mtime_ns,
            "caption_key": self.caption_key,
            "offsets": torch.tensor(self.offsets, dtype=torch.int64),
            "speaker_ids": self.speaker_ids,
            "has_caption": torch.tensor(self.has_caption, dtype=torch.bool),
        }
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(cache_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def build(
        cls,
        manifest_path: Path,
        *,
        caption_key: str = "caption",
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> _ManifestIndex:
        caption_key = str(caption_key)
        cached = cls._load_cache(manifest_path, caption_key)
        if cached is not None:
            if show_progress:
                print(f"Loaded manifest index cache: {cls._cache_path(manifest_path, caption_key)}")
            return cached

        offsets: list[int] = []
        speaker_ids: list[str | None] = []
        has_caption: list[bool] = []
        total_bytes = manifest_path.stat().st_size
        if progress_desc is None:
            progress_desc = f"Index Manifest ({manifest_path.name})"
        pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            desc=progress_desc,
            disable=not show_progress,
            leave=False,
        )
        with manifest_path.open("r", encoding="utf-8") as f:
            try:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if line == "":
                        break
                    pbar.update(len(line.encode("utf-8")))
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "text" not in item or "latent_path" not in item:
                        raise ValueError(
                            f"Invalid manifest line (needs text and latent_path): {line.rstrip()}"
                        )
                    offsets.append(offset)
                    speaker_id = item.get("speaker_id")
                    speaker_ids.append(None if speaker_id is None else str(speaker_id))
                    has_caption.append(_has_caption(item.get(caption_key)))
            finally:
                pbar.close()
        if not offsets:
            raise ValueError(f"No valid samples in manifest: {manifest_path}")
        index = cls(
            offsets=offsets,
            speaker_ids=speaker_ids,
            has_caption=has_caption,
            caption_key=caption_key,
        )
        index._save_cache(manifest_path)
        return index


@dataclass
class TTSCollator:
    tokenizer: PretrainedTextTokenizer
    caption_tokenizer: PretrainedTextTokenizer | None
    latent_dim: int
    latent_patch_size: int
    fixed_target_latent_steps: int | None = None
    fixed_target_full_mask: bool = False
    max_text_len: int = 256
    max_caption_len: int | None = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        captions = [x["caption"] for x in batch]
        latents = [x["latent"] for x in batch]  # each: (T, D)
        ref_latents = [x["ref_latent"] for x in batch]  # each: (T_ref, D)
        has_speaker = torch.tensor([bool(x["has_speaker"]) for x in batch], dtype=torch.bool)
        has_caption = torch.tensor([bool(x["has_caption"]) for x in batch], dtype=torch.bool)
        bsz = len(latents)

        text_ids, text_mask = self.tokenizer.batch_encode(texts, max_length=self.max_text_len)
        token_counts = text_mask.sum(dim=1)
        caption_ids = None
        caption_mask = None
        if self.caption_tokenizer is not None:
            max_caption_len = self.max_caption_len
            if max_caption_len is None:
                max_caption_len = self.max_text_len
            caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                captions,
                max_length=max_caption_len,
            )
            caption_mask = caption_mask & has_caption[:, None]

        if self.fixed_target_latent_steps is not None:
            max_t = int(self.fixed_target_latent_steps)
            if max_t <= 0:
                raise ValueError(
                    f"fixed_target_latent_steps must be > 0, got {self.fixed_target_latent_steps}"
                )
        else:
            max_t = max(x.shape[0] for x in latents)
        latent_batch = torch.zeros((bsz, max_t, self.latent_dim), dtype=torch.float32)
        latent_mask_valid = torch.zeros((bsz, max_t), dtype=torch.bool)
        for i, latent in enumerate(latents):
            n = min(latent.shape[0], max_t)
            latent_batch[i, :n] = latent[:n]
            latent_mask_valid[i, :n] = True
        latent_mask = latent_mask_valid.clone()
        if self.fixed_target_full_mask:
            latent_mask.fill_(True)

        max_ref_t = max(x.shape[0] for x in ref_latents)
        ref_batch = torch.zeros((bsz, max_ref_t, self.latent_dim), dtype=torch.float32)
        ref_mask = torch.zeros((bsz, max_ref_t), dtype=torch.bool)
        for i, ref_latent in enumerate(ref_latents):
            n = ref_latent.shape[0]
            ref_batch[i, :n] = ref_latent
            ref_mask[i, :n] = True

        latent_patched = patchify_latent(latent_batch, self.latent_patch_size)
        # Keep reference in latent-patched space. The model applies an extra
        # speaker_patch_size patching internally for speaker conditioning.
        ref_patched = patchify_latent(ref_batch, self.latent_patch_size)

        def _patch_mask(mask: torch.Tensor) -> torch.Tensor:
            if self.latent_patch_size <= 1:
                return mask
            usable = (mask.shape[1] // self.latent_patch_size) * self.latent_patch_size
            return (
                mask[:, :usable]
                .reshape(bsz, usable // self.latent_patch_size, self.latent_patch_size)
                .all(dim=-1)
            )

        latent_mask_patched = _patch_mask(latent_mask)
        latent_mask_valid_patched = _patch_mask(latent_mask_valid)
        ref_mask_patched = _patch_mask(ref_mask)

        out = {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "num_frames": torch.tensor([int(x["num_frames"]) for x in batch], dtype=torch.long),
            "duration_features": build_duration_features(
                texts,
                token_counts=token_counts,
                max_text_len=self.max_text_len,
                has_speaker=has_speaker,
            ),
            "latent": latent_batch,
            "latent_mask": latent_mask,
            "latent_mask_valid": latent_mask_valid,
            "latent_patched": latent_patched,
            "latent_mask_patched": latent_mask_patched,
            "latent_mask_valid_patched": latent_mask_valid_patched,
            "latent_padding_mask_patched": ~latent_mask_valid_patched,
            "ref_latent": ref_batch,
            "ref_latent_mask": ref_mask,
            "ref_latent_patched": ref_patched,
            "ref_latent_mask_patched": ref_mask_patched,
            "has_speaker": has_speaker,
        }
        if caption_ids is not None and caption_mask is not None:
            out["caption_ids"] = caption_ids
            out["caption_mask"] = caption_mask
            out["has_caption"] = has_caption
        return out
