#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.platform import auto_enable_mps_fallback

_AUTO_ENABLED_MPS_FALLBACK = auto_enable_mps_fallback()

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_codec_runtime_device,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)

FIXED_SECONDS = 30.0


def _parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected float or one of [none, null, off, disable, disabled]."
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"Expected finite float for value={value!r}.")
    return out


def _print_timings(timings: list[tuple[str, float]], total_to_decode: float) -> None:
    print("[timing] ---- post-model-load to decode ----")
    for name, sec in timings:
        print(f"[timing] {name}: {sec * 1000.0:.1f} ms")
    print(f"[timing] total_to_decode: {total_to_decode:.3f} s")


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        checkpoint_path = Path(str(args.checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    repo_id = str(args.hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for Irodori-TTS.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help=(
            "Hugging Face model repo id to download model.safetensors from "
            "(e.g. your-org/your-model)."
        ),
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", default="output.wav")
    parser.add_argument(
        "--model-device",
        default=default_runtime_device(),
        help="Model inference device (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-device",
        default=default_codec_runtime_device(),
        help="Codec device for reference encode/decode (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Codec precision for weights/compute.",
    )
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
        help="Use deterministic DACVAE decode watermark-message path (default: enabled).",
    )
    parser.add_argument(
        "--enable-watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DACVAE watermark branch during decode (default: disabled).",
    )
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=30.0,
        help="Maximum reference duration in seconds. Set <=0 to disable the cap.",
    )
    parser.add_argument(
        "--ref-normalize-db",
        type=_parse_optional_float,
        default=-16.0,
        help=(
            "Target loudness (dB/LUFS-like) for reference audio before DACVAE encode "
            "(e.g. -16.0). Set to 'none' to disable. Default: -16."
        ),
    )
    parser.add_argument(
        "--ref-ensure-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scale reference audio down only when peak exceeds 1.0 after optional loudness "
            "normalization. Effective only when --ref-normalize-db is none/null/off "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=None,
        help=(
            "Maximum token length for text conditioning. "
            "Defaults to checkpoint metadata max_text_len when available, else 256."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of candidates to generate in a single batched sampling pass.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=["sequential", "batch"],
        default="sequential",
        help=(
            "Codec decode mode. "
            "'sequential': decode each candidate one-by-one (lower VRAM), "
            "'batch': decode all candidates at once (faster, higher VRAM)."
        ),
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for core inference methods (default: disabled).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic=True for torch.compile (default: disabled).",
    )
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
        help=(
            "CFG formulation. "
            "'independent': text/speaker unconds separately (3x NFE when both enabled), "
            "'joint': drop both conditions together (2x NFE), "
            "'alternating': alternate text and speaker uncond each step (2x NFE)."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Deprecated. If set, overrides both --cfg-scale-text and --cfg-scale-speaker.",
    )
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument(
        "--truncation-factor",
        type=float,
        default=None,
        help=(
            "Scale initial Gaussian noise before Euler sampling "
            "(e.g., 0.8 flat / 0.9 sharp). Default: disabled."
        ),
    )
    parser.add_argument(
        "--rescale-k",
        type=float,
        default=None,
        help=(
            "Temporal score rescaling k (Xu et al., 2025). "
            "Set together with --rescale-sigma. Default: disabled."
        ),
    )
    parser.add_argument(
        "--rescale-sigma",
        type=float,
        default=None,
        help=(
            "Temporal score rescaling sigma (Xu et al., 2025). "
            "Set together with --rescale-k. Default: disabled."
        ),
    )
    parser.add_argument(
        "--context-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute per-layer text/speaker context K/V projections for faster sampling "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--speaker-kv-scale",
        type=float,
        default=None,
        help=(
            "Force-speaker mode: scale speaker K/V projections by this factor (>1 strengthens speaker identity). "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--speaker-kv-min-t",
        type=float,
        default=0.9,
        help=(
            "Disable speaker KV scaling after crossing this timestep threshold "
            "(applies while t >= value). Default: 0.9."
        ),
    )
    parser.add_argument(
        "--speaker-kv-max-layers",
        type=int,
        default=None,
        help="Apply speaker KV scaling only to first N diffusion layers (default: all layers).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. If omitted, a random seed is generated per request.",
    )
    parser.add_argument(
        "--trim-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim trailing near-zero latent region with Echo-style flattening heuristic "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--tail-window-size", type=int, default=20)
    parser.add_argument("--tail-std-threshold", type=float, default=0.05)
    parser.add_argument("--tail-mean-threshold", type=float, default=0.1)
    parser.add_argument(
        "--show-timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print per-stage timings from post-model-load through latent decode (default: enabled)."
        ),
    )
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--ref-wav", default=None, help="Reference waveform path for speaker conditioning."
    )
    ref_group.add_argument(
        "--ref-latent", default=None, help="Reference latent (.pt) path for speaker conditioning."
    )
    ref_group.add_argument(
        "--no-ref", action="store_true", help="Run without speaker reference conditioning."
    )
    args = parser.parse_args()

    if _AUTO_ENABLED_MPS_FALLBACK:
        print(
            "[runtime] enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for macOS compatibility.",
            flush=True,
        )

    cfg_scale_text, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_speaker=float(args.cfg_scale_speaker),
        cfg_scale=float(args.cfg_scale) if args.cfg_scale is not None else None,
    )
    for msg in scale_messages:
        print(msg)

    checkpoint_path = _resolve_checkpoint_path(args)

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(args.model_device),
            codec_repo=str(args.codec_repo),
            model_precision=str(args.model_precision),
            codec_device=str(args.codec_device),
            codec_precision=str(args.codec_precision),
            codec_deterministic_encode=bool(args.codec_deterministic_encode),
            codec_deterministic_decode=bool(args.codec_deterministic_decode),
            enable_watermark=bool(args.enable_watermark),
            compile_model=bool(args.compile_model),
            compile_dynamic=bool(args.compile_dynamic),
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=str(args.text),
            ref_wav=args.ref_wav,
            ref_latent=args.ref_latent,
            no_ref=bool(args.no_ref),
            ref_normalize_db=args.ref_normalize_db,
            ref_ensure_max=bool(args.ref_ensure_max),
            num_candidates=int(args.num_candidates),
            decode_mode=str(args.decode_mode),
            seconds=FIXED_SECONDS,
            max_ref_seconds=float(args.max_ref_seconds)
            if args.max_ref_seconds is not None
            else None,
            max_text_len=None if args.max_text_len is None else int(args.max_text_len),
            num_steps=int(args.num_steps),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=str(args.cfg_guidance_mode),
            cfg_scale=None,
            cfg_min_t=float(args.cfg_min_t),
            cfg_max_t=float(args.cfg_max_t),
            truncation_factor=None
            if args.truncation_factor is None
            else float(args.truncation_factor),
            rescale_k=None if args.rescale_k is None else float(args.rescale_k),
            rescale_sigma=None if args.rescale_sigma is None else float(args.rescale_sigma),
            context_kv_cache=bool(args.context_kv_cache),
            speaker_kv_scale=None
            if args.speaker_kv_scale is None
            else float(args.speaker_kv_scale),
            speaker_kv_min_t=None
            if args.speaker_kv_scale is None
            else float(args.speaker_kv_min_t),
            speaker_kv_max_layers=None
            if args.speaker_kv_max_layers is None
            else int(args.speaker_kv_max_layers),
            seed=None if args.seed is None else int(args.seed),
            trim_tail=bool(args.trim_tail),
            tail_window_size=int(args.tail_window_size),
            tail_std_threshold=float(args.tail_std_threshold),
            tail_mean_threshold=float(args.tail_mean_threshold),
        ),
        log_fn=None,
    )

    print(f"[seed] used_seed: {result.used_seed}")
    if int(args.num_candidates) == 1:
        out_path = save_wav(args.output_wav, result.audio, result.sample_rate)
        print(f"Saved: {out_path}")
    else:
        base_path = Path(str(args.output_wav))
        suffix = base_path.suffix if base_path.suffix else ".wav"
        for i, audio in enumerate(result.audios, start=1):
            out_path = base_path.with_name(f"{base_path.stem}_{i:03d}{suffix}")
            saved = save_wav(out_path, audio, result.sample_rate)
            print(f"Saved[{i}]: {saved}")
    if args.show_timings:
        _print_timings(result.stage_timings, result.total_to_decode)


if __name__ == "__main__":
    main()
