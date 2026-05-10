#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)
from irodori_tts.waveex import WaveExConfig, available_wavelets, parse_ode_step_indices

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
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption/style-control text for caption-enabled voice-design checkpoints.",
    )
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
        default=default_runtime_device(),
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
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=None,
        help=(
            "Maximum token length for caption conditioning. "
            "Defaults to checkpoint metadata max_caption_len when available, else max_text_len."
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
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
        help=(
            "CFG formulation. "
            "'independent': each enabled condition uses its own uncond pass, "
            "'joint': drop all enabled conditions together (2x NFE), "
            "'alternating': alternate enabled condition unconds each step."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Deprecated. If set, overrides --cfg-scale-text/--cfg-scale-caption/--cfg-scale-speaker.",
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
        "--waveex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable WaveEx wavelet-guided extrapolation acceleration "
            "(Liu et al., AAAI 2026). Default: disabled."
        ),
    )
    parser.add_argument(
        "--waveex-ode-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated step indices where the model is evaluated normally. "
            "All other indices are predicted via wavelet-guided extrapolation. "
            "Use 'auto' (default) to derive a schedule from the calibrated "
            "default (0,1,2,5,10,20 at 40 NFE) rescaled to --num-steps."
        ),
    )
    parser.add_argument(
        "--waveex-wavelet",
        choices=available_wavelets(),
        default="haar",
        help=(
            "Wavelet basis used for the multi-scale decomposition. Default: haar "
            "(longer wavelets like sym6 require larger --waveex-history-size to "
            "avoid periodic-boundary distortion)."
        ),
    )
    parser.add_argument(
        "--waveex-taylor-order",
        type=int,
        choices=[1, 2],
        default=1,
        help="Order of Taylor extrapolation in each band. Default: 1 (paper).",
    )
    parser.add_argument(
        "--waveex-history-size",
        type=int,
        default=2,
        help=(
            "Number of past latent states retained for the extrapolation window. "
            "Default: 2 (uses direct Taylor on raw latents — empirically best on "
            "the duration-control checkpoint; the wavelet path with haar + small "
            "history is unstable). Set >= 4 to engage the wavelet decomposition."
        ),
    )
    parser.add_argument(
        "--waveex-high-freq-mode",
        choices=["extrapolate", "freeze", "zero"],
        default="extrapolate",
        help="How to handle the high-frequency band during reconstruction.",
    )
    parser.add_argument(
        "--show-timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print per-stage timings from post-model-load through latent decode (default: enabled)."
        ),
    )
    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument(
        "--ref-wav", default=None, help="Reference waveform path for speaker conditioning."
    )
    ref_group.add_argument(
        "--ref-latent", default=None, help="Reference latent (.pt) path for speaker conditioning."
    )
    ref_group.add_argument(
        "--no-ref",
        action="store_true",
        help="Run without speaker reference conditioning. Use this for voice-design checkpoints.",
    )
    args = parser.parse_args()

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
    if runtime.model_cfg.use_speaker_condition and not (
        args.no_ref or args.ref_wav is not None or args.ref_latent is not None
    ):
        parser.error(
            "speaker-conditioned checkpoints require one of --ref-wav, --ref-latent, or --no-ref."
        )
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_caption=float(args.cfg_scale_caption),
        cfg_scale_speaker=float(args.cfg_scale_speaker),
        cfg_scale=float(args.cfg_scale) if args.cfg_scale is not None else None,
        use_caption_condition=bool(
            runtime.model_cfg.use_caption_condition
            and args.caption is not None
            and str(args.caption).strip() != ""
        ),
        use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
    )
    for msg in scale_messages:
        print(msg)

    waveex_cfg: WaveExConfig | None = None
    if bool(args.waveex):
        waveex_cfg = WaveExConfig(
            enabled=True,
            ode_step_indices=parse_ode_step_indices(
                args.waveex_ode_steps,
                num_steps=int(args.num_steps),
            ),
            wavelet=str(args.waveex_wavelet),
            taylor_order=int(args.waveex_taylor_order),
            history_size=int(args.waveex_history_size),
            high_freq_mode=str(args.waveex_high_freq_mode),
        )

    result = runtime.synthesize(
        SamplingRequest(
            text=str(args.text),
            caption=None if args.caption is None else str(args.caption),
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
            max_caption_len=None if args.max_caption_len is None else int(args.max_caption_len),
            num_steps=int(args.num_steps),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
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
            waveex=waveex_cfg,
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
