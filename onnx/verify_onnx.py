"""End-to-end verification of the exported ONNX pipeline against PyTorch.

Runs the full synthesis twice for the same text/caption/seed:

  1. Reference: the PyTorch InferenceRuntime.
  2. ONNX: tokenize -> encoder.onnx -> a pure numpy Euler/CFG sampler
     -> dacvae_decoder.onnx (no torch in the synthesis path).

and reports max|diff| / correlation between the two waveforms. The numpy
sampler in this file is also the reference implementation for porting the
pipeline to other runtimes (mobile, etc.): sway t-schedule, independent CFG
inside the [0.5, 1.0] time window, Euler integration.

Notes for porting:
  - `normalize_text` must be applied before tokenization (it changes token
    ids, e.g. full-width punctuation) and a BOS id is prepended by the
    tokenizer wrapper. Skipping it destroys output quality.
  - The v2-VoiceDesign checkpoint has no duration predictor; `--seconds`
    sizes the latent sequence (25 latent frames per second).
  - Euler needs enough steps to converge: 8 matches the reference quality,
    5-6 is a usable fast setting, 4 and below is audibly degraded.

Usage:
  python onnx/verify_onnx.py --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
      --onnx-dir onnx_out --text "こんにちは。" --caption "落ち着いた大人の女性の声"
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)
from irodori_tts.text_normalization import normalize_text  # noqa: E402

LATENT_HZ = 25  # DACVAE latent frame rate (hop 1920 @ 48 kHz)
CFG_WINDOW = (0.5, 1.0)  # CFG is applied only inside this t range


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        path = os.path.expanduser(args.checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")


def sway_t_schedule(num_steps: int, sway: float = -1.0) -> np.ndarray:
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = np.clip(u + sway * (np.cos(0.5 * np.pi * u) + u - 1.0), 0.0, 1.0)
    return (1.0 - u) * 0.999


def sample_onnx(
    dit_session,
    decoder_session,
    text_state: np.ndarray,
    text_mask: np.ndarray,
    caption_state: np.ndarray,
    caption_mask: np.ndarray,
    seconds: float,
    num_steps: int,
    cfg_text: float,
    cfg_caption: float,
    seed: int,
) -> np.ndarray:
    """Pure numpy/ONNX synthesis: noise -> Euler/CFG -> DACVAE decode."""
    S = round(seconds * LATENT_HZ)
    latent_dim = dit_session.get_inputs()[0].shape[-1]  # x_t: (B, S, latent_dim)
    generator = torch.Generator().manual_seed(seed)  # match runtime seeding
    x = torch.randn((1, S, latent_dim), generator=generator).numpy().astype(np.float32)

    text_uncond = np.zeros_like(text_state)
    text_mask_uncond = np.zeros_like(text_mask)
    caption_uncond = np.zeros_like(caption_state)
    caption_mask_uncond = np.zeros_like(caption_mask)

    def dit(x_np, t_scalar, ts, tm, cs, cm):
        return dit_session.run(
            None,
            {
                "x_t": x_np,
                "t": np.array([t_scalar], np.float32),
                "text_state": ts,
                "text_mask": tm,
                "caption_state": cs,
                "caption_mask": cm,
            },
        )[0]

    t_schedule = sway_t_schedule(num_steps)
    for i in range(num_steps):
        t, t_next = float(t_schedule[i]), float(t_schedule[i + 1])
        if CFG_WINDOW[0] <= t <= CFG_WINDOW[1]:
            v_cond = dit(x, t, text_state, text_mask, caption_state, caption_mask)
            v_text_uncond = dit(x, t, text_uncond, text_mask_uncond, caption_state, caption_mask)
            v_caption_uncond = dit(x, t, text_state, text_mask, caption_uncond, caption_mask_uncond)
            v = v_cond + cfg_text * (v_cond - v_text_uncond) + cfg_caption * (v_cond - v_caption_uncond)
        else:
            v = dit(x, t, text_state, text_mask, caption_state, caption_mask)
        x = x + v * (t_next - t)

    z = np.transpose(x, (0, 2, 1)).astype(np.float32)  # (1, latent_dim, S)
    return decoder_session.run(None, {"z": z})[0].squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the exported ONNX pipeline.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", default=None, help="Local checkpoint path.")
    checkpoint_group.add_argument("--hf-checkpoint", default=None, help="HF repo id for model.safetensors.")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--onnx-dir", default="onnx_out", help="Directory with the exported .onnx files.")
    parser.add_argument("--text", default="こんにちは、音声合成のテストです。")
    parser.add_argument("--caption", default="落ち着いた大人の女性の声")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--cfg-text", type=float, default=3.0)
    parser.add_argument("--cfg-caption", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-corr", type=float, default=0.999, help="Fail below this correlation.")
    parser.add_argument("--output-prefix", default="verify", help="Prefix for the two output wav files.")
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device="cpu",
            model_precision="fp32",
            codec_repo=args.codec_repo,
            codec_device="cpu",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )

    # ---- reference (PyTorch runtime) ----
    reference = runtime.synthesize(
        SamplingRequest(
            text=args.text,
            caption=args.caption,
            no_ref=True,
            num_steps=args.num_steps,
            seconds=args.seconds,
            cfg_scale_text=args.cfg_text,
            cfg_scale_caption=args.cfg_caption,
            cfg_guidance_mode="independent",
            decode_mode="batch",
            context_kv_cache=False,
            t_schedule_mode="sway",
            seed=args.seed,
        )
    )
    reference_audio = reference.audios[0].squeeze().float().numpy()

    # ---- ONNX pipeline ----
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, os.cpu_count() // 2)
    providers = ["CPUExecutionProvider"]
    encoder_session = ort.InferenceSession(
        os.path.join(args.onnx_dir, "encoder.onnx"), session_options, providers=providers
    )
    dit_session = ort.InferenceSession(
        os.path.join(args.onnx_dir, "dit.onnx"), session_options, providers=providers
    )
    decoder_session = ort.InferenceSession(
        os.path.join(args.onnx_dir, "dacvae_decoder.onnx"), session_options, providers=providers
    )

    tokenizer = runtime.tokenizer
    caption_tokenizer = runtime.caption_tokenizer or runtime.tokenizer
    text_ids, text_mask = tokenizer.batch_encode([normalize_text(args.text)], 64)
    caption_ids, caption_mask = caption_tokenizer.batch_encode([normalize_text(args.caption)], 64)

    text_state, caption_state = encoder_session.run(
        None,
        {
            "text_ids": text_ids.numpy().astype(np.int64),
            "text_mask": text_mask.numpy(),
            "caption_ids": caption_ids.numpy().astype(np.int64),
            "caption_mask": caption_mask.numpy(),
        },
    )
    onnx_audio = sample_onnx(
        dit_session,
        decoder_session,
        text_state,
        text_mask.numpy(),
        caption_state,
        caption_mask.numpy(),
        seconds=args.seconds,
        num_steps=args.num_steps,
        cfg_text=args.cfg_text,
        cfg_caption=args.cfg_caption,
        seed=args.seed,
    )

    # ---- compare ----
    n = min(len(onnx_audio), len(reference_audio))
    max_diff = float(np.abs(onnx_audio[:n] - reference_audio[:n]).max())
    corr = float(np.corrcoef(onnx_audio[:n], reference_audio[:n])[0, 1])
    print(f"[verify] samples={n}  max|diff|={max_diff:.2e}  corr={corr:.6f}")

    sample_rate = runtime.codec.sample_rate
    reference_path = save_wav(f"{args.output_prefix}_pytorch.wav", torch.from_numpy(reference_audio)[None, :], sample_rate)
    onnx_path = save_wav(f"{args.output_prefix}_onnx.wav", torch.from_numpy(onnx_audio)[None, :], sample_rate)
    print(f"[verify] wrote {reference_path} and {onnx_path}")

    if corr < args.min_corr:
        print(f"[verify] FAIL: corr {corr:.6f} < {args.min_corr}")
        raise SystemExit(1)
    print("[verify] PASS")


if __name__ == "__main__":
    main()
