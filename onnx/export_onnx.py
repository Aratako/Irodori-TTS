"""Export the Irodori-TTS neural components to ONNX for non-PyTorch runtimes.

Exports three graphs (fp32, CPU) and verifies parity against PyTorch:

  - encoder.onnx          text/caption conditioning encoder
                          (text_ids, text_mask, caption_ids, caption_mask)
                          -> (text_state, caption_state)
  - dit.onnx              one RF-DiT velocity step with dynamic batch
                          (x_t, t, text_state, text_mask, caption_state, caption_mask) -> v
                          The batch axis lets a driver run all CFG variants
                          (cond / text-uncond / caption-uncond) in a single call.
  - dacvae_decoder.onnx   DACVAE latent -> 48 kHz waveform
                          (z: B x latent_dim x T) -> (audio: B x 1 x S)
                          NOTE: weights are stored in an external
                          `dacvae_decoder.onnx.data` file that must be kept
                          next to the graph.

The sampler (sway t-schedule, CFG combine, Euler integration) is intentionally
NOT exported -- it is a few lines of driver code. `verify_onnx.py` is the
reference implementation and validates the full pipeline end to end.

ONNX has no complex dtype, so the rotary embedding is monkeypatched to an
equivalent real-valued form for the duration of the export (numerically
identical to the complex implementation; the patch does not affect normal
library usage).

Scope: text+caption conditioning (the VoiceDesign no-reference path). Tested
with Irodori-TTS-500M-v2-VoiceDesign. The reference-audio branch and the v3
duration predictor are not covered.

Usage:
  python onnx/export_onnx.py --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign
  python onnx/export_onnx.py --checkpoint /path/to/model.safetensors --out-dir onnx_out
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import irodori_tts.model as irodori_model  # noqa: E402
from irodori_tts.codec import DACVAECodec  # noqa: E402
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey  # noqa: E402


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        path = os.path.expanduser(args.checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")


class _Encoder(nn.Module):
    """Text + caption conditioning (no-reference VoiceDesign path)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, text_ids, text_mask, caption_ids, caption_mask):
        ts, tm, ss, sm, cs, cm = self.model.encode_conditions(
            text_ids, text_mask, None, None, caption_ids, caption_mask
        )
        return ts, cs


class _DitStep(nn.Module):
    """One velocity prediction v = f(x_t, t, conditions)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_t, t, text_state, text_mask, caption_state, caption_mask):
        return self.model.forward_with_encoded_conditions(
            x_t, t, text_state, text_mask, None, None, caption_state, caption_mask
        )


class _Decoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


def _precompute_freqs_real(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Real-valued equivalent of precompute_freqs_cis: (end, dim/2, 2) = [cos, sin]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _apply_rotary_emb_real(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs (x[..., 0::2], x[..., 1::2]) with the real [cos, sin] table."""
    cos = freqs[..., 0][None, :, None, :]
    sin = freqs[..., 1][None, :, None, :]
    xf = x.float()
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape_as(x)
    return out.type_as(x)


def _patch_rope_for_export(model: nn.Module) -> None:
    with torch.no_grad():
        x = torch.randn(1, 40, 8, 32)
        ref = irodori_model.apply_rotary_emb(x, irodori_model.precompute_freqs_cis(32, 40))
        real = _apply_rotary_emb_real(x, _precompute_freqs_real(32, 40))
        diff = (ref - real).abs().max().item()
    print(f"[rope] complex vs real max|diff|={diff:.2e}")
    irodori_model.precompute_freqs_cis = _precompute_freqs_real
    irodori_model.apply_rotary_emb = _apply_rotary_emb_real
    # Invalidate cached complex tables so they are rebuilt in real form.
    for module in model.modules():
        if hasattr(module, "_freqs_cis_cache"):
            module._freqs_cis_cache = torch.zeros(1, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Irodori-TTS to ONNX.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help="Hugging Face model repo id to download model.safetensors from.",
    )
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        help="DACVAE codec repo id.",
    )
    parser.add_argument("--out-dir", default="onnx_out", help="Output directory.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--text",
        default="こんにちは、音声合成のテストです。",
        help="Sample text used for tracing and parity checks.",
    )
    parser.add_argument(
        "--caption",
        default="落ち着いた大人の女性の声",
        help="Sample caption used for tracing and parity checks.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    checkpoint = _resolve_checkpoint(args)
    print(f"[checkpoint] {checkpoint}")
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
    model = runtime.model.eval()
    tokenizer = runtime.tokenizer
    caption_tokenizer = runtime.caption_tokenizer or runtime.tokenizer

    text_ids, text_mask = tokenizer.batch_encode([args.text], 64)
    caption_ids, caption_mask = caption_tokenizer.batch_encode([args.caption], 64)

    encoder = _Encoder(model).eval()
    dit = _DitStep(model).eval()

    # PyTorch references (before the RoPE patch; the patch must not change them).
    with torch.no_grad():
        ts_ref, cs_ref = encoder(text_ids, text_mask, caption_ids, caption_mask)
    patch_dim = model.cfg.latent_dim * model.cfg.latent_patch_size
    S = 90
    x_t = torch.randn(1, S, patch_dim)
    t = torch.tensor([0.5])
    with torch.no_grad():
        v_ref = dit(x_t, t, ts_ref, text_mask, cs_ref, caption_mask)

    _patch_rope_for_export(model)

    # ---- encoder ----
    encoder_path = os.path.join(args.out_dir, "encoder.onnx")
    torch.onnx.export(
        encoder,
        (text_ids, text_mask, caption_ids, caption_mask),
        encoder_path,
        input_names=["text_ids", "text_mask", "caption_ids", "caption_mask"],
        output_names=["text_state", "caption_state"],
        dynamic_axes={
            "text_ids": {1: "Tt"},
            "text_mask": {1: "Tt"},
            "caption_ids": {1: "Tc"},
            "caption_mask": {1: "Tc"},
            "text_state": {1: "Tt"},
            "caption_state": {1: "Tc"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"[onnx] exported {encoder_path}")

    # ---- DiT step (dynamic batch for single-call CFG) ----
    dit_path = os.path.join(args.out_dir, "dit.onnx")
    torch.onnx.export(
        dit,
        (x_t, t, ts_ref, text_mask, cs_ref, caption_mask),
        dit_path,
        input_names=["x_t", "t", "text_state", "text_mask", "caption_state", "caption_mask"],
        output_names=["v"],
        dynamic_axes={
            "x_t": {0: "B", 1: "S"},
            "t": {0: "B"},
            "text_state": {0: "B", 1: "Tt"},
            "text_mask": {0: "B", 1: "Tt"},
            "caption_state": {0: "B", 1: "Tc"},
            "caption_mask": {0: "B", 1: "Tc"},
            "v": {0: "B", 1: "S"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"[onnx] exported {dit_path}")

    # ---- DACVAE decoder ----
    codec = DACVAECodec.load(
        repo_id=args.codec_repo, device="cpu", dtype=torch.float32, deterministic_decode=True
    )
    decoder = _Decoder(codec.model).eval()
    z = torch.randn(1, codec.latent_dim, 100)
    with torch.no_grad():
        audio_ref = decoder(z)
    decoder_path = os.path.join(args.out_dir, "dacvae_decoder.onnx")
    torch.onnx.export(
        decoder,
        (z,),
        decoder_path,
        input_names=["z"],
        output_names=["audio"],
        dynamic_axes={"z": {2: "T"}, "audio": {2: "S"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[onnx] exported {decoder_path} (+ external .data file)")

    # ---- parity ----
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, os.cpu_count() // 2)
    providers = ["CPUExecutionProvider"]

    encoder_session = ort.InferenceSession(encoder_path, session_options, providers=providers)
    ts_onnx, cs_onnx = encoder_session.run(
        None,
        {
            "text_ids": text_ids.numpy(),
            "text_mask": text_mask.numpy(),
            "caption_ids": caption_ids.numpy(),
            "caption_mask": caption_mask.numpy(),
        },
    )
    print(
        f"[parity encoder] text_state={np.abs(ts_onnx - ts_ref.numpy()).max():.2e}"
        f"  caption_state={np.abs(cs_onnx - cs_ref.numpy()).max():.2e}"
    )

    dit_session = ort.InferenceSession(dit_path, session_options, providers=providers)
    dit_inputs = {
        "x_t": x_t.numpy(),
        "t": t.numpy(),
        "text_state": ts_ref.numpy(),
        "text_mask": text_mask.numpy(),
        "caption_state": cs_ref.numpy(),
        "caption_mask": caption_mask.numpy(),
    }
    v_onnx = dit_session.run(None, dit_inputs)[0]
    print(f"[parity dit] v={np.abs(v_onnx - v_ref.numpy()).max():.2e}")

    # Batched CFG contract: every batch row must equal its batch-1 counterpart,
    # including the fully-masked unconditional rows.
    ts_np, cs_np = ts_ref.numpy(), cs_ref.numpy()
    tm_np, cm_np = text_mask.numpy(), caption_mask.numpy()
    batch3 = dit_session.run(
        None,
        {
            "x_t": np.repeat(x_t.numpy(), 3, axis=0),
            "t": np.repeat(t.numpy(), 3, axis=0),
            "text_state": np.concatenate([ts_np, np.zeros_like(ts_np), ts_np]),
            "text_mask": np.concatenate([tm_np, np.zeros_like(tm_np), tm_np]),
            "caption_state": np.concatenate([cs_np, cs_np, np.zeros_like(cs_np)]),
            "caption_mask": np.concatenate([cm_np, cm_np, np.zeros_like(cm_np)]),
        },
    )[0]
    text_uncond = dit_session.run(
        None,
        {**dit_inputs, "text_state": np.zeros_like(ts_np), "text_mask": np.zeros_like(tm_np)},
    )[0]
    caption_uncond = dit_session.run(
        None,
        {**dit_inputs, "caption_state": np.zeros_like(cs_np), "caption_mask": np.zeros_like(cm_np)},
    )[0]
    print(
        f"[parity dit batch] cond={np.abs(batch3[0] - v_onnx[0]).max():.2e}"
        f"  text-uncond={np.abs(batch3[1] - text_uncond[0]).max():.2e}"
        f"  caption-uncond={np.abs(batch3[2] - caption_uncond[0]).max():.2e}"
    )

    decoder_session = ort.InferenceSession(decoder_path, session_options, providers=providers)
    audio_onnx = decoder_session.run(None, {"z": z.numpy()})[0]
    print(f"[parity decoder] audio={np.abs(audio_onnx - audio_ref.numpy()).max():.2e}")

    print(f"[done] all components exported to {args.out_dir}/")


if __name__ == "__main__":
    main()
