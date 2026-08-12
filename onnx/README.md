# ONNX Export

Export the Irodori-TTS neural components to ONNX so the model can run on
non-PyTorch runtimes (onnxruntime on mobile/edge devices, etc.).

## What gets exported

| File | Graph | I/O |
|---|---|---|
| `encoder.onnx` | text/caption conditioning encoder | `(text_ids, text_mask, caption_ids, caption_mask) -> (text_state, caption_state)` |
| `dit.onnx` | one RF-DiT velocity step, **dynamic batch** | `(x_t, t, text_state, text_mask, caption_state, caption_mask) -> v` |
| `dacvae_decoder.onnx` | DACVAE latent → 48 kHz waveform | `(z: B×latent_dim×T) -> (audio: B×1×S)` |

The sampler itself (sway t-schedule, independent CFG, Euler integration) is
deliberately **not** exported — it is a few lines of driver code that a target
runtime implements natively. `verify_onnx.py` contains the reference
implementation and doubles as the porting spec.

## Usage

```bash
# 1. Export (downloads the checkpoint from HF, or use --checkpoint for a local file)
python onnx/export_onnx.py --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign

# 2. Verify the full pipeline end to end (PyTorch vs pure ONNX, same seed)
python onnx/verify_onnx.py --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
    --onnx-dir onnx_out
```

`export_onnx.py` prints per-component parity against PyTorch; `verify_onnx.py`
synthesizes the same utterance through both stacks and fails if waveform
correlation drops below `--min-corr` (default 0.999).

Measured on Irodori-TTS-500M-v2-VoiceDesign (fp32, CPU):

| Check | Result |
|---|---|
| encoder parity (max abs diff) | ~4e-6 |
| DiT step parity | ~1e-5 |
| batched CFG rows vs batch-1 (cond / text-uncond / caption-uncond) | 0.0 (bit-exact) |
| decoder parity | ~2e-5 |
| end-to-end waveform correlation (8 steps, CFG 3/3, seed 0) | ≥ 0.9999 |

## Implementation notes

- **Rotary embeddings**: ONNX has no complex dtype, so the export monkeypatches
  `precompute_freqs_cis` / `apply_rotary_emb` with an equivalent real-valued
  form (`[cos, sin]` tables, adjacent-pair rotation; max diff vs the complex
  implementation ~2e-7). The patch is confined to the export process.
- **`normalize_text` is mandatory** before tokenization when driving the ONNX
  pipeline yourself — it changes token ids (e.g. full-width punctuation) and
  skipping it audibly destroys output.
- **Duration**: the v2-VoiceDesign checkpoint has no duration predictor. Size
  the latent sequence from the requested duration (25 latent frames per
  second) — generating the model's full 750-frame window and trimming wastes
  ~8× compute for short utterances.
- **Batched CFG**: `dit.onnx` has a dynamic batch axis so a driver can run all
  three CFG variants (cond / text-uncond / caption-uncond) in a single call.
  Unconditional variants are zero states with all-false masks.
- **Sampler steps**: Euler with the sway schedule needs enough steps to
  converge — 8 matches reference quality, 5–6 is a usable fast setting, and
  ≤4 is audibly degraded (the flow has not converged).
- **External weights**: `dacvae_decoder.onnx` stores its weights in an
  external `dacvae_decoder.onnx.data` file; keep it next to the graph.

## Scope

Covers the text+caption conditioning path (VoiceDesign without reference
audio), tested with **Irodori-TTS-500M-v2-VoiceDesign**. The reference-audio
branch and the v3 duration predictor are not exported. The pipeline has been
validated on-device with onnxruntime-android (arm64, fp32).
