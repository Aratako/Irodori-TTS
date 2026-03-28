# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

> [!IMPORTANT]
> `main` tracks the **v2** codebase and is intended for use with the **Irodori-TTS-500M-v2** model release.
> If you need the previous v1 code, use the `v1` tag.
> v1 and v2 checkpoints / preprocessing are not compatible across versions.
> The previous public v1 model is available at [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M).

For model weights and audio samples, please refer to the [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Multi-GPU Training**: Distributed training via `uv run torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The model consists of three main components:

1. **Text Encoder**: Token embeddings initialized from a pretrained LLM, followed by self-attention + SwiGLU transformer layers with RoPE
2. **Reference Latent Encoder**: Encodes patched reference audio latents for speaker/style conditioning via self-attention + SwiGLU layers
3. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. v2 uses the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync
```

**Note**: For Linux/Windows with CUDA, PyTorch is automatically installed from the cu128 index. For macOS (MPS) or CPU-only usage, `uv sync` will install the default PyTorch build.

### macOS Notes

- Apple Silicon is supported through PyTorch MPS.
- The CLI / Gradio app now automatically set `PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS unless you already set it yourself.
- On macOS, the default runtime is `model_device=mps` and `codec_device=cpu`. This keeps generation on MPS while avoiding common DACVAE / torchaudio MPS incompatibilities.
- `--compile-model` is currently CUDA-only in practice. On macOS / MPS, leave it disabled.

## Quick Start

### Simple Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

macOS example with explicit devices:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --model-device mps \
  --codec-device cpu \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v2 demo is available at [Aratako/Irodori-TTS-500M-v2-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo).

## Inference

### CLI

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--text` | (required) | Text to synthesize |
| `--ref-wav` | None | Reference audio for voice cloning |
| `--ref-latent` | None | Pre-computed reference latent (.pt) |
| `--no-ref` | False | Unconditional generation (no reference) |
| `--ref-normalize-db` | -16.0 | Reference loudness target before DACVAE encode (set `none` to disable) |
| `--ref-ensure-max` | True | Scale reference down only when peak exceeds 1.0 (used when `--ref-normalize-db` is disabled) |
| `--codec-deterministic-encode` | True | Use deterministic DACVAE encode path |
| `--codec-deterministic-decode` | True | Use deterministic DACVAE watermark-message decode path |
| `--num-steps` | 40 | Number of Euler integration steps |
| `--cfg-scale-text` | 3.0 | CFG scale for text conditioning |
| `--cfg-scale-speaker` | 5.0 | CFG scale for speaker conditioning |
| `--guidance-mode` | `independent` | CFG mode: `independent`, `joint`, `alternating` |
| `--model-device` | auto | Device for model (`cuda`, `mps`, `cpu`) |
| `--codec-device` | auto (`cpu` on macOS/MPS) | Device for DACVAE codec |
| `--model-precision` | auto | Model precision (`fp32`, `bf16`) |
| `--codec-precision` | auto | Codec precision (`fp32`, `bf16`) |
| `--seed` | random | Random seed for reproducibility |
| `--compile-model` | False | Enable `torch.compile` for faster inference (currently avoid on macOS / MPS) |
| `--trim-tail` | True | Trim trailing silence via flattening heuristic |

## Training

### 1. Prepare Manifest (Precompute DACVAE Latents)

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `speaker_id` in the manifest (for speaker-conditioned training):

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Training

Single-GPU training:

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

Multi-GPU DDP training:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run python train.py --help` for all available options.

#### Fine-Tuning from Released Weights

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh.

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_ft \
  --init-checkpoint path/to/Irodori-TTS-500M-v2.safetensors
```

LoRA fine-tuning:

```bash
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2.safetensors
```

Available LoRA target presets:

- `text_attn_mlp`: text encoder attention + attention gate + MLP
- `speaker_attn_mlp`: speaker encoder attention + attention gate + MLP, plus `speaker_encoder.in_proj`
- `diffusion_attn`: diffusion attention only, including text/speaker context KV and attention gate
- `diffusion_attn_mlp`: `diffusion_attn` + diffusion MLP
- `all_attn`: all attention blocks across text/speaker/diffusion, including attention gates
- `diffusion_full`: diffusion stack broadly: `cond_module`, `in_proj/out_proj`, diffusion attention, diffusion MLP, and AdaLN
- `adaln`: diffusion-block AdaLN layers only
- `conditioning`: conditioning-side projections only: `cond_module`, `speaker_encoder.in_proj`, and diffusion context KV projections
- `all_attn_mlp`: `all_attn` + text/speaker/diffusion MLP, plus `speaker_encoder.in_proj`
- `all_linear`: all `nn.Linear` layers in the model; embeddings and norm weights are not included

`--lora-target-modules` also accepts a regex string or a comma-separated list of module suffixes. Resume automatically restores the saved LoRA config from the training checkpoint unless you explicitly override it.

When `--lora` is enabled, checkpoints are saved as adapter-only directories containing PEFT adapter weights plus trainer state for resume.

#### Resuming Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 3. Checkpoint Conversion

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
│
├── irodori_tts/                # Core library
│   ├── model.py                # TextToLatentRFDiT architecture
│   ├── rf.py                   # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                # DACVAE codec wrapper
│   ├── dataset.py              # Dataset and collator
│   ├── tokenizer.py            # Pretrained LLM tokenizer wrapper
│   ├── config.py               # Model / Train / Sampling config dataclasses
│   ├── inference_runtime.py    # Cached, thread-safe inference runtime
│   ├── lora.py                 # PEFT LoRA integration helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_500m_v2.yaml       # 500M v2 model config
    ├── train_500m_v2_lora.yaml  # 500M v2 LoRA fine-tuning config
    ├── train_500m.yaml          # 500M v1 model config
    └── train_2.5b.yaml          # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
