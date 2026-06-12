# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo)
[![VoiceDesign Demo](https://img.shields.io/badge/VoiceDesign%20Demo-HuggingFace%20Space-red)](https://huggingface.co/spaces/Aratako/Irodori-TTS-600M-v3-VoiceDesign-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

For an OpenAI-compatible inference API server, see [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

> [!IMPORTANT]
> `main` tracks the **v3** codebase and is intended for use with the **Irodori-TTS-500M-v3** base model release.
> It also supports the **Irodori-TTS-600M-v3-VoiceDesign** 3-branch VoiceDesign release.
> The current code remains backward-compatible with **Irodori-TTS-500M-v2** checkpoints, including **Irodori-TTS-500M-v2-VoiceDesign**.
> If you need the previous v2 codebase state, use the `v2` tag. If you need the previous v1 code, use the `v1` tag.
> v1 checkpoints / preprocessing are not compatible with v2/v3.
> The previous public v1 model is available at [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M).

For model weights and audio samples, please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Multi-modal Voice Design**: v3 VoiceDesign can combine text, reference speech, and caption text for voice identity plus style/emotion control
- **Emoji-based Style Control**: Emoji annotations in input text can influence delivery and non-verbal vocal expressions in supported checkpoints
- **Automatic Duration Prediction**: v3 base and v3 VoiceDesign checkpoints estimate output length without manual `--seconds`
- **Automatic Watermarking**: Generated audio is watermarked with [SilentCipher](https://github.com/sony/silentcipher) when available
- **Multi-GPU Training**: Distributed training via `uv run --no-sync torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Speaker Inversion**: Learn reusable speaker embedding tokens for a target voice while freezing the base model
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The current codebase supports two closely related checkpoint families:

1. **Base model (`Aratako/Irodori-TTS-500M-v3`)**:
   Text encoder + reference latent encoder + diffusion transformer + duration predictor. The reference latent encoder consumes patched DACVAE latents from reference audio for speaker/style conditioning. v2 base checkpoints remain supported for inference.
2. **VoiceDesign model (`Aratako/Irodori-TTS-600M-v3-VoiceDesign`)**:
   Text encoder + reference latent encoder + caption encoder + diffusion transformer + duration predictor. The v3 VoiceDesign path supports 3-branch conditioning from text, reference speech, and caption text. v2 VoiceDesign remains supported as a backward-compatible caption-only checkpoint family.

Shared building blocks:

1. **Text Encoder**: Token embeddings initialized from a pretrained LLM, followed by self-attention + SwiGLU transformer layers with RoPE
2. **Reference Latent Encoder**: Encodes patched reference audio latents for speaker identity conditioning
3. **Caption Encoder**: Encodes style-control text for emotion, tone, speaking style, and acoustic context
4. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs
5. **Duration Predictor**: v3 checkpoints include an integrated predictor for automatic output length estimation

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. The released v2/v3 checkpoints use the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync --extra cu128  # NVIDIA CUDA 12.8 (Linux/Windows)
```

If you want to explicitly select a PyTorch backend, use one of the backend
extras below:

```bash
# NVIDIA CUDA 12.8 on Linux/Windows
uv sync --extra cu128

# AMD ROCm on Linux/WSL
uv sync --extra rocm

# AMD ROCm on Windows native (Python 3.12 required)
uv sync --extra rocm-win --python 3.12

# Intel XPU on Linux/Windows
uv sync --extra xpu

# CPU-only, or macOS CPU/MPS via PyPI
uv sync --extra cpu
```

The PyTorch backend extras are mutually exclusive. The `cu128` extra uses the
PyTorch CUDA 12.8 index, the `rocm` extra uses the PyTorch ROCm index on
Linux, the `rocm-win` extra uses AMD's Windows ROCm 7.2 packages directly from
`repo.radeon.com`, and the `xpu` extra uses the PyTorch XPU index on
Linux/Windows. The `cpu` extra uses the CPU PyTorch index on Linux/Windows and
falls back to the standard PyPI PyTorch wheels on macOS.

After syncing with a backend extra, use `uv run --no-sync ...` for the commands
below to avoid re-syncing the environment without the selected PyTorch backend
extra.

The `rocm` extra includes `pytorch-triton-rocm` because `triton-rocm` alone does
not provide `triton.language` for the `transformers` to `torch._dynamo` import
path. This was validated with AMD GPU inference.

> **Note (rocm-win):** The Windows ROCm 7.2 wheels ship PyTorch 2.9.1 (Python
> 3.12 only). A graphics driver version 26.1.1 or later is required. See the
> [AMD ROCm Windows installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html)
> for driver prerequisites. `torchcodec` is resolved from PyPI for this backend;
> if a version compatible with torch 2.9.x is unavailable, audio decoding may
> fall back to `torchaudio`.

## Quick Start

### Simple Inference

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

Pure VoiceDesign from text + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

Style-controlled voice cloning with text + reference speech + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --ref-wav path/to/reference.wav \
  --caption "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。" \
  --output-wav outputs/sample_voice_design_clone.wav
```

### Speaker Inversion Inference

Use a learned Speaker Inversion embedding instead of reference audio:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed path/to/my.speaker.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Gradio Web UI

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v3 demo is available at [Aratako/Irodori-TTS-500M-v3-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo).
The reference input area supports either reference audio/latent input or a Speaker Inversion embedding via tabs.

For VoiceDesign checkpoints, use the dedicated UI:

```bash
uv run --no-sync python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The hosted VoiceDesign demo is available at [Aratako/Irodori-TTS-600M-v3-VoiceDesign-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-600M-v3-VoiceDesign-Demo).

`gradio_app.py` is for `Aratako/Irodori-TTS-500M-v3`. `gradio_app_voicedesign.py` is for `Aratako/Irodori-TTS-600M-v3-VoiceDesign` and remains compatible with v2 VoiceDesign checkpoints.

## Inference

### CLI

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run --no-sync python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

VoiceDesign checkpoints support caption conditioning. The v3 VoiceDesign model can run with
caption only by passing `--no-ref`, or with both reference speech and caption by passing
`--ref-wav`, `--ref-latent`, or `--ref-embed`.

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。" \
  --caption "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_voice_design_ref_caption.wav
```

The older `Aratako/Irodori-TTS-500M-v2-VoiceDesign` checkpoint is still supported, but it is caption-only and intentionally ignores speaker/reference conditioning.

LoRA adapter directories can be loaded dynamically at inference time without
exporting a merged checkpoint:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/base_model.safetensors \
  --lora-adapter outputs/irodori_tts_lora/checkpoint_final \
  --text "こんにちは、私はAIです。これはLoRA推論のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_lora.wav
```

Speaker Inversion embedding checkpoints can be used with the same base model that
was used for inversion training. Pass the embedding with `--ref-embed`;
it is mutually exclusive with `--ref-wav`, `--ref-latent`, and `--no-ref`.

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、私はAIです。これはSpeaker Inversion推論のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Output Duration

The v3 base and v3 VoiceDesign models integrate duration prediction into inference.
When `--seconds` is omitted, the runtime estimates the output length from the input
text and enabled conditions, then generates audio for that estimated duration. Use
`--duration-scale` to multiply the predicted length (`>1` longer, `<1` shorter). For
exact control, pass `--seconds` manually.

Older v2 checkpoints were trained with fixed-length 30-second targets. They remain
supported by the v3 codebase and still accept manual `--seconds`, but forcing a
non-default duration can reduce audio quality; prefer the v3 base model for automatic
or scaled duration control.

### Sway Sampling

For faster experimental inference, Sway Sampling can be combined with fewer Euler
steps:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav outputs/sample_sway.wav
```

### Additional Inference Notes

For tuning guidance and detailed explanations of inference options, see the
[Parameter Guide](docs/parameters.md).

Generated audio is passed through [SilentCipher](https://github.com/sony/silentcipher) watermarking automatically when the dependency and model files are available.

## Training

### 1. Prepare Manifest (Precompute DACVAE Latents)

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run --no-sync python prepare_manifest.py \
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
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `caption` in the manifest (for caption-conditioned voice design training):

```bash
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

Speaker/reference labels depend on the training mode:

- For v2 VoiceDesign training, `speaker_id` is optional because the model learns from
  `text + caption`.
- For v3 VoiceDesign training, keep `speaker_id` available so the model can learn from
  `text + speaker/reference + caption`.
- For Speaker Inversion training, `speaker_id` is not required because the run learns one
  shared speaker embedding from the target speaker samples.

The manifest `caption` value may also be a list of strings; training randomly selects one
non-empty caption each time that row is loaded.

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Training

Single-GPU training:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

v3 release training uses two phases. After training the body, initialize the integrated
duration predictor from the phase-1 checkpoint:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_phase2_duration.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_duration \
  --init-checkpoint outputs/irodori_tts/checkpoint_final.pt
```

v2 VoiceDesign training uses a dedicated config:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

`configs/train_500m_v2_voice_design.yaml` sets `use_caption_condition: true` and disables the
speaker/reference branch. Caption-free configs continue to use speaker conditioning when
`speaker_id` / reference inputs are available.

v3 VoiceDesign training uses two phases. Phase 1 initializes the RF/DiT body from the
v3 base checkpoint while adding the caption branch and skipping the base duration
predictor:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_voice_design_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_phase1 \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors
```

Phase 2 adds and trains a newly initialized duration predictor with text + speaker +
caption conditioning:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_voice_design_phase2_duration.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_phase2 \
  --init-checkpoint outputs/irodori_tts_voice_design_phase1/checkpoint_final.pt
```

The VoiceDesign config also enables `caption_warmup: true` for optional caption-branch warmup.
`warmup_steps` controls the LR scheduler, while `caption_warmup_steps` controls how long
non-caption gradients are discarded before normal joint training resumes.

### v3 Duration Predictor Training

v3 training uses two phases: `configs/train_500m_v3_phase1_body.yaml` trains the
variable-length DiT body, then `configs/train_500m_v3_phase2_duration.yaml` freezes the
body and trains the duration predictor.

The duration predictor regresses `log1p(num_frames)` with Huber loss. The current v3 phase2
config uses the token-sum duration predictor selected from ablations; see the parameter
guide for the architecture details.

Multi-GPU DDP training:

```bash
uv run --no-sync torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run --no-sync python train.py --help` for all available options.
For a more detailed explanation of model and training config fields, see [Parameter Guide](docs/parameters.md).

#### Fine-Tuning from Released Weights

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh. For the v3 base release, the LoRA config keeps the duration predictor as part of the saved adapter by default.

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors
```

v3 VoiceDesign LoRA fine-tuning:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_voice_design_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_lora \
  --init-checkpoint path/to/Irodori-TTS-600M-v3-VoiceDesign.safetensors
```

For the older v2 VoiceDesign checkpoint, use `configs/train_500m_v2_voice_design_lora.yaml`
and initialize from `Irodori-TTS-500M-v2-VoiceDesign.safetensors`.

LoRA target presets, adapter saving behavior, and resume details are covered in the
[Parameter Guide](docs/parameters.md).

#### Speaker Inversion

Speaker Inversion trains only a small set of speaker embedding tokens while keeping the
base Irodori-TTS model frozen. It is useful when you want a reusable speaker identity
checkpoint instead of providing reference audio at every inference call.

Prepare a manifest from the target speaker's audio, then initialize from the released v3
base checkpoint:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --output-dir outputs/speaker_inversion/name
```

The saved checkpoints are embedding-only `.speaker.safetensors` files, for example
`outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors`. Use that file
with the base model during inference:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

To continue from a saved embedding, set `speaker_inversion_init_embedding` in the
config or pass `--speaker-inversion-init-embedding path/to/checkpoint.speaker.safetensors`.
Full trainer `--resume` is intentionally not used for Speaker Inversion checkpoints.
Enable `gradient_checkpointing: true` or pass `--gradient-checkpointing` if GPU memory is tight.

#### Resuming Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run --no-sync python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 3. Checkpoint Conversion

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference. If you do not want to merge the adapter, pass the adapter directory directly to `infer.py --lora-adapter` or the matching Gradio field.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # Gradio web UI for VoiceDesign checkpoints
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
│
├── docs/
│   └── parameters.md         # Detailed parameter guide
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
│   ├── speaker_inversion.py    # Speaker Inversion embedding save/load helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_500m_v3_phase1_body.yaml        # 500M v3 body training config
    ├── train_500m_v3_phase2_duration.yaml    # 500M v3 duration-predictor training config
    ├── train_500m_v3_voice_design_phase1_body.yaml     # 600M v3 VoiceDesign body config
    ├── train_500m_v3_voice_design_phase2_duration.yaml # 600M v3 VoiceDesign duration config
    ├── train_500m_v3_voice_design_lora.yaml            # 600M v3 VoiceDesign RF+duration LoRA config
    ├── train_500m_v3_lora.yaml               # 500M v3 LoRA fine-tuning config
    ├── train_500m_v3_speaker_inversion.yaml  # 500M v3 Speaker Inversion config
    ├── train_500m_v2.yaml                    # 500M v2 backward-compatible model config
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRA fine-tuning config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign full fine-tuning config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRA fine-tuning config
    ├── train_500m.yaml                       # 500M v1 model config
    └── train_2.5b.yaml                       # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
- [SilentCipher](https://github.com/sony/silentcipher) — Audio watermarking

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
