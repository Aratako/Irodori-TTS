# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo)
[![VoiceDesign Demo](https://img.shields.io/badge/VoiceDesign%20Demo-HuggingFace%20Space-red)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

For an OpenAI-compatible inference API server, see [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

> [!IMPORTANT]
> `main` tracks the **v3** codebase and is intended for use with the **Irodori-TTS-500M-v3** base model release.
> The current code remains backward-compatible with **Irodori-TTS-500M-v2** checkpoints, including **Irodori-TTS-500M-v2-VoiceDesign**.
> If you need the previous v2 codebase state, use the `v2` tag. If you need the previous v1 code, use the `v1` tag.
> v1 checkpoints / preprocessing are not compatible with v2/v3.
> The previous public v1 model is available at [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M).

For model weights and audio samples, please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Voice Design**: Caption-conditioned style control
- **Automatic Duration Prediction**: v3 base checkpoints estimate output length without manual `--seconds`
- **Automatic Watermarking**: Generated audio is watermarked with [SilentCipher](https://github.com/sony/silentcipher) when available
- **Multi-GPU Training**: Distributed training via `uv run torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Speaker Inversion**: Learn reusable speaker embedding tokens for a target voice while freezing the base model
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The current codebase supports two closely related checkpoint families:

1. **Base model (`Aratako/Irodori-TTS-500M-v3`)**:
   Text encoder + reference latent encoder + diffusion transformer + duration predictor. The reference latent encoder consumes patched DACVAE latents from reference audio for speaker/style conditioning. v2 base checkpoints remain supported for inference.
2. **VoiceDesign model (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`)**:
   Text encoder + caption encoder + diffusion transformer. The caption encoder consumes style-control text and the speaker/reference branch is disabled. A v3 VoiceDesign release is not available yet, so this path still uses the v2 checkpoint.

Shared building blocks:

1. **Text Encoder**: Token embeddings initialized from a pretrained LLM, followed by self-attention + SwiGLU transformer layers with RoPE
2. **Condition Encoder**: Either a reference latent encoder for the base model or a caption encoder for the VoiceDesign model
3. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs
4. **Duration Predictor**: v3 base checkpoints include an integrated predictor for automatic output length estimation

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. The released v2/v3 checkpoints use the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync
```

**Note**: For Linux/Windows with CUDA, PyTorch is automatically installed from the cu128 index. For macOS (MPS) or CPU-only usage, `uv sync` will install the default PyTorch build.

## Quick Start

### Simple Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

### Speaker Inversion Inference

Use a learned Speaker Inversion embedding instead of reference audio:

```bash
uv run python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed path/to/my.speaker.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v3 demo is available at [Aratako/Irodori-TTS-500M-v3-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo).
The reference input area supports either reference audio/latent input or a Speaker Inversion embedding via tabs.

For the VoiceDesign checkpoint, use the dedicated UI:

```bash
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The hosted VoiceDesign demo is available at [Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo).

`gradio_app.py` is for `Aratako/Irodori-TTS-500M-v3`. `gradio_app_voicedesign.py` is for `Aratako/Irodori-TTS-500M-v2-VoiceDesign`.

## Inference

### CLI

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

VoiceDesign checkpoints also support caption conditioning:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

LoRA adapter directories can be loaded dynamically at inference time without
exporting a merged checkpoint:

```bash
uv run python infer.py \
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
uv run python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、私はAIです。これはSpeaker Inversion推論のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Output Duration

The v3 base model integrates duration prediction into inference.
When `--seconds` is omitted, the runtime estimates the output length from the input
text and, for speaker-conditioned checkpoints, the reference audio, then generates
audio for that estimated duration. Use `--duration-scale` to multiply the predicted
length (`>1` longer, `<1` shorter). For exact control, pass `--seconds` manually.

Older v2 checkpoints were trained with fixed-length 30-second targets. They remain
supported by the v3 codebase and still accept manual `--seconds`, but forcing a
non-default duration can reduce audio quality; prefer the v3 base model for automatic
or scaled duration control.

### Sway Sampling

For faster experimental inference, Sway Sampling can be combined with fewer Euler
steps:

```bash
uv run python infer.py \
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

To include `caption` in the manifest (for caption-conditioned voice design training):

```bash
uv run python prepare_manifest.py \
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

When training the caption-conditioned voice-design model, `speaker_id` is optional. The
voice-design path disables speaker/reference conditioning and learns from `text + caption`.
For Speaker Inversion training, the manifest should contain samples of the target voice; `speaker_id`
is not required because the run learns one shared speaker embedding.

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Training

Single-GPU training:

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

v3 release training uses two phases. After training the body, initialize the integrated
duration predictor from the phase-1 checkpoint:

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase2_duration.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_duration \
  --init-checkpoint outputs/irodori_tts/checkpoint_final.pt
```

VoiceDesign training uses a dedicated config:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

`configs/train_500m_v2_voice_design.yaml` sets `use_caption_condition: true` and disables the
speaker/reference branch. Caption-free configs continue to use speaker conditioning when
`speaker_id` / reference inputs are available.

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
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run python train.py --help` for all available options.
For a more detailed explanation of model and training config fields, see [Parameter Guide](docs/parameters.md).

#### Fine-Tuning from Released Weights

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh. For the v3 base release, the LoRA config keeps the duration predictor as part of the saved adapter by default.

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors
```

Caption-conditioned voice-design LoRA fine-tuning:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2-VoiceDesign.safetensors
```

LoRA target presets, adapter saving behavior, and resume details are covered in the
[Parameter Guide](docs/parameters.md).

#### Speaker Inversion

Speaker Inversion trains only a small set of speaker embedding tokens while keeping the
base Irodori-TTS model frozen. It is useful when you want a reusable speaker identity
checkpoint instead of providing reference audio at every inference call.

Prepare a manifest from the target speaker's audio, then initialize from the released v3
base checkpoint:

```bash
uv run python train.py \
  --config configs/train_500m_v3_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --output-dir outputs/speaker_inversion/name
```

The saved checkpoints are embedding-only `.speaker.safetensors` files, for example
`outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors`. Use that file
with the base model during inference:

```bash
uv run python infer.py \
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
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
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
- **Model Weights**: Please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) for licensing details

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
