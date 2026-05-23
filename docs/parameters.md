# Irodori-TTS Parameter Guide

This document explains the main inference and training parameters used by Irodori-TTS.

## Version Notes

`main` currently targets the v3 codebase. It remains backward-compatible with v2
checkpoints, including the current VoiceDesign release.

- v3 base checkpoints include the integrated duration predictor and can estimate output
  length automatically when `--seconds` is omitted.
- v3 base release training is split into two phases: first the RF/DiT body, then
  `duration_only` training for the duration predictor.
- v2 checkpoints do not include the duration predictor and were trained with fixed
  30-second targets, but they are still supported by the v3 code.
- VoiceDesign checkpoints currently use the v2 release, set `use_caption_condition: true`,
  and intentionally disable speaker/reference conditioning.

## Inference Parameters

### Checkpoint Selection

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--checkpoint` | required unless `--hf-checkpoint` is set | Local `.pt` or `.safetensors` checkpoint. Use this for converted local checkpoints or downloaded model files that you want to reference directly. |
| `--hf-checkpoint` | required unless `--checkpoint` is set | Hugging Face repo id. The runtime downloads `model.safetensors` from the repo. |
| `--lora-adapter` | `None` | Optional PEFT LoRA adapter directory loaded dynamically at inference time. The adapter is not merged into the base checkpoint. |
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | DACVAE codec used to encode reference audio and decode generated latents. It should normally match the checkpoint metadata. |

Use either `--checkpoint` or `--hf-checkpoint`, not both.

### Text, Caption, and Reference Conditioning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--text` | required | Text to synthesize. It is tokenized with the checkpoint's text tokenizer. |
| `--caption` | `None` | Voice and style-control text for VoiceDesign checkpoints. Ignored or ineffective for checkpoints without caption conditioning. |
| `--ref-wav` | `None` | Reference waveform used for speaker/style conditioning in the base model. |
| `--ref-latent` | `None` | Precomputed reference latent (`.pt`) used instead of encoding `--ref-wav` at inference time. Useful for repeated inference with the same reference. |
| `--no-ref` | `False` | Disables speaker/reference conditioning. Use this for VoiceDesign checkpoints, or for text-only inference with base checkpoints. |
| `--ref-embed` | `None` | Speaker Inversion embedding (`.speaker.safetensors`) path. Mutually exclusive with `--ref-wav`, `--ref-latent`, and `--no-ref`. Use the file produced by Speaker Inversion training instead of a reference waveform. |
| `--max-ref-seconds` | `30.0` | Caps the reference audio duration before encoding. The released models were trained on audio up to 30 seconds, so keeping the default cap is recommended. Set `<=0` only when you intentionally want to disable the cap. |
| `--ref-normalize-db` | `-16.0` | Loudness target applied to reference audio before DACVAE encode. This normalization was used when training the codec, so keeping the default is recommended. Use `none` only for controlled experiments. |
| `--ref-ensure-max` | `True` | When loudness normalization is disabled, scales the reference down only if peak amplitude exceeds `1.0`. In normal use, prefer leaving loudness normalization enabled instead of relying on this fallback. |
| `--max-text-len` | checkpoint metadata or `256` | Maximum text token length. Longer text is truncated. Keeping the checkpoint/training-time setting is recommended. |
| `--max-caption-len` | checkpoint metadata or `max_text_len` | Maximum caption token length for VoiceDesign checkpoints. Keeping the checkpoint/training-time setting is recommended. |

Reference audio is a conditioning signal, not the generated target. Short, clean
references may work well, but the more important point is to avoid music, noise, or
multiple speakers.
For speaker-conditioned checkpoints, `--ref-latent` is the fastest path when the same
speaker reference is reused many times.

### Duration Control

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--seconds` | `None` | Manual output duration. If set, it always overrides the default duration behavior. |
| `--duration-scale` | `1.0` | Multiplies the predicted duration when duration prediction is used. Values above `1.0` produce longer audio; values below `1.0` produce shorter audio. |

The recommended duration behavior depends on the checkpoint:

- v2 checkpoints, including the VoiceDesign release, were trained with fixed 30-second
  targets. Setting a different duration is not recommended because it moves inference
  away from the training setup.
- v3 base checkpoints were trained with variable-length targets and an integrated
  duration predictor. For these checkpoints, leaving `--seconds` unset is recommended so
  the model can choose the duration automatically. Manual `--seconds` is still available
  when exact control is needed.

When `--seconds` is omitted, the runtime checks whether the loaded checkpoint has
duration-predictor weights. If it does, the predicted frame count is used and then scaled
by `--duration-scale`. If it does not, the runtime falls back to 30 seconds.

### Sampling and Candidate Generation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--num-steps` | `40` | Number of Euler integration steps. Higher values are slower and can improve stability up to a point. |
| `--t-schedule-mode` | `linear` | Timestep schedule for RF Euler sampling. Use `sway` to enable Sway Sampling. |
| `--sway-coeff` | `-1.0` | Sway Sampling coefficient. Negative values allocate more schedule resolution to the noise side. |
| `--num-candidates` | `1` | Number of candidates generated in one batched sampling pass. Higher values increase VRAM use. |
| `--decode-mode` | `sequential` | `sequential` decodes candidates one by one and uses less VRAM. `batch` decodes all candidates together and can be faster. |
| `--seed` | random | Sampling seed. Set it for reproducible results with the same checkpoint and parameters. |
| `--truncation-factor` | `None` | Scales the initial Gaussian noise before sampling. Values such as `0.8` or `0.9` can reduce variation, but may also reduce expressiveness. |
| `--rescale-k` / `--rescale-sigma` | `None` | Temporal score rescaling parameters. Set both together or leave both unset. |

`--num-steps` is usually the first quality/speed knob to try. For quick experiments,
lower values can be acceptable; for final samples, start from the default before making
other changes.

For lower-latency experiments, try Sway Sampling with fewer steps:

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

### Classifier-Free Guidance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--cfg-scale-text` | `3.0` | Guidance strength for text conditioning. Higher values force the text condition more strongly. |
| `--cfg-scale-caption` | `3.0` | Guidance strength for caption/style conditioning. Applies to VoiceDesign checkpoints. |
| `--cfg-scale-speaker` | `5.0` | Guidance strength for reference speaker conditioning. Ignored when speaker conditioning is disabled. |
| `--cfg-guidance-mode` | `independent` | CFG formulation: `independent`, `joint`, or `alternating`. |
| `--cfg-scale` | `None` | Deprecated shared override for all enabled CFG scales. Prefer the per-condition scale parameters. |
| `--cfg-min-t` | `0.5` | Lower timestep bound where CFG is active. |
| `--cfg-max-t` | `1.0` | Upper timestep bound where CFG is active. |

In `independent` mode, each enabled condition gets its own unconditional branch in a
single larger batch. This is the most flexible mode for using different text, caption,
and speaker scales, but the batch size during CFG steps grows with the number of enabled
conditions, so it can use more VRAM and compute. In NFE terms, it is
`1 + number_of_enabled_cfg_conditions` during CFG-active steps; the released base and
VoiceDesign setups commonly have two enabled CFG conditions, so this is typically 3x.
`joint` drops all enabled conditions together and expects equal CFG scales; it uses the
conditional branch plus one joint unconditional branch during CFG steps, so it is 2x
NFE. `alternating` also uses one unconditional branch per CFG step, so it is 2x NFE,
but alternates which condition is dropped at each step.

Increasing a CFG scale can improve adherence to that condition, but very high values may
make speech less natural. If pronunciation is weak, try increasing `--cfg-scale-text`
slightly. If speaker similarity is weak, try `--cfg-scale-speaker` or the speaker K/V
controls below.

### Speaker K/V Scaling

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--speaker-kv-scale` | `None` | Extra scaling applied to speaker context K/V projections. Values above `1.0` can strengthen speaker identity. |
| `--speaker-kv-min-t` | `0.9` | Applies speaker K/V scaling only while `t >= value`. |
| `--speaker-kv-max-layers` | `None` | Limits speaker K/V scaling to the first N diffusion layers. |

These parameters are experimental speaker-similarity controls. They are only meaningful
for speaker-conditioned checkpoints with reference conditioning enabled. If the generated
voice drifts from the reference, try moderate `--speaker-kv-scale` values before making
large CFG changes.

### Speaker Inversion

Speaker Inversion trains a compact set of learned tokens that represent a speaker identity.
At inference, load the resulting `.speaker.safetensors` file with `--ref-embed` instead of
a reference waveform.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--speaker-uncond-mode` | `mask` | Unconditional speaker formulation for CFG when using `--ref-embed`. `mask`: zero tokens with a false mask (default, lower VRAM). `noise`: Gaussian noise scaled to the standard deviation of the embedding. |

### Devices, Precision, and Compilation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model-device` | auto | Device for the RF-DiT model, for example `cuda`, `mps`, or `cpu`. |
| `--codec-device` | auto | Device for DACVAE encode/decode. It can differ from `--model-device`. |
| `--model-precision` | `fp32` | Model compute precision: `fp32` or `bf16`. |
| `--codec-precision` | `fp32` | Codec compute precision: `fp32` or `bf16`. |
| `--compile-model` | `False` | Enables `torch.compile` for core inference methods. First run is slower due to compilation. |
| `--compile-dynamic` | `False` | Uses `dynamic=True` with `torch.compile`. |
| `--context-kv-cache` | `True` | Precomputes text/speaker/caption K/V projections for faster sampling. |

For CUDA inference, `bf16` can reduce memory use and improve speed on supported GPUs.
For CPU or MPS, `fp32` is the safer default. `--compile-model` is most useful when
running many requests with similar shapes.

### Tail Trimming and Timings

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--trim-tail` | `True` | Trims trailing near-zero latent regions with a flattening heuristic. |
| `--tail-window-size` | `20` | Window size used by the tail-trimming heuristic. |
| `--tail-std-threshold` | `0.05` | Standard deviation threshold for tail trimming. |
| `--tail-mean-threshold` | `0.1` | Mean threshold for tail trimming. |
| `--show-timings` | `True` | Prints timing breakdowns for major inference stages. |

Tail trimming was mainly introduced for v2 checkpoints, which generate fixed 30-second
outputs and can leave unused trailing regions after the spoken content. It is less
important for v3 base checkpoints because they predict a more appropriate output length.
If valid audio is being trimmed too aggressively, disable `--trim-tail` first.
Adjust the tail thresholds only when you need fine control over the trimming heuristic.

## Training Parameters

Training is configured through YAML files with `model` and `train` sections. CLI options
override YAML values when explicitly provided.

### Data and Checkpoint Flow

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--config` | `None` | YAML file containing `model` and `train` overrides. |
| `--manifest` | required | JSONL manifest produced by `prepare_manifest.py`. Each row must include `text` and `latent_path`; `speaker_id` and `caption` are optional depending on the model. |
| `--output-dir` | `outputs/irodori_tts` | Directory for checkpoints, trainer state, configs, and logs. |
| `--init-checkpoint` | `None` | Initializes model weights from a `.pt` or `.safetensors` checkpoint, then starts optimizer/scheduler state from scratch. |
| `--resume` | `None` | Restores full training state from a training checkpoint. Use `.pt` for full-model runs and checkpoint directories for LoRA runs. |

Use `--init-checkpoint` for fine-tuning from released inference weights. Use `--resume`
only to continue an interrupted training run.

### Model Config

| Field | Notes |
|-------|-------|
| `latent_dim` | DACVAE latent dimension expected by the model. v2/v3 500M configs use `32`. |
| `latent_patch_size` | Number of latent frames grouped per model token. |
| `model_dim`, `num_layers`, `num_heads`, `mlp_ratio` | Main diffusion transformer width, depth, attention heads, and MLP expansion. |
| `text_tokenizer_repo`, `text_vocab_size`, `text_add_bos` | Tokenizer and vocabulary settings for the text encoder. |
| `text_dim`, `text_layers`, `text_heads`, `text_mlp_ratio` | Text encoder size. |
| `speaker_dim`, `speaker_layers`, `speaker_heads`, `speaker_patch_size`, `speaker_mlp_ratio` | Reference/speaker encoder size. Ignored when caption conditioning disables speaker conditioning. |
| `use_caption_condition` | Enables the VoiceDesign caption path and disables speaker/reference conditioning. |
| `caption_*` fields | Caption encoder tokenizer and architecture. When left unset, many fields inherit the corresponding text settings. |
| `timestep_embed_dim`, `adaln_rank`, `norm_eps` | Diffusion conditioning and normalization parameters. |
| `use_duration_predictor` | Enables v3 duration prediction. |
| `duration_*` fields | Duration predictor architecture, hidden size, depth, dropout, speaker conditioning, and token-sum initialization. |

Architecture fields should match the checkpoint you initialize from. Changing dimensions,
layer counts, vocabulary sizes, or conditioning branches usually prevents checkpoint
loading unless you are intentionally training from scratch or using an upgrade path
handled by the code.

### Batch Size, Length, and Masking

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `batch_size` / `--batch-size` | `8` | Per-process micro-batch size. In DDP, global batch size is multiplied by world size. |
| `gradient_accumulation_steps` / `--gradient-accumulation-steps` | `1` | Accumulates gradients over multiple micro-batches before optimizer update. |
| `max_text_len` / `--max-text-len` | `256` | Maximum token length for text conditioning. |
| `max_caption_len` / `--max-caption-len` | `None` | Maximum token length for caption conditioning; defaults to `max_text_len`. |
| `max_latent_steps` / `--max-latent-steps` | `750` | Maximum latent length loaded from each sample. At 25 fps, `750` is about 30 seconds. |
| `fixed_target_latent_steps` / `--fixed-target-latent-steps` | `750` | If set, all training targets are padded/truncated to this length. Set to `null` in YAML for variable-length training. |
| `fixed_target_full_mask` / `--fixed-target-full-mask` | `True` | For fixed-length training, includes padded tail positions in the loss mask. |
| `rf_loss_mode` / `--rf-loss-mode` | `echo` | RF loss normalization. `utterance_mean` averages per utterance and is used by v3 variable-length configs. |

The v2 and current VoiceDesign configs use fixed 30-second targets. The v3 phase-1 body
and phase-2 duration configs set `fixed_target_latent_steps: null`,
`fixed_target_full_mask: false`, and `rf_loss_mode: utterance_mean` so samples can keep
their natural lengths.

### Optimizer and Schedule

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `precision` / `--precision` | `bf16` | Forward-pass compute precision. Weights and optimizer states remain FP32. |
| `allow_tf32` / `--tf32` | `False` | Enables TF32 CUDA kernels for speed. |
| `compile_model` / `--compile-model` | `False` | Enables `torch.compile` during training. |
| `gradient_checkpointing` / `--gradient-checkpointing` | `False` | Enables activation checkpointing on diffusion blocks to reduce memory usage at the cost of extra compute. |
| `optimizer` / `--optimizer` | `muon` | `muon` or `adamw`. |
| `learning_rate` / `--lr` | `1e-4` | Base learning rate. |
| `weight_decay` / `--weight-decay` | `0.01` | Weight decay for optimizer groups that use it. |
| `adam_beta1`, `adam_beta2`, `adam_eps` | `0.9`, `0.999`, `1e-8` | AdamW hyperparameters. |
| `muon_momentum` / `--muon-momentum` | `0.95` | Momentum used by Muon. |
| `lr_scheduler` / `--lr-scheduler` | `none` | `none`, `cosine`, or `wsd`. |
| `warmup_steps` / `--warmup-steps` | `0` | Number of optimizer steps for LR warmup. |
| `stable_steps` / `--stable-steps` | `0` | Stable plateau length for the WSD schedule. |
| `min_lr_scale` / `--min-lr-scale` | `0.1` | Minimum LR multiplier at the end of decay. |
| `grad_clip_norm` | `1.0` | Gradient clipping norm. Currently configured through YAML. |

The 500M example configs use `optimizer: muon` and `lr_scheduler: wsd`. When changing
effective batch size, revisit the learning rate and warmup length together.

### Condition Dropout and Timesteps

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `text_condition_dropout` / `--text-condition-dropout` | `0.1` | Probability of replacing text conditioning with the null condition during training. |
| `caption_condition_dropout` / `--caption-condition-dropout` | `0.1` | Same idea for caption conditioning. |
| `speaker_condition_dropout` / `--speaker-condition-dropout` | `0.1` | Same idea for speaker/reference conditioning. |
| `timestep_stratified` / `--timestep-stratified` | `True` | Uses stratified logit-normal timestep sampling. |
| `timestep_logit_mean`, `timestep_logit_std` | `0.0`, `1.0` | Logit-normal timestep distribution parameters. |
| `timestep_min`, `timestep_max` | `0.001`, `0.999` | Lower and upper timestep sampling bounds. |

Condition dropout is required for classifier-free guidance to work at inference time.
Very low dropout can weaken CFG behavior; very high dropout can reduce conditioning
quality.

### VoiceDesign and Caption Warmup

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `use_caption_condition` | `False` | Model config field that enables caption conditioning and disables speaker/reference conditioning. |
| `caption_warmup` / `--caption-warmup` | `False` | During early training, updates only caption-only parameters. |
| `caption_warmup_steps` / `--caption-warmup-steps` | `0` | Number of optimizer steps for caption-only warmup. |

`caption_warmup` is useful when adapting a base architecture to VoiceDesign because the
caption branch may need to catch up before normal joint training. `warmup_steps` still
controls the learning-rate scheduler; `caption_warmup_steps` controls which parameters
receive gradients during the caption warmup phase.

The public VoiceDesign checkpoint is currently v2-based. Use the v2 VoiceDesign configs
for this path until a v3 VoiceDesign checkpoint is released.

### Duration Predictor

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `use_duration_predictor` | `False` | Enables duration prediction in the model. |
| `train_mode` / `--train-mode` | `rf` | `rf` trains the RF model; `duration_only` freezes non-duration parameters and trains only the duration predictor. |
| `duration_loss_weight` / `--duration-loss-weight` | `0.1` | Weight of duration loss when training jointly with RF loss. |
| `duration_speaker_dropout` / `--duration-speaker-dropout` | `0.1` | Dropout for speaker features in duration prediction. |
| `duration_huber_delta` / `--duration-huber-delta` | `0.1` | Huber delta for the log-duration regression loss. |
| `duration_architecture` | `token_sum_adarn_zero_no_aux` | Duration predictor architecture. |
| `duration_hidden_dim`, `duration_layers`, `duration_dropout` | `1024`, `3`, `0.1` | Duration predictor residual SwiGLU width, depth, and dropout. |
| `duration_attention_heads` | `8` | Attention heads used by pooled duration variants. It is kept in config for shared DP construction; the token-sum phase2 config does not use pooling attention. |
| `duration_speaker_fusion` | `adarn_zero` | Speaker conditioning mode. `token_sum_adarn_zero_no_aux` requires `adarn_zero`. |
| `duration_token_init_frames` | `9.0` | Initial frames-per-token for token-sum duration heads. Initial predictions are roughly `valid_token_count * duration_token_init_frames`. |
| `duration_aux_dim` | `14` | Size of auxiliary duration features produced by the dataset pipeline. Token-sum no-aux models validate/pass this tensor for pipeline compatibility but do not use it in the prediction. |

The duration target is `log1p(num_frames)` and the runtime converts predictions back to
latent frames for inference. The v3 base release uses this predictor as an integrated
part of inference. Use `duration_only` when you want to add or refine duration prediction
without updating the main RF model.

The current v3 phase2 duration config uses a token contribution sum predictor after
ablation against pooled-vector speaker fusion variants. It keeps the encoded text sequence,
conditions residual SwiGLU blocks with speaker AdaRN-Zero, predicts a non-negative
per-token frame contribution with `softplus`, sums those contributions under the text
mask, and returns `log1p(total_frames)`.

### LoRA Fine-Tuning

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `lora_enabled` / `--lora` | `False` | Enables PEFT LoRA fine-tuning. |
| `lora_r` / `--lora-r` | `16` | LoRA rank. Higher values increase trainable capacity and checkpoint size. |
| `lora_alpha` / `--lora-alpha` | `32` | LoRA scaling factor. |
| `lora_dropout` / `--lora-dropout` | `0.0` | Dropout inside LoRA layers. |
| `lora_bias` / `--lora-bias` | `none` | Bias handling passed to PEFT: `none`, `all`, or `lora_only`. |
| `lora_target_modules` / `--lora-target-modules` | `diffusion_attn` | Preset name, regex, or comma-separated module suffix list. |
| `lora_modules_to_save` / `--lora-modules-to-save` | `auto` | Extra modules saved with the adapter. `auto` saves `duration_predictor` for v3 duration-enabled models. Use `none` to disable. |

For inference, pass the saved adapter directory to `infer.py --lora-adapter`.
Dynamic LoRA loading requires `--compile-model` to remain disabled.

Common presets:

- `diffusion_attn`: small, focused adaptation of diffusion attention.
- `diffusion_attn_mlp`: broader diffusion adaptation.
- `conditioning`: adapts conditioning projections.
- `all_attn_mlp`: broad adaptation across encoders and diffusion blocks.
- `all_linear`: largest preset; useful only when you intentionally want broad coverage.

LoRA checkpoints are saved as adapter directories. During conversion, adapter weights are
merged into the base model so the exported `.safetensors` can be used directly for
inference.

### Speaker Inversion

Speaker Inversion freezes the entire model and trains only a small set of learned speaker
embedding tokens. The output is a `.speaker.safetensors` file used at inference with
`--ref-embed`. An example config is provided at
`configs/train_500m_v3_speaker_inversion.yaml`.

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `speaker_inversion_enabled` / `--speaker-inversion` | `False` | Enables Speaker Inversion mode. All model parameters are frozen; only the speaker embedding tokens receive gradients. |
| `speaker_inversion_tokens` / `--speaker-inversion-tokens` | `16` | Number of learned speaker embedding tokens. |
| `speaker_inversion_init_std` / `--speaker-inversion-init-std` | `0.02` | Standard deviation for random initialization of the embedding tokens. |
| `speaker_inversion_init_embedding` / `--speaker-inversion-init-embedding` | `None` | Path to an existing `.speaker.safetensors` to resume from or warm-start a new optimization. |

Use `--init-checkpoint` with the base model weights and `--manifest` with audio from the
target speaker. All condition dropout values should be set to `0.0` so the embedding
learns to represent the target speaker unconditionally.

### Logging, Validation, and DDP

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `log_every` / `--log-every` | `100` | Logging interval in optimizer steps. |
| `save_every` / `--save-every` | `1000` | Checkpoint interval in optimizer steps. |
| `checkpoint_best_n` / `--checkpoint-best-n` | `0` | Keeps best validation checkpoints when validation is enabled; otherwise limits periodic checkpoint count. |
| `valid_ratio` / `--valid-ratio` | `0.0` | Splits a ratio of the manifest for validation. |
| `valid_every` / `--valid-every` | `0` | Validation interval. Set `<=0` to disable validation. |
| `wandb_enabled` / `--wandb` | `False` | Enables Weights & Biases logging. |
| `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_mode` | varies | W&B run metadata and mode. |
| `ddp_find_unused_parameters` / `--ddp-find-unused-parameters` | `False` | Enables DDP unused-parameter detection for conditional branches. |
| `progress` / `--progress` | `True` | Enables tqdm progress bars. |
| `progress_all_ranks` / `--progress-all` | `False` | Shows progress bars for all DDP ranks. |
| `seed` / `--seed` | `0` | Random seed for training setup and data split behavior. |

For multi-GPU training, launch with `uv run torchrun --nproc_per_node N train.py ...`.
The configured `batch_size` is per process, so the effective global batch size is
`batch_size * gradient_accumulation_steps * world_size`.

## Manifest Preparation Parameters

`prepare_manifest.py` encodes dataset audio into DACVAE latents and writes a JSONL
manifest consumed by `train.py`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--dataset` | required | Hugging Face dataset name or local dataset script/path accepted by `datasets.load_dataset`. |
| `--config` | `None` | Dataset config/subset. |
| `--split` | `train` | Dataset split to read. |
| `--data-files` | `None` | Optional data file paths/globs or split-qualified entries. |
| `--audio-column` | required | Column containing audio. |
| `--text-column` | required | Column containing transcript text. |
| `--text-normalize` | `True` | Applies Irodori-TTS text normalization before writing manifest text. |
| `--caption-column` | `None` | Optional style caption column. Written as `caption` in the manifest. |
| `--speaker-column` | `None` | Optional speaker/source column. Can be specified multiple times or as comma-separated names. |
| `--speaker-id-prefix` | dataset name | Namespace prefix for generated `speaker_id` values. |
| `--output-manifest` | required | Output JSONL path. |
| `--latent-dir` | required | Directory where `.pt` latent files are written. |
| `--normalize-db` | `-16.0` | Loudness normalization before codec encode. Use `none` to disable. |
| `--target-sample-rate` | `None` | Optional decode sample rate. |
| `--min-sample-rate` | `0` | Skips decoded samples below this sample rate. |
| `--max-seconds` | `None` | Trims source audio before encode. |
| `--max-samples` | `None` | Maximum number of accepted samples to write per rank. |
| `--num-gpus` | `None` | Spawns local multiprocessing with one process per GPU. |
| `--shard-strategy` | `auto` | Sample sharding strategy for multiprocessing. |
| `--merge-output` | `False` | Merges per-rank manifest shards after multi-GPU preprocessing. |
| `--streaming` | `False` | Loads the dataset in streaming mode. |

For speaker-conditioned training, include a stable `speaker_id`. For VoiceDesign
training, include `caption`; `speaker_id` is optional because the model disables the
speaker/reference branch when `use_caption_condition: true`.

## Tuning Recipes

### Better Text Adherence

Start with the default `--num-steps 40`. If pronunciation or text following is weak,
try a slightly higher `--cfg-scale-text`. If artifacts increase, back off the scale
before increasing other guidance values.

### Stronger Speaker Similarity

Use clean reference audio and keep `--ref-normalize-db` enabled. Then try increasing
`--cfg-scale-speaker` modestly. If that is not enough, test a moderate
`--speaker-kv-scale` with the default `--speaker-kv-min-t 0.9`.

### Faster Inference

Reduce `--num-steps`, keep `--context-kv-cache` enabled, and use `--decode-mode batch`
when VRAM allows. On supported CUDA GPUs, try `--model-precision bf16` and
`--codec-precision bf16`. Use `--compile-model` when serving many requests with similar
lengths.

### Lower VRAM Inference

Use `--decode-mode sequential`, keep `--num-candidates 1`, and prefer `fp32`/`bf16`
settings that are stable on your device. Avoid large `--seconds` values and high
candidate counts.

### Fine-Tuning Released Weights

Use `--init-checkpoint` with the released `.safetensors`, choose the closest YAML config,
and start with LoRA unless you intentionally need full-model updates. For small datasets,
keep validation enabled with a small `valid_ratio` and monitor overfitting.
