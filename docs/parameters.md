# Irodori-TTS Parameter Guide

This document explains the main inference and training parameters used by Irodori-TTS.

## Version Notes

`main` targets the unified `Aratako/Irodori-TTS-v4.1-Small` release. One checkpoint supports
text, speaker/reference, and caption conditioning, including multiple reference clips with
a combined trained limit of 120 seconds.

- v4-Small uses a shared, fine-tuned ModernBERT backbone for text and captions, with separate
  condition projectors. Its safetensors metadata and bundled tokenizer are sufficient to
  reconstruct the trained encoder without loading the original ModernBERT weights.
- v4-Small includes its duration predictor and estimates output length automatically when
  `--seconds` is omitted.
- Released v2/v3 checkpoints remain supported for inference. v2 checkpoints use fixed
  30-second targets; v3 base and VoiceDesign checkpoints include duration prediction.
- Legacy v2 VoiceDesign is caption-only. Legacy v3 VoiceDesign and v4-Small support
  text + speaker/reference + caption conditioning.

## Inference Parameters

### Checkpoint Selection

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--checkpoint` | required unless `--hf-checkpoint` is set | Local `.pt` or `.safetensors` checkpoint. Use this for converted local checkpoints or downloaded model files that you want to reference directly. |
| `--hf-checkpoint` | required unless `--checkpoint` is set | Hugging Face repo id or `repo/subfolder`. The runtime downloads `model.safetensors` and bundled tokenizer assets from the selected location. |
| `--lora-adapter` | `None` | Optional PEFT LoRA adapter directory loaded dynamically at inference time. The adapter is not merged into the base checkpoint. |
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | DACVAE codec used to encode reference audio and decode generated latents. It should normally match the checkpoint metadata. |

Use either `--checkpoint` or `--hf-checkpoint`, not both.

### Text, Caption, and Reference Conditioning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--text` | required | Text to synthesize. It is tokenized with the checkpoint's text tokenizer. |
| `--caption` | `None` | Voice and style-control text for v4-Small and other VoiceDesign checkpoints. Ignored or ineffective for checkpoints without caption conditioning. |
| `--ref-wav` | `None` | Reference waveform used for speaker/style conditioning in speaker-enabled checkpoints, including v4-Small. |
| `--ref-wavs PATH [PATH ...]` | `None` | Multiple reference waveforms. Each clip is encoded independently, then the raw latents are concatenated in input order and capped by `--max-ref-seconds`. For v4-Small, multiple shorter clips from the same speaker match long-reference training. |
| `--ref-latent` | `None` | Precomputed reference latent (`.pt`) used instead of encoding `--ref-wav` at inference time. Useful for repeated inference with the same reference. |
| `--ref-latents PATH [PATH ...]` | `None` | Multiple precomputed reference latents concatenated in input order and then capped by `--max-ref-seconds`. |
| `--no-ref` | `False` | Disables speaker/reference conditioning for the request. Use this for v4-Small text-only/text+caption-only inference and legacy caption-only checkpoints. |
| `--ref-embed` | `None` | Speaker Inversion embedding (`.speaker.safetensors`) path. Mutually exclusive with waveform/latent reference options and `--no-ref`. Use the file produced by Speaker Inversion training instead of reference audio. |
| `--max-ref-seconds` | checkpoint metadata or `30.0` | Caps the reference duration for both waveform and precomputed-latent inputs. When omitted, inference uses the checkpoint's `ref_max_seconds`; v4-Small specifies 120 seconds and checkpoints without metadata fall back to 30 seconds. Set `<=0` only when you intentionally want to disable the cap. |
| `--ref-normalize-db` | `-16.0` | Loudness target applied to reference audio before DACVAE encode. This normalization was used when training the codec, so keeping the default is recommended. Use `none` only for controlled experiments. |
| `--ref-ensure-max` | `True` | When loudness normalization is disabled, scales the reference down only if peak amplitude exceeds `1.0`. In normal use, prefer leaving loudness normalization enabled instead of relying on this fallback. |
| `--max-text-len` | checkpoint metadata or `256` | Maximum text token length. Longer text is truncated. Keeping the checkpoint/training-time setting is recommended. |
| `--max-caption-len` | checkpoint metadata or `max_text_len` | Maximum caption token length for VoiceDesign checkpoints. Keeping the checkpoint/training-time setting is recommended. |

Reference audio is a conditioning signal, not the generated target. Short, clean
references may work well, but the more important point is to avoid music, noise, or
multiple speakers.

For v4-Small long-reference cloning, prefer multiple clean, shorter clips from the same
speaker. Training formed long references by randomly concatenating short utterances, and the
reported reference-length evaluation used the same construction. Approximately 30 seconds of
combined reference audio captured most of the measured speaker-similarity gain. A single
uninterrupted long recording is accepted, but its benefit has not been evaluated.

The reference options are mutually exclusive: use one of `--ref-wav`, `--ref-wavs`,
`--ref-latent`, `--ref-latents`, `--ref-embed`, or `--no-ref`. For plural inputs,
ordering is significant. Clips are concatenated in the order given, and once the combined
latent reaches the maximum reference length, later clips have no effect.

The checkpoint-aware default gives v4-Small its trained 120-second limit while preserving
the 30-second fallback for legacy checkpoints without this metadata.
`convert_checkpoint_to_safetensors.py` copies `ref_max_seconds` from the training checkpoint
into safetensors inference metadata.

For speaker-conditioned checkpoints, `--ref-latent` is the fastest path when the same
speaker reference is reused many times.

### Duration Control

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--seconds` | `None` | Manual output duration. If set, it always overrides the default duration behavior. |
| `--duration-scale` | `1.0` | Multiplies the predicted duration when duration prediction is used. Values above `1.0` produce longer audio; values below `1.0` produce shorter audio. |

The recommended duration behavior depends on the checkpoint:

- v2 checkpoints, including the v2 VoiceDesign release, were trained with fixed 30-second
  targets. Setting a different duration is not recommended because it moves inference
  away from the training setup.
- v4-Small and the v3 checkpoint families use variable-length targets and an integrated
  duration predictor. Leaving `--seconds` unset is recommended so the model can choose the
  duration automatically. Manual `--seconds` remains available for exact control.

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
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
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
`1 + number_of_enabled_cfg_conditions` during CFG-active steps. Three-branch v4-Small
inference can enable text, speaker, and caption CFG at the same time.
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
important for v4-Small and v3 checkpoints because they predict a more appropriate output
length.
If valid audio is being trimmed too aggressively, disable `--trim-tail` first.
Adjust the tail thresholds only when you need fine control over the trimming heuristic.

## Training Parameters

Training is configured through YAML files with `model` and `train` sections. CLI options
override YAML values when explicitly provided.

### Data and Checkpoint Flow

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--config` | required | YAML file containing the `model` and `train` settings. |
| `--manifest` | required | JSONL manifest produced by `prepare_manifest.py`. Each row must include `text` and `latent_path`; `speaker_id` and `caption` are optional depending on the model. `caption` may be either a string or a list of strings; list captions are sampled randomly each time the row is loaded. |
| `--output-dir` | `outputs/irodori_tts` | Directory for checkpoints, trainer state, configs, and logs. |
| `--init-checkpoint` | `None` | Initializes model weights from a `.pt` or `.safetensors` checkpoint, then starts optimizer/scheduler state from scratch. |
| `--resume` | `None` | Restores full training state from a training checkpoint. Use `.pt` for full-model runs and checkpoint directories for LoRA runs. |

Use `--init-checkpoint` for fine-tuning from released inference weights. Use `--resume`
only to continue an interrupted training run.

### Model Config

| Field | Notes |
|-------|-------|
| `latent_dim` | DACVAE latent dimension expected by the model. v4-Small and the released v2/v3 checkpoints use `32`. |
| `latent_patch_size` | Number of latent frames grouped per model token. |
| `model_dim`, `num_layers`, `num_heads`, `mlp_ratio` | Main diffusion transformer width, depth, attention heads, and MLP expansion. |
| `text_tokenizer_repo`, `text_vocab_size`, `text_add_bos` | Tokenizer and vocabulary settings for the text encoder. In pretrained mode, the same repository supplies the tokenizer and backbone; `text_vocab_size` is retained for checkpoint compatibility but is not used to build an embedding. |
| `text_encoder_revision` | Optional Hugging Face revision used for both the tokenizer and pretrained backbone. Pin a commit SHA for reproducible checkpoints. |
| `text_encoder_type` | `scratch` (default) builds the Irodori text Transformer. `pretrained` initializes a shared, trainable Hugging Face backbone (or only its encoder for encoder-decoder models) and adds a projector into `text_dim`. |
| `pretrained_projector_type` | Projector used in pretrained mode: `linear` (default) or `residual_mlp`. The latter adds a zero-initialized residual `RMSNorm -> Linear -> SiLU -> Linear` branch around the base Linear mapping. |
| `pretrained_projector_hidden_ratio`, `pretrained_projector_dropout` | Residual MLP hidden width relative to the condition output dimension and its dropout. Defaults are `2.0` and `0.0`. |
| `text_dim`, `text_layers`, `text_heads`, `text_mlp_ratio` | Text condition output size and scratch encoder architecture. In pretrained mode only `text_dim` is used (as the projector output size). |
| `speaker_dim`, `speaker_layers`, `speaker_heads`, `speaker_patch_size`, `speaker_mlp_ratio` | Reference/speaker encoder size. Used when resolved speaker conditioning is enabled. |
| `use_caption_condition` | Enables the VoiceDesign caption path. |
| `use_speaker_condition` | Optional explicit speaker-conditioning flag. `null` keeps legacy behavior: caption-free configs enable speaker conditioning, caption-enabled configs disable it. v4-Small explicitly enables it together with caption conditioning. |
| `caption_*` fields | Caption encoder tokenizer and architecture. When left unset, many fields inherit the corresponding text settings. In pretrained mode the caption tokenizer repository must match the text repository; one backbone is shared while text and caption keep separate trainable projectors. |
| `timestep_embed_dim`, `adaln_rank`, `norm_eps` | Diffusion conditioning and normalization parameters. |
| `use_duration_predictor` | Enables integrated duration prediction. |
| `duration_*` fields | Duration predictor architecture, hidden size, depth, dropout, speaker conditioning, and token-sum initialization. |

Architecture fields should match the checkpoint you initialize from. Changing dimensions,
layer counts, vocabulary sizes, or conditioning branches usually prevents checkpoint
loading unless you are intentionally training from scratch or using an upgrade path
handled by the code.

With `text_encoder_type: pretrained`, the Hugging Face backbone is optimized with the TTS model
and saved in full for exact resume and inference. Text and caption calls share these weights and
accumulate gradients into the same backbone. The checkpoint converter embeds the encoder
architecture config in `model.safetensors` and exports its tokenizer under `tokenizer/`. Inference
prefers these bundled assets; legacy checkpoints without them continue to resolve their tokenizer
from `text_tokenizer_repo`.

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
| `rf_loss_mode` / `--rf-loss-mode` | `echo` | RF loss normalization. `utterance_mean` averages per utterance and is used by v4-Small and variable-length v3 configs. |

The v2 configs use fixed 30-second targets. v4-Small and the variable-length v3 configs set
`fixed_target_latent_steps: null`, `fixed_target_full_mask: false`, and
`rf_loss_mode: utterance_mean` so samples can keep their natural lengths.

### Optimizer and Schedule

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `precision` / `--precision` | `bf16` | Forward-pass compute precision. Weights and optimizer states remain FP32. |
| `allow_tf32` / `--tf32` | `False` | Enables TF32 CUDA kernels for speed. |
| `compile_model` / `--compile-model` | `False` | Enables `torch.compile` during training. |
| `gradient_checkpointing` / `--gradient-checkpointing` | `False` | Enables activation checkpointing on diffusion blocks and, when supported, a trainable pretrained text encoder. Reduces memory usage at the cost of extra compute. |
| `optimizer` / `--optimizer` | `muon` | `muon` or `adamw`. |
| `learning_rate` / `--lr` | `1e-4` | Base learning rate. |
| `pretrained_text_encoder_learning_rate` / `--pretrained-text-encoder-learning-rate` | `1e-5` | AdamW learning rate for a trainable pretrained text/caption backbone. It receives the same scheduler multiplier as the main LR and should be tuned for the selected backbone. |
| `weight_decay` / `--weight-decay` | `0.01` | Weight decay for optimizer groups that use it. |
| `adam_beta1`, `adam_beta2`, `adam_eps` | `0.9`, `0.999`, `1e-8` | AdamW hyperparameters. |
| `muon_momentum` / `--muon-momentum` | `0.95` | Momentum used by Muon. |
| `lr_scheduler` / `--lr-scheduler` | `none` | `none`, `cosine`, or `wsd`. |
| `warmup_steps` / `--warmup-steps` | `0` | Number of optimizer steps for LR warmup. |
| `stable_steps` / `--stable-steps` | `0` | Stable plateau length for the WSD schedule. |
| `min_lr_scale` / `--min-lr-scale` | `0.1` | Minimum LR multiplier at the end of decay. |
| `grad_clip_norm` | `1.0` | Gradient clipping norm. Currently configured through YAML. |

The v4-Small and full-training v3 example configs use `optimizer: muon` and
`lr_scheduler: wsd`.
When changing effective batch size, revisit the learning rate and warmup length together.
During full training, all pretrained-backbone parameters use a dedicated AdamW group, including
matrix weights that would otherwise be assigned to Muon. The remaining TTS model keeps the
selected main optimizer. During LoRA training, PEFT freezes base parameters and saves only LoRA
weights plus explicitly selected `modules_to_save`; LoRA can therefore also be used with a
pretrained text encoder.

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
| `use_caption_condition` | `False` | Model config field that enables caption conditioning. |
| `use_speaker_condition` | `None` | Explicit speaker branch control. `None` preserves legacy caption-implies-no-speaker behavior. |
| `caption_warmup` / `--caption-warmup` | `False` | During early training, updates only caption-only parameters. |
| `caption_warmup_steps` / `--caption-warmup-steps` | `0` | Number of optimizer steps for caption-only warmup. |
| `pretrained_projector_warmup_steps` / `--pretrained-projector-warmup-steps` | `0` | When upgrading to a pretrained encoder, update only the new text/caption projectors for this many optimizer steps. Afterward the rest of the TTS model and a trainable backbone update normally. |

`caption_warmup` is useful when adapting a base architecture to VoiceDesign because the
caption branch may need to catch up before normal joint training. `warmup_steps` still
controls the learning-rate scheduler; `caption_warmup_steps` controls which parameters
receive gradients during the caption warmup phase.

Pretrained projector warmup is intended for `--init-checkpoint` conversion from a trained
scratch text encoder. It requires `train_mode: rf` and is incompatible with caption warmup,
LoRA, duration-only mode, and Speaker Inversion. The old text/caption encoder keys are the
only checkpoint keys discarded; compatible DiT, speaker, duration, normalization, and
conditioning parameters must still match and are loaded. Use `--init-checkpoint`, because
`--resume` expects an already-upgraded checkpoint and optimizer state.

When the pretrained backbone width matches `text_dim`, the new linear projector starts as
identity. When the dimensions differ, it uses Xavier initialization. The appropriate backbone
learning rate depends on the selected model and training setup.

`residual_mlp` preserves that base initialization and zero-initializes its final residual
projection, so its initial output is exactly equal to `linear`. The residual branch is then
learned during projector-only warmup. Set `pretrained_projector_type: linear` for the
lower-capacity baseline.

The v4-Small config trains RF and duration losses jointly with text, speaker/reference,
and caption conditions. Its ModernBERT backbone remains trainable through a dedicated
AdamW parameter group.

### Duration Predictor

| Parameter / Field | Default in dataclass | Notes |
|-------------------|----------------------|-------|
| `use_duration_predictor` | `False` | Enables duration prediction in the model. |
| `train_mode` / `--train-mode` | `rf` | `rf` trains the RF model; `duration_only` freezes non-duration parameters and trains only the duration predictor. |
| `duration_loss_weight` / `--duration-loss-weight` | `0.1` | Weight of duration loss when training jointly with RF loss. |
| `duration_backprop_to_condition` / `--duration-backprop-to-condition` | `False` | In joint `train_mode: rf`, allows duration loss to update text/caption projectors and the speaker condition path. `duration_only` always detaches these condition states. |
| `duration_speaker_dropout` / `--duration-speaker-dropout` | `0.1` | Dropout for speaker features in duration prediction. |
| `duration_caption_dropout` / `--duration-caption-dropout` | `0.1` | Dropout for caption features in duration prediction. |
| `duration_huber_delta` / `--duration-huber-delta` | `0.1` | Huber delta for the log-duration regression loss. |
| `duration_architecture` | `token_sum_adarn_zero_no_aux` | Duration predictor architecture. v4-Small uses `token_sum_dual_adarn_zero_no_aux`. |
| `duration_hidden_dim`, `duration_layers`, `duration_dropout` | `1024`, `3`, `0.1` | Duration predictor residual SwiGLU width, depth, and dropout. |
| `duration_attention_heads` | `8` | Attention heads used by pooled duration variants. It is kept in config for shared DP construction; the token-sum phase2 config does not use pooling attention. |
| `duration_speaker_fusion` | `adarn_zero` | Speaker conditioning mode. `token_sum_adarn_zero_no_aux` requires `adarn_zero`. |
| `duration_caption_fusion` | `adarn_zero` | Caption conditioning mode for duration prediction. `token_sum_dual_adarn_zero_no_aux` requires `adarn_zero`. |
| `duration_caption_pooling` | `masked_mean` | Caption pooling strategy used before caption-conditioned duration fusion. |
| `duration_token_init_frames` | `9.0` | Initial frames-per-token for token-sum duration heads. Initial predictions are roughly `valid_token_count * duration_token_init_frames`. |
| `duration_aux_dim` | `14` | Size of auxiliary duration features produced by the dataset pipeline. Token-sum no-aux models validate/pass this tensor for pipeline compatibility but do not use it in the prediction. |

The duration target is `log1p(num_frames)` and the runtime converts predictions back to
latent frames for inference. v4-Small and the v3 releases use this predictor as an
integrated part of inference. Use `duration_only` when you want to add or refine duration
prediction without updating the main RF model.

The legacy v3 phase-2 duration config uses a token contribution sum predictor after
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
| `lora_modules_to_save` / `--lora-modules-to-save` | `auto` | Extra modules saved with the adapter. `auto` saves `duration_predictor` for duration-enabled models. Use `none` to disable. |

For inference, pass the saved adapter directory to `infer.py --lora-adapter`.
Dynamic LoRA loading requires `--compile-model` to remain disabled.

Common presets:

- `diffusion_attn`: small, focused adaptation of diffusion attention.
- `diffusion_attn_mlp`: broader diffusion adaptation.
- `pretrained_backbone_attn`: attention projections in the shared ModernBERT or T5Gemma2
  text/caption backbone.
- `pretrained_backbone_attn_mlp`: attention and MLP projections in either supported backbone.
- `conditioning`: adapts conditioning projections.
- `all_attn_mlp`: broad adaptation across encoders and diffusion blocks; includes the supported
  ModernBERT or T5Gemma2 backbone when present.
- `all_linear`: largest preset; useful only when you intentionally want broad coverage.

The pretrained-backbone presets explicitly support ModernBERT and the T5Gemma2 encoder. Other
Hugging Face architectures are not matched automatically because their attention and MLP module
names vary; provide a custom target-module regex or suffix list for another backbone.

LoRA checkpoints are saved as adapter directories. During conversion, adapter weights are
merged into the base model so the exported `.safetensors` can be used directly for
inference. If the base checkpoint has a bundled `tokenizer/`, conversion copies those exact
assets to the output instead of resolving the original Hugging Face tokenizer again.
Use `configs/train_v4_small_lora.yaml` as the v4-Small LoRA example.

### Speaker Inversion

Speaker Inversion freezes the entire model and trains only a small set of learned speaker
embedding tokens. The output is a `.speaker.safetensors` file used at inference with
`--ref-embed`. An example config is provided at
`configs/train_v4_small_speaker_inversion.yaml`.

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

For v4-style training, include both `caption` and a stable `speaker_id` so all three
condition branches can be trained. Legacy v2 VoiceDesign requires `caption` but does not
use speaker/reference conditioning.

## Tuning Recipes

### Better Text Adherence

Start with the default `--num-steps 40`. If pronunciation or text following is weak,
try a slightly higher `--cfg-scale-text`. If artifacts increase, back off the scale
before increasing other guidance values.

### Stronger Speaker Similarity

Use clean reference audio and keep `--ref-normalize-db` enabled. With v4-Small, first try
multiple shorter clips from the same speaker totaling approximately 30 seconds. Then try increasing
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
