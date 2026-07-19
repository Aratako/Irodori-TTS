#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import _load_checkpoint_for_inference
from irodori_tts.lora import (
    LORA_METADATA_NAME,
    LORA_TRAINER_STATE_NAME,
    checkpoint_state_uses_lora,
    is_lora_adapter_dir,
    load_lora_adapter,
)
from irodori_tts.model import TextToLatentRFDiT

CONFIG_META_KEY = "config_json"
INFERENCE_CONFIG_KEYS = ("max_text_len", "max_caption_len", "fixed_target_latent_steps")


def _default_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}.safetensors"
    return input_path.with_suffix(".safetensors")


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    load_params = inspect.signature(torch.load).parameters
    if "weights_only" in load_params:
        load_kwargs["weights_only"] = True
    if "mmap" in load_params:
        load_kwargs["mmap"] = True

    payload = torch.load(path, **load_kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")
    return payload


def _extract_model_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload

    if not isinstance(raw_model, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary under 'model'.")

    model_state: dict[str, torch.Tensor] = {}
    for key, value in raw_model.items():
        if not isinstance(key, str):
            raise ValueError(f"Model state key must be str, got {type(key)!r}.")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Model state '{key}' is not a tensor (got {type(value)!r}).")
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        model_state[key] = tensor

    if not model_state:
        raise ValueError("Model state is empty.")
    return model_state


def _extract_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Checkpoint is missing 'model_config' dictionary required for inference compatibility."
        )
    return model_cfg


def _extract_train_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    train_cfg = payload.get("train_config")
    if train_cfg is None:
        return None
    if not isinstance(train_cfg, dict):
        raise ValueError("Checkpoint 'train_config' must be a dictionary when present.")
    return train_cfg


def _extract_inference_config(payload: dict[str, Any]) -> dict[str, int]:
    raw = _extract_train_config(payload)
    if raw is None:
        return {}

    inference_cfg: dict[str, int] = {}
    for key in INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if isinstance(value, int):
            inference_cfg[key] = int(value)
    return inference_cfg


def _build_flat_config(payload: dict[str, Any]) -> dict[str, Any]:
    flat_cfg = dict(_extract_model_config(payload))
    flat_cfg.update(_extract_inference_config(payload))
    return flat_cfg


def _build_safetensors_metadata(*, flat_config: dict[str, Any]) -> dict[str, str]:
    return {
        CONFIG_META_KEY: json.dumps(flat_config, ensure_ascii=False, separators=(",", ":")),
    }


def _load_saved_config(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    config_path = adapter_dir / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Adapter config root must be a mapping: {config_path}")
        model_cfg = payload.get("model")
        train_cfg = payload.get("train")
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Adapter config is missing model section: {config_path}")
        if train_cfg is not None and not isinstance(train_cfg, dict):
            raise ValueError(f"Adapter config train section must be a mapping: {config_path}")
        return model_cfg, train_cfg

    trainer_state = _load_checkpoint(adapter_dir / LORA_TRAINER_STATE_NAME)
    model_cfg = trainer_state.get("model_config")
    train_cfg = trainer_state.get("train_config")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Adapter trainer state is missing model_config: {adapter_dir}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Adapter trainer state train_config must be a mapping: {adapter_dir}")
    return model_cfg, train_cfg


def _load_adapter_metadata(adapter_dir: Path) -> dict[str, Any] | None:
    metadata_path = adapter_dir / LORA_METADATA_NAME
    if not metadata_path.is_file():
        trainer_state_path = adapter_dir / LORA_TRAINER_STATE_NAME
        if not trainer_state_path.is_file():
            return None
        trainer_state = _load_checkpoint(trainer_state_path)
        raw = trainer_state.get("base_init")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(
                f"Adapter trainer state base_init must be a mapping: {trainer_state_path}"
            )
        return raw

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter metadata root must be a mapping: {metadata_path}")
    raw = payload.get("base_init")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Adapter metadata base_init must be a mapping: {metadata_path}")
    return raw


def _resolve_base_checkpoint(adapter_dir: Path, override: str | None) -> Path:
    if override:
        return _normalize_checkpoint_path(override)

    metadata = _load_adapter_metadata(adapter_dir)
    if metadata is None:
        raise ValueError(
            "Adapter checkpoint does not record a base checkpoint path. Pass --base-checkpoint."
        )

    checkpoint_path = metadata.get("checkpoint_path")
    if (
        metadata.get("mode") != "checkpoint"
        or not isinstance(checkpoint_path, str)
        or not checkpoint_path
    ):
        raise ValueError(
            "Adapter checkpoint cannot be merged without a base checkpoint path. Pass --base-checkpoint."
        )
    return _normalize_checkpoint_path(checkpoint_path)


def _initialize_embedding_from_pretrained(
    embedding: torch.nn.Embedding,
    *,
    repo_id: str,
) -> None:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_backbone = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pretrained_embedding = text_backbone.get_input_embeddings()
    if pretrained_embedding is None:
        raise ValueError(f"Pretrained model has no input embeddings: {repo_id}")
    src_weight = pretrained_embedding.weight.detach().to(device="cpu", dtype=torch.float32)
    tgt_weight = embedding.weight
    src_vocab, src_dim = tuple(src_weight.shape)
    tgt_vocab, tgt_dim = tuple(tgt_weight.shape)
    if src_dim != tgt_dim:
        raise ValueError(
            f"Embedding hidden size mismatch: pretrained={src_dim} model={tgt_dim} for repo={repo_id}."
        )

    copy_rows = min(src_vocab, tgt_vocab)
    with torch.no_grad():
        tgt_weight[:copy_rows].copy_(
            src_weight[:copy_rows].to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        )


def _initialize_caption_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
) -> None:
    if model.caption_encoder is None:
        raise RuntimeError(
            "Caption embedding initialization requested but caption encoder is absent."
        )
    _initialize_embedding_from_pretrained(
        model.caption_encoder.text_embedding,
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
    )


def _checkpoint_uses_caption_condition(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_caption_condition:
            return True
    return any(
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def _checkpoint_uses_duration_predictor(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_duration_predictor:
            return True
    return any(key.startswith("duration_predictor.") for key in state_dict)


def _is_caption_only_parameter(key: str) -> bool:
    return (
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
    )


def _is_speaker_only_parameter(key: str) -> bool:
    return (
        key.startswith("speaker_encoder.")
        or key.startswith("speaker_norm.")
        or ".wk_speaker." in key
        or ".wv_speaker." in key
    )


def _is_duration_only_parameter(key: str) -> bool:
    return key.startswith("duration_predictor.")


def _load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_extra: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_extra.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    return missing_keys, skipped_shape, skipped_extra


def _validate_checkpoint_upgrade_partial_load(
    checkpoint_path: Path,
    missing_keys: list[str],
    skipped_shape: list[str],
    skipped_extra: list[str],
    *,
    allow_caption_missing: bool,
    allow_duration_missing: bool,
    allow_duration_extra: bool,
    allow_speaker_extra: bool,
) -> None:
    if skipped_shape:
        raise ValueError(
            "Checkpoint/config shape mismatch while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_shape={skipped_shape[:8]}"
        )

    unexpected_extra = skipped_extra
    if allow_speaker_extra:
        unexpected_extra = [key for key in unexpected_extra if not _is_speaker_only_parameter(key)]
    if allow_duration_extra:
        unexpected_extra = [key for key in unexpected_extra if not _is_duration_only_parameter(key)]
    if unexpected_extra:
        raise ValueError(
            "Unexpected checkpoint keys while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_extra={unexpected_extra[:8]}"
        )

    def _allowed_missing(key: str) -> bool:
        return (allow_caption_missing and _is_caption_only_parameter(key)) or (
            allow_duration_missing and _is_duration_only_parameter(key)
        )

    unexpected_missing = [key for key in missing_keys if not _allowed_missing(key)]
    if unexpected_missing:
        raise ValueError(
            "Partial init from checkpoint left unexpected parameters missing: "
            f"{checkpoint_path} missing={unexpected_missing[:8]}"
        )


def _load_adapter_checkpoint(
    adapter_dir: Path,
    *,
    base_checkpoint: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    model_cfg, train_cfg = _load_saved_config(adapter_dir)
    base_path = _resolve_base_checkpoint(adapter_dir, base_checkpoint)
    base_state, base_model_cfg, _ = _load_checkpoint_for_inference(base_path)
    resolved_model_cfg = ModelConfig(**model_cfg)

    model = TextToLatentRFDiT(resolved_model_cfg)
    checkpoint_has_caption = _checkpoint_uses_caption_condition(base_model_cfg, base_state)
    current_has_caption = bool(resolved_model_cfg.use_caption_condition)
    checkpoint_has_duration = _checkpoint_uses_duration_predictor(base_model_cfg, base_state)
    current_has_duration = bool(resolved_model_cfg.use_duration_predictor)
    drop_duration = checkpoint_has_duration and not current_has_duration
    if checkpoint_has_caption and not current_has_caption:
        raise ValueError(
            "Caption-conditioned base checkpoint cannot initialize a caption-free adapter config."
        )
    if drop_duration and not (current_has_caption and not checkpoint_has_caption):
        raise ValueError(
            "Duration-predictor base checkpoint cannot initialize a duration-free adapter config "
            "unless upgrading a caption-free base checkpoint to a caption-enabled phase-1 config."
        )
    upgrade_caption = current_has_caption and not checkpoint_has_caption
    upgrade_duration = current_has_duration and not checkpoint_has_duration
    if upgrade_caption or upgrade_duration or drop_duration:
        missing_keys, skipped_shape, skipped_extra = _load_model_state_partially(model, base_state)
        _validate_checkpoint_upgrade_partial_load(
            base_path,
            missing_keys,
            skipped_shape,
            skipped_extra,
            allow_caption_missing=upgrade_caption,
            allow_duration_missing=upgrade_duration,
            allow_duration_extra=drop_duration,
            allow_speaker_extra=(
                upgrade_caption and not resolved_model_cfg.use_speaker_condition_resolved
            ),
        )
    else:
        model.load_state_dict(base_state, strict=True)

    if upgrade_caption:
        _initialize_caption_embedding_from_pretrained(model, resolved_model_cfg)
    peft_model = load_lora_adapter(model, adapter_dir, is_trainable=False)
    if not hasattr(peft_model, "merge_and_unload"):
        raise RuntimeError("Loaded PEFT adapter does not support merge_and_unload().")
    merged = peft_model.merge_and_unload()

    flat_config = dict(model_cfg)
    if isinstance(train_cfg, dict):
        for key in INFERENCE_CONFIG_KEYS:
            value = train_cfg.get(key)
            if isinstance(value, int):
                flat_config[key] = int(value)

    merged_state: dict[str, torch.Tensor] = {}
    for key, value in merged.state_dict().items():
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        merged_state[key] = tensor
    return merged_state, flat_config, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert checkpoints (.pt or LoRA adapter dirs) to safetensors for inference. "
        )
    )
    parser.add_argument(
        "input_checkpoint",
        help="Path to source checkpoint (.pt or LoRA adapter directory).",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Base model checkpoint used to merge adapter-only LoRA checkpoints.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .safetensors path (default: input path with .safetensors suffix).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_checkpoint).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    output_path = (
        Path(args.output).expanduser() if args.output else _default_output_path(input_path)
    )
    if output_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Output must use .safetensors suffix: {output_path}")

    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")

    if is_lora_adapter_dir(input_path):
        model_state, flat_config, merged_lora = _load_adapter_checkpoint(
            input_path,
            base_checkpoint=args.base_checkpoint,
        )
    else:
        payload = _load_checkpoint(input_path)
        raw_model_state = _extract_model_state(payload)
        if checkpoint_state_uses_lora(raw_model_state):
            raise ValueError(
                "LoRA checkpoints must be passed as adapter checkpoint directories, not .pt files."
            )
        model_state = raw_model_state
        merged_lora = False
        flat_config = _build_flat_config(payload)

    metadata = _build_safetensors_metadata(
        flat_config=flat_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model_state, str(output_path), metadata=metadata)

    total_params = sum(int(t.numel()) for t in model_state.values())
    total_bytes = sum(int(t.numel()) * int(t.element_size()) for t in model_state.values())
    print(f"Input: {input_path}")
    print(f"Saved: {output_path}")
    print(f"Tensors: {len(model_state)}")
    print(f"Total params: {total_params:,}")
    print(f"Approx tensor bytes: {total_bytes / (1024**3):.2f} GiB")
    if merged_lora:
        print("Merged LoRA adapter weights into the base model before export.")


if __name__ == "__main__":
    main()
