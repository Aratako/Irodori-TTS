#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import _load_checkpoint_for_inference
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.quantization import (
    DEFAULT_INT4_GROUP_SIZE,
    INT4_GROUP_SIZES,
    QUANTIZATION_CLI_CHOICES,
    QUANTIZATION_PROFILES,
    flatten_quantized_state_dict,
    normalize_quantization_type,
    parse_quantization_metadata,
    quantization_cli_name,
    quantize_model,
)


def _default_output_path(input_path: Path, quantization: str) -> Path:
    suffix = quantization_cli_name(quantization)
    return input_path.with_name(f"{input_path.stem}.{suffix}.safetensors")


def _resolve_device(raw: str) -> torch.device:
    normalized = str(raw).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.xpu.is_available():
                return torch.device("xpu")
        except AttributeError:
            pass
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    if device.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Quantization device must be one of: auto, cpu, cuda, xpu.")
    return device


def _copy_bundled_tokenizer(input_path: Path, output_path: Path) -> Path | None:
    source = input_path.parent / "tokenizer"
    if not (source / "tokenizer_config.json").is_file():
        return None
    destination = output_path.parent / "tokenizer"
    if source.resolve() != destination.resolve():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Irodori inference safetensors checkpoint with torchao."
    )
    parser.add_argument("input_checkpoint", help="Source Irodori .safetensors checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .safetensors path (default: <input>.<quantization>.safetensors).",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CLI_CHOICES,
        default="int8-weight-only",
        help="Weight/activation quantization scheme (default: int8-weight-only).",
    )
    parser.add_argument(
        "--profile",
        choices=QUANTIZATION_PROFILES,
        default="core",
        help=(
            "'core' quantizes transformer attention/MLP weights while preserving projectors, "
            "AdaLN, and duration prediction in BF16; 'all-linear' quantizes every compatible "
            "Linear."
        ),
    )
    parser.add_argument(
        "--int4-group-size",
        type=int,
        choices=INT4_GROUP_SIZES,
        default=DEFAULT_INT4_GROUP_SIZE,
        help="Group size for int4-weight-only (default: 128).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Temporary conversion device: auto, cpu, cuda, or xpu (default: auto).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_checkpoint).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if input_path.suffix.lower() != ".safetensors":
        raise ValueError(
            "Quantization requires an inference .safetensors checkpoint. Convert training or "
            "LoRA checkpoints with convert_checkpoint_to_safetensors.py first."
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path, args.quantization)
    )
    if output_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Output must use .safetensors suffix: {output_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")

    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
    if parse_quantization_metadata(source_metadata) is not None:
        raise ValueError(f"Input checkpoint is already quantized: {input_path}")

    model_state, model_cfg_dict, _, text_encoder_config = _load_checkpoint_for_inference(
        input_path
    )
    model_cfg = merge_dataclass_overrides(
        ModelConfig(),
        model_cfg_dict,
        section="checkpoint model_config",
    )
    model = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=text_encoder_config,
        load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
    )
    model.load_state_dict(model_state, assign=model_cfg.use_pretrained_text_encoder)
    del model_state

    device = _resolve_device(args.device)
    compute_dtype = torch.bfloat16
    model = model.eval().to(device=device, dtype=compute_dtype)
    quantization_type = normalize_quantization_type(args.quantization)
    selected_modules = quantize_model(
        model,
        quantization_type=quantization_type,
        profile=args.profile,
        int4_group_size=args.int4_group_size,
    )
    model = model.to(device="cpu")

    flattened_state, metadata = flatten_quantized_state_dict(
        model.state_dict(),
        base_metadata=source_metadata,
        quantization_type=quantization_type,
        profile=args.profile,
        compute_dtype=compute_dtype,
        quantized_modules=len(selected_modules),
        int4_group_size=args.int4_group_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flattened_state, str(output_path), metadata=metadata)
    tokenizer_dir = _copy_bundled_tokenizer(input_path, output_path)

    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in flattened_state.values())
    print(f"Input: {input_path}")
    print(f"Saved: {output_path}")
    print(f"Quantization: {args.quantization}")
    if args.quantization == "int4-weight-only":
        print(f"INT4 group size: {args.int4_group_size}")
    print(f"Profile: {args.profile}")
    print(f"Quantized Linear modules: {len(selected_modules):,}")
    print(f"Stored tensor bytes: {tensor_bytes / (1024**3):.2f} GiB")
    if tokenizer_dir is not None:
        print(f"Tokenizer: {tokenizer_dir}")


if __name__ == "__main__":
    main()
