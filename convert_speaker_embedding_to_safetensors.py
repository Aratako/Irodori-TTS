#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_INVERSION_SAFETENSORS_SUFFIX,
    default_speaker_inversion_safetensors_path,
    is_speaker_inversion_safetensors_path,
    load_speaker_inversion_payload,
    save_speaker_inversion_safetensors,
)

TORCH_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Speaker Inversion embedding checkpoint to speaker-specific "
            "safetensors format."
        )
    )
    parser.add_argument(
        "input_embedding",
        help="Path to a Speaker Inversion embedding checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. Must end with "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r}. "
            "Default: input stem + '.speaker.safetensors'."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp32",
        help="Floating-point precision for the output safetensors file (default: fp32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_embedding).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input speaker embedding not found: {input_path}")

    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else default_speaker_inversion_safetensors_path(input_path)
    )
    if not is_speaker_inversion_safetensors_path(output_path):
        raise ValueError(
            "Speaker embedding safetensors output must end with "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r}: {output_path}"
        )
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")
    assert args.dtype in TORCH_DTYPE_MAP, f"Unsupported dtype: {args.dtype}"

    payload = load_speaker_inversion_payload(input_path)
    save_speaker_inversion_safetensors(
        output_path,
        payload,
        dtype=TORCH_DTYPE_MAP[args.dtype],
    )

    embedding = payload[SPEAKER_EMBEDDING_KEY]
    print(f"Input: {input_path}")
    print(f"Saved: {output_path}")
    print(f"Speaker embedding: shape={tuple(embedding.shape)}")


if __name__ == "__main__":
    main()
