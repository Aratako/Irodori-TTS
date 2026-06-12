"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents."""

from . import _compat  # noqa: F401 — must run before dacvae/audiotools imports

from .config import ModelConfig, SamplingConfig, TrainConfig
from .lora import LORA_TARGET_PRESETS
from .model import TextToLatentRFDiT
from .tokenizer import ByteTokenizer, PretrainedTextTokenizer

__all__ = [
    "ByteTokenizer",
    "LORA_TARGET_PRESETS",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "SamplingConfig",
    "TextToLatentRFDiT",
    "TrainConfig",
]
