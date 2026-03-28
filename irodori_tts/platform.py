from __future__ import annotations

import os
import sys


def is_macos() -> bool:
    return sys.platform == "darwin"


def auto_enable_mps_fallback() -> bool:
    """
    Enable PyTorch CPU fallback for unsupported MPS ops unless the user already chose.
    """
    if not is_macos():
        return False
    if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
        return False
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return True


def default_codec_device_for_runtime(model_device: str) -> str:
    """
    Keep the main model on MPS when available, but default the codec to CPU on macOS.
    This avoids common DAC/torchaudio/MPS incompatibilities while preserving model speed.
    """
    normalized = str(model_device).strip().lower()
    if normalized == "mps":
        return "cpu"
    return normalized
