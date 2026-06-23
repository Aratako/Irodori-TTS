from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _download_file(repo_id: str, filename: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        print(f"[models] already exists: {output_path}", flush=True)
        return output_path
    print(f"[models] downloading hf://{repo_id}/{filename}", flush=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_path.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != output_path.resolve() and downloaded_path.is_file():
        # hf_hub_download(local_dir=...) normally places the file at output_path.
        # This fallback keeps the configured path stable if behavior differs.
        output_path.write_bytes(downloaded_path.read_bytes())
    if not output_path.is_file():
        raise FileNotFoundError(f"Downloaded file was not found at expected path: {output_path}")
    print(f"[models] saved: {output_path}", flush=True)
    return output_path


def main() -> None:
    models_dir = Path(_env("IRODORI_LOCAL_MODELS_DIR", "_models")).resolve()
    tts_repo = _env("IRODORI_TTS_SOURCE_REPO", "Aratako/Irodori-TTS-600M-v3-VoiceDesign")
    codec_repo = _env("IRODORI_CODEC_SOURCE_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")
    tokenizer_repo = _env("IRODORI_TOKENIZER_REPO", "llm-jp/llm-jp-3-150m")

    checkpoint_raw = _env(
        "IRODORI_TTS_CHECKPOINT",
        str(models_dir / "Irodori-TTS-600M-v3-VoiceDesign" / "model.safetensors"),
    )
    codec_raw = _env(
        "IRODORI_CODEC_REPO",
        str(models_dir / "Semantic-DACVAE-Japanese-32dim" / "weights.pth"),
    )

    # Older config files may still store Hugging Face repo ids in these variables.
    # For this downloader, keep the actual files under _models by default.
    if "/" in checkpoint_raw and not checkpoint_raw.lower().endswith((".pt", ".safetensors")):
        checkpoint_raw = str(models_dir / "Irodori-TTS-600M-v3-VoiceDesign" / "model.safetensors")
    if "/" in codec_raw and not codec_raw.lower().endswith(".pth"):
        codec_raw = str(models_dir / "Semantic-DACVAE-Japanese-32dim" / "weights.pth")

    checkpoint_path = Path(checkpoint_raw)
    codec_path = Path(codec_raw)

    print(f"[models] root: {models_dir}", flush=True)
    _download_file(tts_repo, "model.safetensors", checkpoint_path)
    _download_file(codec_repo, "weights.pth", codec_path)

    # The current runtime still loads tokenizer files by repo id from the Transformers cache.
    # Preload them here so later offline runs can reuse the local cache.
    if tokenizer_repo:
        print(f"[models] caching tokenizer: {tokenizer_repo}", flush=True)
        AutoTokenizer.from_pretrained(tokenizer_repo)
        print("[models] tokenizer cache ready", flush=True)

    print("[models] download/check completed", flush=True)


if __name__ == "__main__":
    main()
