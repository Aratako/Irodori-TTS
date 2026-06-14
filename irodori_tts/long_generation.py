from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def split_long_text(text: str, max_chars: int = 80) -> list[str]:
    """Split long Japanese narration text at punctuation/newlines where possible."""
    if max_chars < 20:
        raise ValueError("max_chars must be >= 20")

    text = text.replace("\r\n", "\n").strip()
    parts = re.split(r"(?<=[縲ゑｼ・ｼ・?])\s*|\n+", text)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if buf:
                chunks.append(buf)
            buf = part

        # Fallback for very long sentences without punctuation.
        while len(buf) > int(max_chars * 1.5):
            chunks.append(buf[:max_chars])
            buf = buf[max_chars:]

    if buf:
        chunks.append(buf)
    return chunks


@dataclass(frozen=True)
class LongExportOptions:
    output_dir: Path
    output_stem: str | None = None
    output_format: str = "wav"
    pause_ms: int = 250
    keep_final_wav_when_mp3: bool = False
    keep_chunk_wavs: bool = True
    save_json: bool = True
    write_mp3_tag: bool = True
    jsonl_path: Path | None = None
    ffmpeg_exe: str | None = None


def _require_pydub():
    try:
        from pydub import AudioSegment
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pydub is required for long generation export. Install it with: uv pip install pydub mutagen"
        ) from exc
    return AudioSegment


def _write_mp3_comment_tag(mp3_path: Path, params: dict[str, Any]) -> None:
    try:
        from mutagen.id3 import ID3, COMM, TXXX
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mutagen is required for MP3 metadata. Install it with: uv pip install mutagen"
        ) from exc

    try:
        tags = ID3(str(mp3_path))
    except Exception:
        tags = ID3()

    comment_json = json.dumps(params, ensure_ascii=False, indent=2)
    try:
        tags.delall("COMM")
    except Exception:
        pass
    try:
        tags.delall("TXXX:Irodori-TTS Parameters")
    except Exception:
        pass

    # Write both standard comment and custom text frames for viewer compatibility.
    tags.add(COMM(encoding=3, lang="jpn", desc="", text=comment_json))
    tags.add(COMM(encoding=3, lang="jpn", desc="Irodori-TTS Parameters", text=comment_json))
    tags.add(TXXX(encoding=3, desc="Irodori-TTS Parameters", text=comment_json))
    tags.save(str(mp3_path), v2_version=3)


def export_long_audio(
    *,
    chunk_wavs: Iterable[Path | str],
    options: LongExportOptions,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Join chunk WAV files, export final WAV/MP3, and write metadata sidecars/tags."""
    AudioSegment = _require_pydub()
    if options.ffmpeg_exe:
        ffmpeg_path = Path(options.ffmpeg_exe)
        if ffmpeg_path.exists():
            AudioSegment.converter = str(ffmpeg_path)

    output_format = str(options.output_format).lower().strip()
    if output_format not in {"wav", "mp3"}:
        raise ValueError("output_format must be 'wav' or 'mp3'")
    if options.pause_ms < 0:
        raise ValueError("pause_ms must be >= 0")

    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_stem = options.output_stem or f"irodori_long_{stamp}"
    final_wav = output_dir / f"{output_stem}.wav"
    final_mp3 = output_dir / f"{output_stem}.mp3"

    combined = AudioSegment.silent(duration=0)
    pause = AudioSegment.silent(duration=int(options.pause_ms))
    chunk_paths = [Path(p) for p in chunk_wavs]
    for i, wav_path in enumerate(chunk_paths, start=1):
        combined += AudioSegment.from_wav(wav_path)
        if i != len(chunk_paths):
            combined += pause

    combined.export(final_wav, format="wav")
    final_audio = final_wav
    if output_format == "mp3":
        combined.export(final_mp3, format="mp3", bitrate="192k")
        final_audio = final_mp3
        if not options.keep_final_wav_when_mp3:
            final_wav.unlink(missing_ok=True)

    if not options.keep_chunk_wavs:
        for wav_path in chunk_paths:
            wav_path.unlink(missing_ok=True)

    metadata = dict(metadata)
    metadata.update(
        {
            "metadata_version": 1,
            "output_audio": str(final_audio),
            "output_wav": str(final_wav) if final_wav.exists() else None,
            "output_mp3": str(final_mp3) if final_mp3.exists() else None,
            "output_format": output_format,
            "pause_ms": int(options.pause_ms),
            "final_duration_ms": len(combined),
        }
    )

    if options.save_json:
        json_path = output_dir / f"{output_stem}.json"
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        jsonl_path = options.jsonl_path or (output_dir / "generation_log.jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        metadata["metadata_json"] = str(json_path)
        metadata["metadata_jsonl"] = str(jsonl_path)

    if output_format == "mp3" and options.write_mp3_tag and final_mp3.exists():
        _write_mp3_comment_tag(final_mp3, metadata)

    return final_audio, metadata
