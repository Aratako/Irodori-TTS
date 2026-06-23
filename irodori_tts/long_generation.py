from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def split_long_text(text: str, max_chars: int = 80) -> list[str]:
    """Split long Japanese narration text at punctuation/newlines where possible."""
    if max_chars < 20:
        raise ValueError("max_chars must be >= 20")

    text = text.replace("\r\n", "\n").strip()
    parts = re.split(r"(?<=[。．！？!?])\s*|\n+", text)

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


def _require_soundfile():
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile and numpy are required for long generation export. "
            "Install dependencies with: uv sync"
        ) from exc
    return sf, np


def _join_wav_chunks(
    *,
    chunk_paths: list[Path],
    output_wav: Path,
    pause_ms: int,
) -> int:
    """Join WAV chunks and return the final duration in milliseconds."""
    if not chunk_paths:
        raise ValueError("chunk_wavs must contain at least one WAV file")

    sf, np = _require_soundfile()

    first_info = sf.info(str(chunk_paths[0]))
    sample_rate = int(first_info.samplerate)
    channels = int(first_info.channels)
    subtype = first_info.subtype

    if sample_rate <= 0 or channels <= 0:
        raise ValueError(f"Invalid WAV format: {chunk_paths[0]}")

    pause_frames = int(round(sample_rate * pause_ms / 1000))
    total_frames = 0

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        str(output_wav),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype=subtype,
        format="WAV",
    ) as out_file:
        for index, wav_path in enumerate(chunk_paths, start=1):
            info = sf.info(str(wav_path))
            if int(info.samplerate) != sample_rate or int(info.channels) != channels:
                raise ValueError(
                    "All chunk WAV files must use the same sample rate and channel count. "
                    f"Expected {sample_rate} Hz/{channels} ch, got "
                    f"{info.samplerate} Hz/{info.channels} ch: {wav_path}"
                )

            with sf.SoundFile(str(wav_path), mode="r") as in_file:
                for block in in_file.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    out_file.write(block)
                    total_frames += len(block)

            if index != len(chunk_paths) and pause_frames > 0:
                silence = np.zeros((pause_frames, channels), dtype="float32")
                out_file.write(silence)
                total_frames += pause_frames

    return int(round(total_frames * 1000 / sample_rate))


def _resolve_ffmpeg_exe(ffmpeg_exe: str | None) -> str:
    if ffmpeg_exe:
        explicit_path = Path(ffmpeg_exe)
        if explicit_path.exists():
            return str(explicit_path)
        resolved = shutil.which(ffmpeg_exe)
        if resolved:
            return resolved
        raise RuntimeError(f"ffmpeg executable was not found: {ffmpeg_exe}")

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    raise RuntimeError(
        "MP3 export requires ffmpeg. Set --ffmpeg-exe or add ffmpeg.exe to PATH."
    )


def _export_mp3_with_ffmpeg(*, source_wav: Path, output_mp3: Path, ffmpeg_exe: str | None) -> None:
    ffmpeg = _resolve_ffmpeg_exe(ffmpeg_exe)
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        "-vn",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_mp3),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(f"ffmpeg executable was not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg MP3 export failed with exit code {exc.returncode}") from exc


def _id3_syncsafe_int(size: int) -> bytes:
    """ID3v2 tag header用の28bit syncsafe integerを作る。"""
    if size < 0 or size >= (1 << 28):
        raise ValueError("ID3v2 tag size is out of range")

    return bytes(
        [
            (size >> 21) & 0x7F,
            (size >> 14) & 0x7F,
            (size >> 7) & 0x7F,
            size & 0x7F,
        ]
    )


def _read_id3_syncsafe_int(data: bytes) -> int:
    """ID3v2 tag header用の28bit syncsafe integerを読む。"""
    if len(data) != 4:
        raise ValueError("syncsafe integer must be 4 bytes")

    return (
        ((data[0] & 0x7F) << 21)
        | ((data[1] & 0x7F) << 14)
        | ((data[2] & 0x7F) << 7)
        | (data[3] & 0x7F)
    )


def _strip_existing_id3v2_tag(mp3_data: bytes) -> tuple[bytes, int]:
    """
    MP3先頭に既存ID3v2タグがあれば除去する。
    戻り値: (audio_data_without_id3v2, stripped_byte_count)
    """
    if len(mp3_data) < 10 or mp3_data[:3] != b"ID3":
        return mp3_data, 0

    tag_size = _read_id3_syncsafe_int(mp3_data[6:10])
    tag_end = 10 + tag_size

    # 既存タグがID3v2.4 footer付きだった場合の保険。
    # v2.3では通常footerはありません。
    flags = mp3_data[5]
    if flags & 0x10:
        tag_end += 10

    return mp3_data[tag_end:], tag_end


def _strip_id3v1_tag(mp3_data: bytes) -> tuple[bytes, int]:
    """
    末尾のID3v1タグを除去する。
    Explorerや古いタグソフトがID3v1を優先して見てしまう可能性を潰すための保険。
    """
    if len(mp3_data) >= 128 and mp3_data[-128:-125] == b"TAG":
        return mp3_data[:-128], 128
    return mp3_data, 0


def _utf16le_with_bom(text: str) -> bytes:
    """ID3v2.3互換のUTF-16LE BOM付き文字列。"""
    return b"\xff\xfe" + text.encode("utf-16-le")


def _id3v23_frame(frame_id: str, payload: bytes) -> bytes:
    """
    ID3v2.3フレームを作る。
    v2.3のフレームサイズはsyncsafeではなく、通常の32bit big-endian。
    """
    if len(frame_id) != 4:
        raise ValueError("frame_id must be 4 characters")

    return (
        frame_id.encode("ascii")
        + struct.pack(">I", len(payload))
        + b"\x00\x00"
        + payload
    )


def _id3v23_comm_payload(comment_text: str, language: str = "jpn") -> bytes:
    """
    ID3v2.3 COMM frame payload.

    Structure:
      Text encoding: 0x01 = UTF-16
      Language: 3 bytes, e.g. jpn / eng
      Short content description: empty UTF-16 string + terminator
      Actual text: UTF-16 string
    """
    lang = language.encode("ascii", errors="ignore")[:3]
    if len(lang) != 3:
        lang = b"jpn"

    encoding = b"\x01"  # UTF-16 with BOM
    empty_description = b"\xff\xfe\x00\x00"
    text = _utf16le_with_bom(comment_text)

    return encoding + lang + empty_description + text


def _id3v23_txxx_payload(description: str, value: str) -> bytes:
    """
    ID3v2.3 TXXX frame payload.
    タグ確認ツールで見つけやすくするための補助フレーム。
    """
    encoding = b"\x01"  # UTF-16 with BOM
    desc = _utf16le_with_bom(description) + b"\x00\x00"
    val = _utf16le_with_bom(value)
    return encoding + desc + val


def _make_id3v23_tag(comment_text: str, language: str = "jpn") -> bytes:
    frames = b"".join(
        [
            _id3v23_frame("COMM", _id3v23_comm_payload(comment_text, language=language)),
            _id3v23_frame(
                "TXXX",
                _id3v23_txxx_payload("IrodoriTTS:GenerationParameters", comment_text),
            ),
        ]
    )

    # フレーム終端・互換性用のpadding。
    padding = b"\x00" * 2048
    tag_body = frames + padding

    header = (
        b"ID3"
        + b"\x03\x00"  # ID3v2.3.0
        + b"\x00"      # flags
        + _id3_syncsafe_int(len(tag_body))
    )

    return header + tag_body


def _decode_utf16_id3_string(data: bytes) -> str:
    if data.startswith(b"\xff\xfe"):
        return data[2:].decode("utf-16-le", errors="replace")
    if data.startswith(b"\xfe\xff"):
        return data[2:].decode("utf-16-be", errors="replace")
    return data.decode("utf-16-le", errors="replace")


def _split_utf16_terminated(data: bytes) -> tuple[bytes, bytes]:
    """
    UTF-16文字列を最初の 00 00 終端で分割する。
    BOM直後から2バイト境界で探す。
    """
    start = 2 if data.startswith((b"\xff\xfe", b"\xfe\xff")) else 0

    for i in range(start, len(data) - 1, 2):
        if data[i:i + 2] == b"\x00\x00":
            return data[:i], data[i + 2:]

    return data, b""


def _parse_id3v23_frames(mp3_path: Path) -> list[tuple[str, bytes]]:
    data = Path(mp3_path).read_bytes()

    if len(data) < 10 or data[:3] != b"ID3":
        return []

    major = data[3]
    tag_size = _read_id3_syncsafe_int(data[6:10])
    tag_body = data[10:10 + tag_size]

    # This writer emits ID3v2.3. If a user later edits the file with another
    # tool and converts it to another version, avoid misreading it here.
    if major != 3:
        return []

    frames: list[tuple[str, bytes]] = []
    pos = 0

    while pos + 10 <= len(tag_body):
        frame_header = tag_body[pos:pos + 10]

        # paddingに到達
        if frame_header == b"\x00" * 10 or frame_header[:4] == b"\x00" * 4:
            break

        frame_id = frame_header[:4].decode("ascii", errors="replace")
        if not re.fullmatch(r"[A-Z0-9]{4}", frame_id):
            break

        frame_size = struct.unpack(">I", frame_header[4:8])[0]
        payload_start = pos + 10
        payload_end = payload_start + frame_size
        if frame_size < 0 or payload_end > len(tag_body):
            break

        frames.append((frame_id, tag_body[payload_start:payload_end]))
        pos = payload_end

    return frames


def read_mp3_id3v23_comment(mp3_path: Path) -> str | None:
    """Read the first ID3v2.3 COMM text from an MP3 file."""
    frames = _parse_id3v23_frames(Path(mp3_path))

    for frame_id, payload in frames:
        if frame_id != "COMM":
            continue

        if len(payload) < 5:
            continue

        encoding = payload[0]
        rest = payload[4:]  # Skip text encoding and 3-byte language code.

        if encoding == 0x01:
            _description, actual_text = _split_utf16_terminated(rest)
            return _decode_utf16_id3_string(actual_text)

        if encoding == 0x03:
            # UTF-8の場合の保険
            parts = rest.split(b"\x00", 1)
            if len(parts) == 2:
                return parts[1].decode("utf-8", errors="replace")
            return ""

        return None

    return None


def write_mp3_id3v23_comment(mp3_path: Path, comment_text: str) -> None:
    """
    MP3にID3v2.3 COMMコメントを書き込む。
    mutagen等の外部タグライブラリは不要。
    """
    mp3_path = Path(mp3_path)

    original_data = mp3_path.read_bytes()
    audio_data, _stripped_v2 = _strip_existing_id3v2_tag(original_data)
    audio_data, _stripped_v1 = _strip_id3v1_tag(audio_data)

    tag = _make_id3v23_tag(comment_text, language="jpn")

    tmp_path = mp3_path.with_suffix(mp3_path.suffix + ".tmp")
    tmp_path.write_bytes(tag + audio_data)
    os.replace(tmp_path, mp3_path)

    readback = read_mp3_id3v23_comment(mp3_path)
    if readback != comment_text:
        raise RuntimeError("MP3 ID3v2.3 COMM tag verification failed after writing")

def export_long_audio(
    *,
    chunk_wavs: Iterable[Path | str],
    options: LongExportOptions,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Join chunk WAV files, export final WAV/MP3, and write metadata sidecars/tags."""
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
    final_audio = final_mp3 if output_format == "mp3" else final_wav

    chunk_paths = [Path(p) for p in chunk_wavs]
    final_duration_ms = _join_wav_chunks(
        chunk_paths=chunk_paths,
        output_wav=final_wav,
        pause_ms=int(options.pause_ms),
    )

    metadata = dict(metadata)
    if options.save_json:
        json_path = output_dir / f"{output_stem}.json"
        jsonl_path = options.jsonl_path or (output_dir / "generation_log.jsonl")
    else:
        json_path = None
        jsonl_path = None

    metadata.update(
        {
            "metadata_version": 1,
            "output_audio": str(final_audio),
            "output_wav": str(final_wav)
            if output_format == "wav" or options.keep_final_wav_when_mp3
            else None,
            "output_mp3": str(final_mp3) if output_format == "mp3" else None,
            "output_format": output_format,
            "pause_ms": int(options.pause_ms),
            "final_duration_ms": final_duration_ms,
            "mp3_tag_written": output_format == "mp3" and bool(options.write_mp3_tag),
            "mp3_tag_verified": output_format == "mp3" and bool(options.write_mp3_tag),
            "mp3_tag_method": "id3v2.3_comm_txxx"
            if output_format == "mp3" and bool(options.write_mp3_tag)
            else None,
            "metadata_json": str(json_path) if json_path is not None else None,
            "metadata_jsonl": str(jsonl_path) if jsonl_path is not None else None,
        }
    )

    if output_format == "mp3":
        _export_mp3_with_ffmpeg(
            source_wav=final_wav,
            output_mp3=final_mp3,
            ffmpeg_exe=options.ffmpeg_exe,
        )
        if options.write_mp3_tag:
            comment_json = json.dumps(metadata, ensure_ascii=False, indent=2)
            write_mp3_id3v23_comment(final_mp3, comment_json)
        if not options.keep_final_wav_when_mp3:
            final_wav.unlink(missing_ok=True)

    if not options.keep_chunk_wavs:
        for wav_path in chunk_paths:
            wav_path.unlink(missing_ok=True)

    if options.save_json and json_path is not None and jsonl_path is not None:
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    return final_audio, metadata
