from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conversation_engine import CharacterProfile


SUPPORTED_SCHEMA_VERSION = 1

REQUIRED_PROFILE_FIELDS = (
    "name",
    "first_person",
    "personality",
    "speaking_style",
)


@dataclass(frozen=True)
class VoiceConfig:
    """キャラクター音声の設定。"""

    reference_audio: Path


@dataclass(frozen=True)
class CharacterConfig:
    """会話用プロフィールと音声設定をまとめた設定。"""

    profile: CharacterProfile
    voice: VoiceConfig


def load_character_config(config_path: Path) -> CharacterConfig:
    """JSONファイルからキャラクター設定を読み込む。"""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"キャラクター設定ファイルが見つかりません: {config_path}\n"
            "character_profile.example.jsonをコピーして、"
            "character_profile.jsonを作成してください。"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "キャラクター設定ファイルのJSON形式が正しくありません。\n"
            f"行: {error.lineno}, 列: {error.colno}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "キャラクター設定ファイルの最上位はJSONオブジェクトにしてください。"
        )

    _validate_schema_version(data)
    profile = _load_profile(data)
    voice = _load_voice_config(data, config_path)

    return CharacterConfig(
        profile=profile,
        voice=voice,
    )


def load_character_profile(config_path: Path) -> CharacterProfile:
    """互換用: JSONファイルから会話用プロフィールだけを読み込む。"""

    return load_character_config(config_path).profile


def _validate_schema_version(data: dict[str, Any]) -> None:
    if "schema_version" not in data:
        raise ValueError(
            "character_profile.jsonに設定項目 schema_version がありません。"
        )

    schema_version = data["schema_version"]

    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError(
            "character_profile.jsonの schema_version は整数の1にしてください。"
        )

    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            "character_profile.jsonの schema_version は1にしてください。"
        )


def _load_profile(data: dict[str, Any]) -> CharacterProfile:
    invalid_fields = [
        field_name
        for field_name in REQUIRED_PROFILE_FIELDS
        if not isinstance(data.get(field_name), str)
        or not data[field_name].strip()
    ]

    if invalid_fields:
        fields = ", ".join(invalid_fields)

        raise ValueError(
            "キャラクター設定に未入力または不正な項目があります。\n"
            f"対象項目: {fields}"
        )

    return CharacterProfile(
        name=data["name"].strip(),
        first_person=data["first_person"].strip(),
        personality=data["personality"].strip(),
        speaking_style=data["speaking_style"].strip(),
    )


def _load_voice_config(data: dict[str, Any], config_path: Path) -> VoiceConfig:
    voice_data = data.get("voice")

    if not isinstance(voice_data, dict):
        raise ValueError(
            "character_profile.jsonの voice はオブジェクトにしてください。"
        )

    reference_audio = voice_data.get("reference_audio")

    if not isinstance(reference_audio, str) or not reference_audio.strip():
        raise ValueError(
            "character_profile.jsonの voice.reference_audio は"
            "空でない文字列にしてください。"
        )

    reference_audio_path = Path(reference_audio.strip())

    if reference_audio_path.is_absolute() or reference_audio_path.drive:
        raise ValueError(
            "character_profile.jsonの voice.reference_audio は"
            "相対パスで指定してください。"
        )

    config_dir = config_path.resolve().parent
    resolved_reference_audio = (config_dir / reference_audio_path).resolve()

    if not resolved_reference_audio.is_relative_to(config_dir):
        raise ValueError(
            "character_profile.jsonの voice.reference_audio は"
            "設定ファイルのフォルダ内を指す相対パスにしてください。"
        )

    if not resolved_reference_audio.exists():
        raise FileNotFoundError(
            "character_profile.jsonの voice.reference_audio で指定された"
            f"参照音声ファイルが見つかりません: {resolved_reference_audio}"
        )

    if not resolved_reference_audio.is_file():
        raise ValueError(
            "character_profile.jsonの voice.reference_audio は"
            f"通常ファイルを指定してください: {resolved_reference_audio}"
        )

    return VoiceConfig(reference_audio=resolved_reference_audio)
