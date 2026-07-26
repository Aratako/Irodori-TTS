from __future__ import annotations

import json
from pathlib import Path

from conversation_engine import CharacterProfile


REQUIRED_FIELDS = (
    "name",
    "first_person",
    "personality",
    "speaking_style",
)


def load_character_profile(config_path: Path) -> CharacterProfile:
    """JSONファイルからキャラクター設定を読み込む。"""

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
            f"行: {error.lineno}、列: {error.colno}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "キャラクター設定の最上位はJSONオブジェクトにしてください。"
        )

    invalid_fields = [
        field_name
        for field_name in REQUIRED_FIELDS
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