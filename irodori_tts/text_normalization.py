from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

SIMPLE_REPLACE_MAP: dict[str, str] = {
    "\t": "",
    "[n]": "",
    r"\[n\]": "",
    "　": "",
    "？": "?",
    "！": "!",
    "♥": "♡",
    "●": "○",
    "◯": "○",
    "〇": "○",
}

REGEX_REPLACE_MAP = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]"): "",
    re.compile(r"[\uff5e\u301C]"): "ー",
    re.compile(r"…{3,}"): "……",
}

ENGLISH_SPAN_RE = re.compile(r"(?<![A-Za-z])([A-Za-z](?:[A-Za-z._+-]*[A-Za-z+#])?)")

ENGLISH_WORD_READINGS_ENV_VAR = "IRODORI_TTS_ENGLISH_WORD_READINGS_PATH"
DEFAULT_ENGLISH_WORD_READINGS_PATH = Path(__file__).resolve().parent / "english_word_readings.jsonl"
LOCAL_ENGLISH_WORD_READINGS_FILENAME = "english_word_readings.jsonl"


def get_english_word_readings_paths() -> list[Path]:
    paths = [
        DEFAULT_ENGLISH_WORD_READINGS_PATH,
        Path.cwd() / LOCAL_ENGLISH_WORD_READINGS_FILENAME,
    ]

    env_path = os.environ.get(ENGLISH_WORD_READINGS_ENV_VAR)
    if env_path:
        paths.append(Path(env_path).expanduser())

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(resolved_path)

    return unique_paths


def load_english_word_readings_file(path: Path) -> dict[str, str]:
    readings: dict[str, str] = {}

    if not path.exists():
        return readings

    try:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "failed to parse %s line %d: %s",
                        path,
                        line_number,
                        exc,
                    )
                    continue

                if not isinstance(item, dict):
                    logger.warning(
                        "skipping %s line %d: expected JSON object",
                        path,
                        line_number,
                    )
                    continue

                word = item.get("word")
                reading = item.get("reading")
                if not isinstance(word, str) or not isinstance(reading, str):
                    logger.warning(
                        "skipping %s line %d: expected string word and reading",
                        path,
                        line_number,
                    )
                    continue

                readings[word.lower()] = reading
    except OSError as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return {}

    return readings


def load_english_word_readings() -> dict[str, str]:
    readings: dict[str, str] = {}

    for path in get_english_word_readings_paths():
        readings.update(load_english_word_readings_file(path))

    return readings


def strip_outer_brackets(text: str) -> str:
    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}

    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char = text[-1]

        if start_char in pairs and pairs[start_char] == end_char:
            depth = 0
            is_enclosing_all = True

            for i, char in enumerate(text):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1

                if depth == 0 and i < len(text) - 1:
                    is_enclosing_all = False
                    break

            if is_enclosing_all and depth == 0:
                text = text[1:-1]
                continue

        break

    return text


def english_to_katakana(text: str, readings: dict[str, str] | None = None) -> str:
    key = text.lower()
    english_word_readings = readings if readings is not None else load_english_word_readings()
    if key in english_word_readings:
        return english_word_readings[key]
    return text


def normalize_english_spans(text: str) -> str:
    english_word_readings = load_english_word_readings()
    return ENGLISH_SPAN_RE.sub(
        lambda match: english_to_katakana(match.group(1), english_word_readings),
        text,
    )


def normalize_text(text: str, *, convert_english: bool = True) -> str:
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    for pattern, replacement in REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    text = strip_outer_brackets(text)

    text = unicodedata.normalize("NFKC", text)

    if convert_english:
        text = normalize_english_spans(text)

    text = text.replace("...", "…")
    text = text.replace("..", "…")

    return text
