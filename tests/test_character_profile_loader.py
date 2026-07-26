from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from character_profile_loader import load_character_config


class CharacterProfileLoaderTest(unittest.TestCase):
    def test_loads_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference_audio = base_dir / "reference" / "character.wav"
            reference_audio.parent.mkdir()
            reference_audio.write_bytes(b"dummy wav")
            config_path = self._write_config(base_dir)

            config = load_character_config(config_path)

            self.assertEqual(config.profile.name, "テストキャラクター")
            self.assertEqual(config.profile.first_person, "私")
            self.assertEqual(config.voice.reference_audio, reference_audio.resolve())

    def test_rejects_missing_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(Path(temp_dir), schema_version=None)

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_character_config(config_path)

    def test_rejects_unsupported_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(Path(temp_dir), schema_version=2)

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_character_config(config_path)

    def test_rejects_missing_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                reference_audio=None,
            )

            with self.assertRaisesRegex(ValueError, "voice"):
                load_character_config(config_path)

    def test_rejects_empty_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                reference_audio="   ",
            )

            with self.assertRaisesRegex(ValueError, "voice.reference_audio"):
                load_character_config(config_path)

    def test_rejects_absolute_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute_audio = Path(temp_dir) / "character.wav"
            config_path = self._write_config(
                Path(temp_dir),
                reference_audio=str(absolute_audio),
            )

            with self.assertRaisesRegex(ValueError, "voice.reference_audio"):
                load_character_config(config_path)

    def test_rejects_reference_audio_outside_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "config"
            base_dir.mkdir()
            outside_audio = Path(temp_dir) / "outside.wav"
            outside_audio.write_bytes(b"dummy wav")
            config_path = self._write_config(
                base_dir,
                reference_audio="../outside.wav",
            )

            with self.assertRaisesRegex(ValueError, "voice.reference_audio"):
                load_character_config(config_path)

    def test_rejects_missing_reference_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(Path(temp_dir))

            with self.assertRaisesRegex(FileNotFoundError, "voice.reference_audio"):
                load_character_config(config_path)

    def _write_config(
        self,
        base_dir: Path,
        *,
        schema_version: int | None = 1,
        reference_audio: str | None = "reference/character.wav",
    ) -> Path:
        data: dict[str, object] = {
            "name": "テストキャラクター",
            "first_person": "私",
            "personality": "明るく親しみやすい",
            "speaking_style": "柔らかく自然に話す",
        }

        if schema_version is not None:
            data["schema_version"] = schema_version

        if reference_audio is not None:
            data["voice"] = {
                "reference_audio": reference_audio,
            }

        config_path = base_dir / "character_profile.json"
        config_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
