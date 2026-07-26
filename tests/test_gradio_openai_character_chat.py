from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gradio_openai_character_chat import (
    AppResources,
    _apply_session_settings,
    _validate_session_settings,
)


class GradioCharacterSettingsTest(unittest.TestCase):
    def test_validate_session_settings_accepts_valid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "reference.wav")

            settings = _validate_session_settings(
                name=" テストキャラクター ",
                first_person=" 私 ",
                personality=" 明るく親しみやすい ",
                speaking_style=" 柔らかく自然に話す ",
                reference_audio=str(audio_path),
            )

            self.assertEqual(
                settings,
                {
                    "name": "テストキャラクター",
                    "first_person": "私",
                    "personality": "明るく親しみやすい",
                    "speaking_style": "柔らかく自然に話す",
                    "reference_audio_path": str(audio_path.resolve()),
                },
            )

    def test_validate_session_settings_rejects_empty_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "reference.wav")

            with self.assertRaisesRegex(ValueError, "キャラクター名"):
                _validate_session_settings(
                    name=" ",
                    first_person="私",
                    personality="明るい",
                    speaking_style="自然に話す",
                    reference_audio=str(audio_path),
                )

    def test_validate_session_settings_rejects_missing_audio_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "参照音声"):
            _validate_session_settings(
                name="テスト",
                first_person="私",
                personality="明るい",
                speaking_style="自然に話す",
                reference_audio=None,
            )

    def test_validate_session_settings_rejects_missing_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "参照音声"):
                _validate_session_settings(
                    name="テスト",
                    first_person="私",
                    personality="明るい",
                    speaking_style="自然に話す",
                    reference_audio=str(Path(temp_dir) / "missing.wav"),
                )

    def test_validate_session_settings_rejects_empty_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "empty.wav"
            audio_path.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "空の参照音声"):
                _validate_session_settings(
                    name="テスト",
                    first_person="私",
                    personality="明るい",
                    speaking_style="自然に話す",
                    reference_audio=str(audio_path),
                )

    def test_validate_session_settings_rejects_unsupported_audio_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "reference.txt"
            audio_path.write_bytes(b"dummy")

            with self.assertRaisesRegex(ValueError, "対応拡張子"):
                _validate_session_settings(
                    name="テスト",
                    first_person="私",
                    personality="明るい",
                    speaking_style="自然に話す",
                    reference_audio=str(audio_path),
                )

    def test_apply_session_settings_clears_history_and_audio_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "reference.wav")
            resources = self._resources(audio_path)
            history = [
                {
                    "user_text": "こんにちは",
                    "character_text": "こんにちは。",
                }
            ]

            settings, chat_messages, history_state, audio_update, status = _apply_session_settings(
                name="テスト",
                first_person="私",
                personality="明るい",
                speaking_style="自然に話す",
                reference_audio=str(audio_path),
                settings=resources.initial_settings,
                history=history,
                resources=resources,
            )

            self.assertEqual(settings["reference_audio_path"], str(audio_path.resolve()))
            self.assertEqual(chat_messages, [])
            self.assertEqual(history_state, [])
            self.assertIsNone(audio_update["value"])
            self.assertIn("設定を反映", status)

    def _resources(self, audio_path: Path) -> AppResources:
        return AppResources(
            llm_config=object(),
            voice_engine=object(),
            initial_settings={
                "name": "初期",
                "first_person": "私",
                "personality": "明るい",
                "speaking_style": "自然に話す",
                "reference_audio_path": str(audio_path.resolve()),
            },
        )

    def _write_audio(self, path: Path) -> Path:
        path.write_bytes(b"dummy wav")
        return path


if __name__ == "__main__":
    unittest.main()
