from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gradio as gr

from gradio_openai_character_chat import (
    AppResources,
    _apply_session_settings,
    _clear_conversation,
    _transcribe_microphone_audio,
    _transcribe_microphone_audio_with_status,
    _validate_session_settings,
    build_ui,
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

            (
                settings,
                chat_messages,
                history_state,
                audio_update,
                microphone_update,
                status,
            ) = _apply_session_settings(
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
            self.assertIsNone(microphone_update["value"])
            self.assertIn("設定を反映", status)

    def test_build_ui_enables_autoplay_only_for_generated_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "reference.wav")
            demo = build_ui(self._resources(audio_path))
            audio_components = [
                component
                for component in demo.blocks.values()
                if isinstance(component, gr.Audio)
            ]

            self.assertEqual(len(audio_components), 3)

            reference_audio = next(
                component
                for component in audio_components
                if component.sources == ["upload"]
            )
            microphone_audio = next(
                component
                for component in audio_components
                if component.sources == ["microphone"]
            )
            generated_audio = next(
                component for component in audio_components if not component.interactive
            )

            self.assertFalse(reference_audio.autoplay)
            self.assertFalse(microphone_audio.autoplay)
            self.assertTrue(generated_audio.autoplay)

    def test_transcription_result_is_written_to_user_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            resources = self._resources(audio_path)

            user_input, status = _transcribe_microphone_audio(
                str(audio_path),
                "既存の入力",
                resources,
            )

            self.assertEqual(user_input, "文字起こし結果")
            self.assertIn("確認・修正", status)
            self.assertEqual(resources.transcription_engine.calls, [str(audio_path)])

    def test_transcription_failure_keeps_existing_user_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            resources = self._resources(audio_path)
            resources.transcription_engine = FakeTranscriptionEngine(
                error=RuntimeError("安全なエラー")
            )

            user_input, status = _transcribe_microphone_audio(
                str(audio_path),
                "消さない入力",
                resources,
            )

            self.assertEqual(user_input, "消さない入力")
            self.assertIn("文字起こしに失敗しました", status)

    def test_transcription_status_generator_first_yield_keeps_existing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            resources = self._resources(audio_path)

            generator = _transcribe_microphone_audio_with_status(
                str(audio_path),
                "既存の入力",
                resources,
            )

            self.assertEqual(next(generator), ("既存の入力", "文字起こし中..."))
            self.assertEqual(resources.transcription_engine.calls, [])

    def test_transcription_status_generator_second_yield_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            resources = self._resources(audio_path)
            generator = _transcribe_microphone_audio_with_status(
                str(audio_path),
                "既存の入力",
                resources,
            )

            next(generator)
            user_input, status = next(generator)

            self.assertEqual(user_input, "文字起こし結果")
            self.assertEqual(
                status,
                "文字起こししました。内容を確認・修正してから送信してください。",
            )

    def test_transcription_status_generator_second_yield_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            resources = self._resources(audio_path)
            resources.transcription_engine = FakeTranscriptionEngine(
                error=RuntimeError("安全なエラー")
            )
            generator = _transcribe_microphone_audio_with_status(
                str(audio_path),
                "消さない入力",
                resources,
            )

            next(generator)
            user_input, status = next(generator)

            self.assertEqual(user_input, "消さない入力")
            self.assertIn("文字起こしに失敗しました: 安全なエラー", status)

    def test_clear_conversation_clears_microphone_audio(self) -> None:
        chat_messages, history_state, audio_update, microphone_update, status = (
            _clear_conversation()
        )

        self.assertEqual(chat_messages, [])
        self.assertEqual(history_state, [])
        self.assertIsNone(audio_update["value"])
        self.assertIsNone(microphone_update["value"])
        self.assertIn("会話履歴をクリア", status)

    def test_transcribe_button_does_not_submit_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "reference.wav")
            demo = build_ui(self._resources(audio_path))
            components_by_label = {
                component.label: component
                for component in demo.blocks.values()
                if hasattr(component, "label")
            }
            user_input = components_by_label["あなたの文章"]
            status = components_by_label["状態"]
            chatbot = components_by_label["キャラクターとの会話"]
            generated_audio = components_by_label["生成音声"]

            transcribe_dependency = next(
                dependency
                for dependency in demo.config["dependencies"]
                if dependency["api_name"] == "transcribe_microphone_audio"
            )

            self.assertEqual(
                transcribe_dependency["outputs"],
                [user_input._id, status._id],
            )
            self.assertNotIn(chatbot._id, transcribe_dependency["outputs"])
            self.assertNotIn(generated_audio._id, transcribe_dependency["outputs"])

    def _resources(self, audio_path: Path) -> AppResources:
        return AppResources(
            llm_config=object(),
            voice_engine=object(),
            transcription_engine=FakeTranscriptionEngine(),
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


class FakeTranscriptionEngine:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def transcribe(self, audio_path: str | Path | None) -> str:
        self.calls.append(audio_path)

        if self.error:
            raise self.error

        return "文字起こし結果"


if __name__ == "__main__":
    unittest.main()
