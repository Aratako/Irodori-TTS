from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_config import (
    DEFAULT_TRANSCRIPTION_MODEL,
    LLMConfig,
    TRANSCRIPTION_MODEL_ENV_NAME,
)
from openai_transcription_engine import OpenAITranscriptionEngine


class FakeTranscriptions:
    def __init__(self, text: str = "  こんにちは  ") -> None:
        self.text = text
        self.calls = []

    def create(self, *, model, file, language):
        self.calls.append(
            {
                "model": model,
                "file_name": file.name,
                "file_content": file.read(),
                "language": language,
            }
        )
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text: str = "  こんにちは  ") -> None:
        self.audio = SimpleNamespace(
            transcriptions=FakeTranscriptions(text),
        )


class FailingTranscriptions:
    def __init__(self, message: str) -> None:
        self.message = message

    def create(self, *, model, file, language):
        raise RuntimeError(self.message)


class FailingClient:
    def __init__(self, message: str) -> None:
        self.audio = SimpleNamespace(
            transcriptions=FailingTranscriptions(message),
        )


class OpenAITranscriptionEngineTest(unittest.TestCase):
    def test_transcribe_returns_stripped_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            client = FakeClient("  こんにちは、元気ですか  ")
            engine = OpenAITranscriptionEngine(
                self._config(),
                client=client,
            )

            result = engine.transcribe(audio_path)

            self.assertEqual(result, "こんにちは、元気ですか")

    def test_transcribe_passes_model_file_and_japanese_language(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            client = FakeClient()
            engine = OpenAITranscriptionEngine(
                self._config(transcription_model="custom-transcribe"),
                client=client,
            )

            engine.transcribe(str(audio_path))

            calls = client.audio.transcriptions.calls
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["model"], "custom-transcribe")
            self.assertEqual(calls[0]["file_name"], str(audio_path))
            self.assertEqual(calls[0]["file_content"], b"dummy audio")
            self.assertEqual(calls[0]["language"], "ja")

    def test_from_environment_reads_transcription_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                TRANSCRIPTION_MODEL_ENV_NAME: "env-transcribe",
            },
            clear=True,
        ):
            config = LLMConfig.from_environment()

        self.assertEqual(config.transcription_model, "env-transcribe")

    def test_from_environment_uses_default_transcription_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            config = LLMConfig.from_environment()

        self.assertEqual(config.transcription_model, DEFAULT_TRANSCRIPTION_MODEL)

    def test_rejects_missing_audio_value(self) -> None:
        engine = OpenAITranscriptionEngine(self._config(), client=FakeClient())

        with self.assertRaisesRegex(ValueError, "録音してから文字起こししてください"):
            engine.transcribe(None)

        with self.assertRaisesRegex(ValueError, "録音してから文字起こししてください"):
            engine.transcribe("   ")

    def test_rejects_missing_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OpenAITranscriptionEngine(self._config(), client=FakeClient())

            with self.assertRaisesRegex(
                FileNotFoundError,
                "録音ファイルが見つかりません",
            ):
                engine.transcribe(Path(temp_dir) / "missing.wav")

    def test_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OpenAITranscriptionEngine(self._config(), client=FakeClient())

            with self.assertRaisesRegex(
                FileNotFoundError,
                "録音ファイルが見つかりません",
            ):
                engine.transcribe(temp_dir)

    def test_rejects_empty_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "empty.wav"
            audio_path.write_bytes(b"")
            engine = OpenAITranscriptionEngine(self._config(), client=FakeClient())

            with self.assertRaisesRegex(ValueError, "録音音声が空です"):
                engine.transcribe(audio_path)

    def test_rejects_empty_transcription_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            engine = OpenAITranscriptionEngine(
                self._config(),
                client=FakeClient("   "),
            )

            with self.assertRaisesRegex(RuntimeError, "文字起こし結果が空でした"):
                engine.transcribe(audio_path)

    def test_api_error_message_does_not_include_api_key_or_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._write_audio(Path(temp_dir) / "recording.wav")
            api_key = "secret-test-key"
            engine = OpenAITranscriptionEngine(
                self._config(api_key=api_key),
                client=FailingClient(f"leaked {api_key} {audio_path} response-body"),
            )

            with self.assertRaises(RuntimeError) as context:
                engine.transcribe(audio_path)

            message = str(context.exception)
            self.assertNotIn(api_key, message)
            self.assertNotIn(str(audio_path), message)
            self.assertNotIn("response-body", message)

    def _config(
        self,
        *,
        api_key: str = "test-key",
        transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
    ) -> LLMConfig:
        return LLMConfig(
            api_key=api_key,
            transcription_model=transcription_model,
        )

    def _write_audio(self, path: Path) -> Path:
        path.write_bytes(b"dummy audio")
        return path


if __name__ == "__main__":
    unittest.main()
