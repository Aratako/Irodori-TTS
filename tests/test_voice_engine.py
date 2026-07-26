from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voice_engine import VoiceEngine


class FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def synthesize(self, request, log_fn=None):
        self.requests.append(request)
        return SimpleNamespace(
            audio=b"dummy audio",
            sample_rate=24000,
            used_seed=1234,
            total_to_decode=0.123,
        )


class VoiceEngineGenerateTest(unittest.TestCase):
    def test_generate_uses_specified_reference_audio_without_mutating_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            override_audio = self._write_audio(base_dir / "override.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch("voice_engine.save_wav", return_value=base_dir / "generated.wav"):
                engine.generate("こんにちは", reference_audio=override_audio)

            self.assertEqual(fake_runtime.requests[0].ref_wav, str(override_audio))
            self.assertEqual(engine.reference_audio, default_audio)

    def test_generate_uses_default_reference_audio_when_not_specified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch("voice_engine.save_wav", return_value=base_dir / "generated.wav"):
                engine.generate("こんにちは")

            self.assertEqual(fake_runtime.requests[0].ref_wav, str(default_audio))

    def test_generate_rejects_missing_specified_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            missing_audio = base_dir / "missing.wav"
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()

            with self.assertRaisesRegex(FileNotFoundError, "参照音声"):
                engine.generate("こんにちは", reference_audio=missing_audio)

            self.assertEqual(engine.reference_audio, default_audio)

    def _write_audio(self, path: Path) -> Path:
        path.write_bytes(b"dummy wav")
        return path


if __name__ == "__main__":
    unittest.main()
