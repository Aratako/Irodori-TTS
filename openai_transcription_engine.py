from __future__ import annotations

from pathlib import Path
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from llm_config import LLMConfig


class TranscriptionClient(Protocol):
    audio: object


class OpenAITranscriptionEngine:
    """OpenAI APIで録音音声を日本語テキストへ文字起こしする。"""

    def __init__(
        self,
        config: LLMConfig,
        client: TranscriptionClient | None = None,
    ) -> None:
        config.validate()

        self.config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            timeout=30.0,
        )

    def transcribe(self, audio_path: str | Path | None) -> str:
        """録音音声ファイルを文字起こしして、前後の空白を除いた文字列を返す。"""

        path = self._validate_audio_path(audio_path)

        try:
            with path.open("rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self.config.transcription_model,
                    file=audio_file,
                    language="ja",
                )

        except APITimeoutError as error:
            raise RuntimeError(
                "音声文字起こしの応答が時間内に返りませんでした。もう一度お試しください。"
            ) from error

        except APIConnectionError as error:
            raise RuntimeError(
                "音声文字起こしAPIへ接続できませんでした。通信環境を確認してください。"
            ) from error

        except APIStatusError as error:
            request_id = error.request_id or "不明"

            raise RuntimeError(
                "音声文字起こしAPIがエラーを返しました。\n"
                f"ステータスコード: {error.status_code}\n"
                f"リクエストID: {request_id}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                "音声文字起こしに失敗しました。録音し直してもう一度お試しください。"
            ) from error

        text = str(getattr(response, "text", "") or "").strip()

        if not text:
            raise RuntimeError(
                "文字起こし結果が空でした。もう一度録音してお試しください。"
            )

        return text

    def _validate_audio_path(self, audio_path: str | Path | None) -> Path:
        if audio_path is None:
            raise ValueError("録音してから文字起こししてください。")

        audio_path_text = str(audio_path).strip()

        if not audio_path_text:
            raise ValueError("録音してから文字起こししてください。")

        path = Path(audio_path_text)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                "録音ファイルが見つかりません。録音し直してください。"
            )

        if path.stat().st_size == 0:
            raise ValueError("録音音声が空です。もう一度録音してください。")

        return path
