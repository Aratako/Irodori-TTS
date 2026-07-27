from __future__ import annotations

import os
from dataclasses import dataclass, field


API_KEY_ENV_NAME = "OPENAI_API_KEY"
MODEL_ENV_NAME = "OPENAI_MODEL"
TRANSCRIPTION_MODEL_ENV_NAME = "OPENAI_TRANSCRIPTION_MODEL"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_MAX_HISTORY_TURNS = 8
DEFAULT_MAX_OUTPUT_TOKENS = 200


@dataclass(frozen=True)
class LLMConfig:
    """LLM接続に使用する設定。"""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """環境変数からLLM設定を読み込む。"""

        api_key = os.getenv(API_KEY_ENV_NAME, "").strip()
        model = os.getenv(MODEL_ENV_NAME, DEFAULT_MODEL).strip()
        transcription_model = os.getenv(
            TRANSCRIPTION_MODEL_ENV_NAME,
            DEFAULT_TRANSCRIPTION_MODEL,
        ).strip()

        if not api_key:
            raise RuntimeError(
                f"環境変数 {API_KEY_ENV_NAME} が設定されていません。\n"
                "APIキーをソースコードへ直接書かず、"
                "環境変数へ設定してください。"
            )

        if not model:
            raise RuntimeError(
                f"環境変数 {MODEL_ENV_NAME} の値が空です。"
            )

        if not transcription_model:
            raise RuntimeError(
                f"環境変数 {TRANSCRIPTION_MODEL_ENV_NAME} の値が空です。"
            )

        return cls(
            api_key=api_key,
            model=model,
            transcription_model=transcription_model,
        )

    def validate(self) -> None:
        """設定値が正しい範囲か確認する。"""

        if self.max_history_turns <= 0:
            raise ValueError(
                "max_history_turnsは1以上にしてください。"
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokensは1以上にしてください。"
            )

        if not self.transcription_model.strip():
            raise ValueError(
                "transcription_modelは空にできません。"
            )
