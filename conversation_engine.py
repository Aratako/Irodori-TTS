from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterProfile:
    """会話するキャラクターの基本設定。"""

    name: str
    first_person: str
    personality: str
    speaking_style: str


@dataclass(frozen=True)
class ConversationTurn:
    """ユーザーとキャラクターの1回分の会話。"""

    user_text: str
    character_text: str


class ConversationEngine:
    """ユーザーの入力からキャラクターの返答文を作る。"""

    def __init__(self, profile: CharacterProfile) -> None:
        self.profile = profile
        self._history: list[ConversationTurn] = []

    @property
    def history(self) -> tuple[ConversationTurn, ...]:
        """これまでの会話履歴を変更できない形で返す。"""

        return tuple(self._history)

    def generate_reply(self, user_text: str) -> str:
        """ユーザーの文章に対する仮の返答文を生成する。"""

        cleaned_text = user_text.strip()

        if not cleaned_text:
            raise ValueError("文章が入力されていません。")

        reply = self._create_temporary_reply(cleaned_text)

        self._history.append(
            ConversationTurn(
                user_text=cleaned_text,
                character_text=reply,
            )
        )

        return reply

    def _create_temporary_reply(self, user_text: str) -> str:
        """LLM接続前に使用する簡単な仮返答処理。"""

        if any(
            word in user_text
            for word in ("こんにちは", "おはよう", "こんばんは")
        ):
            return (
                f"こんにちは。{self.profile.name}だよ。"
                "今日も話せてうれしいな。"
            )

        if "ありがとう" in user_text:
            return (
                "どういたしまして。"
                f"{self.profile.first_person}もうれしいよ。"
            )

        if "疲れた" in user_text:
            return (
                "お疲れさま。無理をしすぎず、"
                "少し休んでもいいと思うよ。"
            )

        if user_text.endswith(("？", "?")):
            return (
                f"「{user_text}」について聞きたいんだね。"
                "今は仮の会話処理だけど、"
                "これからもっと自然に答えられるようにするよ。"
            )

        return (
            f"「{user_text}」っていう話なんだね。"
            "もう少し詳しく聞かせてほしいな。"
        )