from __future__ import annotations

from pathlib import Path

from conversation_engine import CharacterProfile
from llm_config import LLMConfig
from openai_conversation_engine import OpenAIConversationEngine
from voice_engine import VoiceEngine


BASE_DIR = Path(__file__).resolve().parent

REFERENCE_AUDIO = BASE_DIR / "reference" / "character_B.wav"
OUTPUT_DIR = BASE_DIR / "outputs"


def create_character_profile() -> CharacterProfile:
    """現在使用するキャラクター設定を作成する。"""

    return CharacterProfile(
        name="かなめまどか",
        first_person="私",
        personality="明るく親しみやすく、相手を優しく励ます",
        speaking_style="柔らかく自然な口調で、短く話す",
    )


def main() -> None:
    profile = create_character_profile()

    try:
        llm_config = LLMConfig.from_environment()

        conversation_engine = OpenAIConversationEngine(
            profile=profile,
            config=llm_config,
        )

    except Exception as error:
        print()
        print("OpenAI APIの設定に失敗しました。")
        print(f"エラー: {error}")
        return

    voice_engine = VoiceEngine(
        reference_audio=REFERENCE_AUDIO,
        output_dir=OUTPUT_DIR,
    )

    try:
        print("音声生成モデルを読み込んでいます...")
        print("初回は少し時間がかかります。")

        voice_engine.load()

        print("モデルの読み込みが完了しました。")

    except Exception as error:
        print()
        print("音声生成モデルの読み込みに失敗しました。")
        print(f"エラー: {error}")
        return

    print()
    print("========================================")
    print(" OpenAI キャラクター会話プログラム")
    print("========================================")
    print(f"会話相手: {profile.name}")
    print(f"使用モデル: {llm_config.model}")
    print("終了するときは exit と入力します。")

    while True:
        print()
        user_text = input("あなた > ").strip()

        if user_text.lower() in {"exit", "quit", "終了"}:
            print("会話を終了します。")
            break

        if not user_text:
            print("文章を入力してください。")
            continue

        try:
            print("返答を生成しています...")

            reply = conversation_engine.generate_reply(user_text)

            print()
            print(f"{profile.name} > {reply.text}")
            print(
                "使用トークン: "
                f"入力 {reply.input_tokens} / "
                f"出力 {reply.output_tokens} / "
                f"合計 {reply.total_tokens}"
            )

            print("返答音声を生成しています...")

            voice_result = voice_engine.generate(reply.text)

            print()
            print(f"使用Seed: {voice_result.used_seed}")
            print(
                f"音声生成時間: "
                f"{voice_result.generation_seconds:.3f}秒"
            )
            print(f"保存先: {voice_result.output_path}")

        except KeyboardInterrupt:
            print()
            print("処理を中断しました。")

        except Exception as error:
            print()
            print("会話処理または音声生成に失敗しました。")
            print(f"エラー: {error}")


if __name__ == "__main__":
    main()