from __future__ import annotations

from pathlib import Path

from character_profile_loader import load_character_config
from conversation_engine import ConversationEngine
from voice_engine import VoiceEngine


BASE_DIR = Path(__file__).resolve().parent

CHARACTER_PROFILE_PATH = BASE_DIR / "character_profile.json"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    try:
        character_config = load_character_config(CHARACTER_PROFILE_PATH)
        profile = character_config.profile

    except Exception as error:
        print()
        print("キャラクター設定の読み込みに失敗しました。")
        print(f"エラー: {error}")
        return

    conversation_engine = ConversationEngine(profile)

    voice_engine = VoiceEngine(
        reference_audio=character_config.voice.reference_audio,
        output_dir=OUTPUT_DIR,
    )

    try:
        print("音声生成モデルを読み込んでいます...")
        print("初回は少し時間がかかります。")

        voice_engine.load()

        print("モデルの読み込みが完了しました。")

    except Exception as error:
        print()
        print("モデルの読み込みに失敗しました。")
        print(f"エラー: {error}")
        return

    print()
    print("========================================")
    print(" キャラクター会話プログラム")
    print("========================================")
    print(f"会話相手: {profile.name}")
    print("終了するときは exit と入力してください。")

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
            reply = conversation_engine.generate_reply(user_text)

            print(f"{profile.name} > {reply}")
            print("返信音声を生成しています...")

            result = voice_engine.generate(reply)

            print()
            print(f"使用Seed: {result.used_seed}")
            print(f"生成処理時間: {result.generation_seconds:.3f}秒")
            print(f"保存先: {result.output_path}")

        except KeyboardInterrupt:
            print()
            print("処理を中断しました。")

        except Exception as error:
            print()
            print("会話処理または音声生成に失敗しました。")
            print(f"エラー: {error}")


if __name__ == "__main__":
    main()
