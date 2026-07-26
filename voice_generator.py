from __future__ import annotations

from pathlib import Path

from voice_engine import VoiceEngine


BASE_DIR = Path(__file__).resolve().parent

REFERENCE_AUDIO = BASE_DIR / "reference" / "character_B.wav"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    engine = VoiceEngine(
        reference_audio=REFERENCE_AUDIO,
        output_dir=OUTPUT_DIR,
    )

    try:
        print("モデルを読み込んでいます...")
        print("初回は少し時間がかかります。")

        engine.load()

        print("モデルの読み込みが完了しました。")

    except Exception as error:
        print()
        print("モデルの読み込みに失敗しました。")
        print(f"エラー: {error}")
        return

    print()
    print("========================================")
    print(" Irodori-TTS 音声生成プログラム")
    print("========================================")
    print("読み上げたい文章を入力してください。")
    print("終了するときは exit と入力します。")

    while True:
        print()
        text = input("セリフ > ").strip()

        if text.lower() in {"exit", "quit", "終了"}:
            print("プログラムを終了します。")
            break

        if not text:
            print("文章を入力してください。")
            continue

        try:
            print("音声を生成しています...")

            result = engine.generate(text)

            print()
            print(f"使用Seed: {result.used_seed}")
            print(f"生成処理時間: {result.generation_seconds:.3f}秒")
            print(f"保存先: {result.output_path}")

        except KeyboardInterrupt:
            print()
            print("生成を中断しました。")

        except Exception as error:
            print()
            print("音声生成に失敗しました。")
            print(f"エラー: {error}")


if __name__ == "__main__":
    main()