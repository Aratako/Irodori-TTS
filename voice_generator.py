from __future__ import annotations

from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)


# =========================================================
# 基本設定
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_AUDIO = BASE_DIR / "reference" / "character_B.wav"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_REPO = "Aratako/Irodori-TTS-500M-v3"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

MODEL_DEVICE = "cuda"
MODEL_PRECISION = "bf16"

CODEC_DEVICE = "cuda"
CODEC_PRECISION = "bf16"


def load_runtime() -> InferenceRuntime:
    """Irodori-TTSのモデルとCodecを1回だけ読み込む。"""

    if not REFERENCE_AUDIO.is_file():
        raise FileNotFoundError(
            f"参照音声が見つかりません。\n"
            f"確認する場所: {REFERENCE_AUDIO}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("モデルのチェックポイントを確認しています...")

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model.safetensors",
    )

    print("Irodori-TTSを読み込んでいます...")
    print("初回は少し時間がかかります。")

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=MODEL_DEVICE,
            model_precision=MODEL_PRECISION,
            codec_repo=CODEC_REPO,
            codec_device=CODEC_DEVICE,
            codec_precision=CODEC_PRECISION,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )

    print("モデルの読み込みが完了しました。")
    return runtime


def create_output_path() -> Path:
    """重複しない日時入りの出力ファイル名を作る。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"voice_{timestamp}.wav"


def generate_voice(
    runtime: InferenceRuntime,
    text: str,
) -> Path:
    """入力された文章から音声を生成して保存する。"""

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("文章が入力されていません。")

    result = runtime.synthesize(
        SamplingRequest(
            text=cleaned_text,
            ref_wav=str(REFERENCE_AUDIO),

            # 現在の暫定設定
            num_steps=16,
            t_schedule_mode="sway",
            sway_coeff=-1.0,

            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_speaker=7.0,

            duration_scale=1.0,
            seed=1234,

            num_candidates=1,
            decode_mode="sequential",

            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            max_ref_seconds=30.0,
        ),
        log_fn=None,
    )

    output_path = create_output_path()

    saved_path = save_wav(
        output_path,
        result.audio,
        result.sample_rate,
    )

    print()
    print(f"使用Seed: {result.used_seed}")
    print(f"生成処理時間: {result.total_to_decode:.3f}秒")
    print(f"保存先: {Path(saved_path).resolve()}")

    return Path(saved_path)


def main() -> None:
    try:
        runtime = load_runtime()
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
            generate_voice(runtime, text)
        except KeyboardInterrupt:
            print()
            print("生成を中断しました。")
        except Exception as error:
            print()
            print("音声生成に失敗しました。")
            print(f"エラー: {error}")


if __name__ == "__main__":
    main()