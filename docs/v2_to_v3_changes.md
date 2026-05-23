# Irodori-TTS v2 → v3 変更点まとめ

このドキュメントは、フォーク元 ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)) における v2 から v3 への主要な変更点をまとめたものです。

## 概要

v3 は v2 のアーキテクチャを大幅に拡張し、**Duration Predictor（発話長予測器）** の統合、**Speaker Inversion（話者埋め込み学習）** 機能、**Sway Sampling** による高速推論、**音声透かし（SilentCipher）**、**絵文字パレットUI** などが追加されました。v2 チェックポイントとの後方互換性は維持されています。

---

## 1. Duration Predictor（発話長予測器）の追加 ⭐ 最大の変更

### v2 の挙動
- 出力長は固定 **30秒** でトレーニング・推論されていた
- 推論時に `--seconds` で手動指定するか、デフォルトの 30 秒が使われる
- 末尾の無音部分は `--trim-tail` ヒューリスティクスで除去していた

### v3 の挙動
- **可変長トレーニング**: `fixed_target_latent_steps: null` でサンプルが自然な長さのまま学習される
- **統合されたDuration Predictor**: テキストのトークン情報から出力フレーム数を自動予測
- `--seconds` を省略すると、モデルが適切な長さを自動決定する
- `--duration-scale` パラメータで予測長をスケーリング可能（例: `1.2` で長め、`0.8` で短め）
- トレーニングは2フェーズ:
  - **Phase 1**: RF/DiT 本体（body）の学習
  - **Phase 2**: Duration Predictor のみの学習（`--train-mode duration_only`）

### 新規ファイル
- `irodori_tts/duration.py` — テキストから補助特徴量（句読点数、かな/漢字比率、絵文字数等）を抽出する関数群
- `configs/train_500m_v3_phase1_body.yaml` — Phase 1 設定
- `configs/train_500m_v3_phase2_duration.yaml` — Phase 2 設定

### 主要パラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_duration_predictor` | `false` | Duration Predictor を有効化 |
| `train_mode` | `rf` | `duration_only` で Duration Predictor のみ学習 |
| `duration_loss_weight` | `0.1` | RF loss と同時学習時の重み |
| `duration_architecture` | `token_sum_adarn_zero_no_aux` | トークンごとのフレーム寄与合算方式 |
| `duration_token_init_frames` | `9.0` | 初期予測値のフレーム/トークン |

---

## 2. Speaker Inversion（話者埋め込み学習） ⭐ 新機能

モデル全体を凍結し、少数の学習可能なスピーカー埋め込みトークンのみを学習する機能です。

### 用途
- 特定の話者の音声を参照音声なしで再現したい場合
- 毎回参照音声を指定する代わりに、学習済みの `.speaker.safetensors` ファイルを使用

### 推論での使い方
```bash
uv run python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed path/to/checkpoint.speaker.safetensors \
  --text "こんにちは" \
  --output-wav output.wav
```

### 新規ファイル
- `irodori_tts/speaker_inversion.py` — 埋め込みの保存/読み込みヘルパー
- `configs/train_500m_v3_speaker_inversion.yaml` — Speaker Inversion 設定

### 主要パラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `speaker_inversion_enabled` | `false` | Speaker Inversion モードの有効化 |
| `speaker_inversion_tokens` | `16` | 学習するトークン数 |
| `speaker_inversion_init_std` | `0.02` | ランダム初期化の標準偏差 |
| `speaker_uncond_mode` | `mask` | CFG 無条件モード（`mask` / `noise`） |

---

## 3. Sway Sampling（高速推論）

F5-TTS のアイデアに基づく新しいタイムステップスケジューリング手法が追加されました。

### 効果
- **ステップ数を大幅に削減** しつつ品質を維持（例: 40 → 6 ステップ）
- 負の `sway_coeff` でノイズ側（初期ステップ）にスケジュール解像度を集中

### 使い方
```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "テスト文章" \
  --ref-wav ref.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav output.wav
```

### 新規パラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--t-schedule-mode` | `linear` | `sway` で Sway Sampling 有効化 |
| `--sway-coeff` | `-1.0` | 負値: ノイズ側集中、正値: データ側集中 |

---

## 4. 音声透かし（Watermark）

[SilentCipher](https://github.com/sony/silentcipher) による音声透かし機能が追加されました。

### 概要
- 生成された音声に `IRDTS` というペイロードを埋め込む
- `silentcipher` パッケージが未インストールの場合は自動的にスキップ（エラーにならない）
- 新規ファイル: `irodori_tts/watermark.py`

---

## 5. 絵文字パレットUI

Gradio UIにクリッカブルな絵文字パレットが追加されました。

### 概要
- テキスト入力欄に隣接して、感情や発話スタイルを指定する絵文字をワンクリックで挿入
- 約 45 種の絵文字がサポート（囁き👂、笑い🤭、怒り😠、早口⏩ など）
- 新規ファイル: `irodori_tts/gradio_emoji_palette.py`

---

## 6. モデルアーキテクチャの拡張

`irodori_tts/model.py` に約 650 行の追加があり、以下のモジュールが新設されました:

| モジュール | 概要 |
|-----------|------|
| `DurationPredictor` | テキスト特徴量から発話長を予測するモジュール |
| `AttentionPooling` | 可変長シーケンスを固定ベクトルにプーリング |
| `CrossAttentionPooling` | 異なる次元のクエリ・コンテキスト間のクロスアテンション |
| `DurationSwiGLUBlock` | AdaRN-Zero 条件付き残差 SwiGLU ブロック |

---

## 7. トレーニングの改善

### 可変長ターゲット
- v2: `fixed_target_latent_steps: 750`（30秒固定）
- v3: `fixed_target_latent_steps: null`（可変長）

### RF Loss モード
- v2: `rf_loss_mode: echo`（デフォルト）
- v3: `rf_loss_mode: utterance_mean`（可変長向け、発話単位の平均化）

### LoRA 改善
- `lora_modules_to_save: auto` — v3 では Duration Predictor を自動的にアダプタに含める
- 動的 LoRA ロード（`--compile-model` 無効時）のサポート追加

### DDP 対応の強化
- `--ddp-find-unused-parameters` オプション追加

---

## 8. 推論パラメータの拡充

### Speaker K/V Scaling（実験的）
| パラメータ | 説明 |
|-----------|------|
| `--speaker-kv-scale` | スピーカーコンテキストの K/V 投影スケーリング |
| `--speaker-kv-min-t` | スケーリングが有効なタイムステップ下限 |
| `--speaker-kv-max-layers` | スケーリング対象のレイヤー数制限 |

### CFG ガイダンスモード
- `independent`（デフォルト）: 各条件に独立した無条件ブランチ
- `joint`: すべての条件を同時にドロップ（VRAM節約）
- `alternating`: ステップごとにドロップ条件を切り替え

---

## 9. AMD ROCm サポート（v3リリース後の追加）

- `pyproject.toml` に ROCm バックエンドの optional dependency を追加
- Linux 上の AMD GPU で CUDA 不要のトレーニング/推論が可能に

---

## 10. WAV 保存の dtype 安全対策（v3リリース後の追加）

- WAV ファイル書き込み時の dtype 変換を安全に処理するパッチ
- float64 等の非標準 dtype でもエラーなく保存可能に

---

## 11. その他の変更

- `docs/parameters.md` — 全パラメータの詳細ガイドが新規追加（約 450 行）
- `pyproject.toml` — `silentcipher` 依存の追加、ROCm バックエンド定義
- `requirements.txt` — `silentcipher` 追加
- `.gitignore` — `data/` ディレクトリの除外追加

---

## v2 → v3 移行時の注意点

> [!IMPORTANT]
> - v2 チェックポイントは v3 コードでそのまま動作する（後方互換性あり）
> - v2 チェックポイントには Duration Predictor が含まれないため、`--seconds` 省略時は従来通り 30 秒にフォールバックする
> - VoiceDesign チェックポイントは現時点で v2 ベースのまま（v3 VoiceDesign は未リリース）
> - v3 ベースチェックポイントのモデルファイルは `Irodori-TTS-500M-v3.safetensors` に変更
