# Speaker Inversion 学習手順

特定の話者の声を `.speaker.safetensors` ファイルとして学習・保存し、推論時に参照音声の代わりに使う手順をまとめたドキュメントです。

> 公式情報のソース: [README.md](../../README.md) `#### Speaker Inversion` セクション、[docs/parameters.md](../../docs/parameters.md) `### Speaker Inversion` セクション、[configs/train_500m_v3_speaker_inversion.yaml](../../configs/train_500m_v3_speaker_inversion.yaml)

---

## 概要

- **何をしているか**: ベースモデル本体の重みは完全に凍結し、少数（デフォルト16個）の **話者埋め込みトークンのみ** を勾配降下で学習します
- **何が得られるか**: 学習対象話者を表す埋め込みだけが入った軽量な `.speaker.safetensors` ファイル
- **何に使うか**: 推論時に `--ref-embed` で渡すと、参照音声 (`--ref-wav`) なしでその話者の声で合成できる

通常のファインチューニング (LoRA / フルパラメータ) と違い、モデル本体は変更しません。「テキスト反転 (Textual Inversion)」の話者版だと考えると理解しやすいです。

---

## 前提

| 項目 | 内容 |
|------|------|
| ベースチェックポイント | `Irodori-TTS-500M-v3.safetensors`（v3 リリース重み） |
| 学習データ | 対象話者の音声 + 書き起こしテキスト |
| GPU | 単 GPU で可。`gradient_checkpointing: true` で VRAM 削減可能 |
| 学習時間 | デフォルト `max_steps: 3000` の小規模学習 |

> `speaker_id` は不要です。1 つの埋め込みを学習するため、manifest 内のサンプルはすべて同じ対象話者の音声であることが前提です。

---

## 手順1: 学習用音声データを用意する

### 必要な音声データの条件

| 項目 | 目安 |
|------|------|
| 総音声長 | 1〜10 分程度（多いほど安定するが過学習にも注意） |
| 1ファイルあたりの長さ | 数秒〜30 秒程度。コーデック側の上限が約 30 秒（`--max-seconds` で制御可能） |
| 書き起こし | 必須。誤字脱字は埋め込みの劣化につながる |
| ノイズ・BGM | 少ないほどよい。学習音声に乗っているノイズも「話者の特徴」として一緒に学習されてしまう |
| サンプリングレート | 任意（DACVAE 側で自動リサンプリングされる） |
| 話者の一貫性 | **同じ話者の音声だけ** にすること。複数話者を混ぜると 1 つの埋め込みに平均化されてしまう |

### 配置パターンA: `audiofolder` 形式（推奨・最も手軽）

`prepare_manifest.py` の内部では Hugging Face の `load_dataset()` が呼ばれるため、HF 公式の `audiofolder` ビルダーがそのまま使えます。以下のように音声ファイルと `metadata.csv` を 1 ディレクトリにまとめます。

```
data/my_speaker/
├── metadata.csv
├── 001.wav
├── 002.wav
└── 003.wav
```

`metadata.csv` の中身（1 列目が `file_name` 固定、2 列目に書き起こし）:

```csv
file_name,text
001.wav,こんにちは、これはテストです。
002.wav,今日はいい天気ですね。
003.wav,音声合成のサンプルです。
```

manifest 生成コマンド:

```bash
uv run python prepare_manifest.py \
  --dataset audiofolder \
  --data-files "data/my_speaker/**" \
  --audio-column audio \
  --text-column text \
  --output-manifest data/target_speaker_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

> `audiofolder` ビルダーは `metadata.csv` の代わりに `metadata.jsonl` でも認識します。

### 配置パターンB: 既存の Hugging Face データセットを使う

公開済み・社内 Hub のデータセットがある場合はそのまま指定できます。

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/target_speaker_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

### 出力されるファイル

| パス | 中身 | 備考 |
|------|------|------|
| `data/latents/*.pt` | 各音声を DACVAE エンコードしたラテント | サイズが大きくなりがち。複数モデルを学習する際は使い回し可能 |
| `data/target_speaker_manifest.jsonl` | 1 行 1 サンプルの JSON（`text` / `latent_path` / `num_frames` 等） | これを `train.py --manifest` に渡す |

### Speaker Inversion ならではの注意点

- `--speaker-column` は **指定しなくてよい**。Speaker Inversion は 1 つの共有埋め込みを学習するモードで、モデル側の `speaker_id` 条件付け branch を使わないため
- `--caption-column` も不要（VoiceDesign 学習でのみ使用）
- 詳しい全オプションは [docs/parameters.md `## Manifest Preparation Parameters`](../../docs/parameters.md) を参照

---

## 手順2: 学習を実行する

専用の設定ファイル [configs/train_500m_v3_speaker_inversion.yaml](../../configs/train_500m_v3_speaker_inversion.yaml) を使い、ベース v3 チェックポイントから初期化します。

```bash
uv run python train.py \
  --config configs/train_500m_v3_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --output-dir outputs/speaker_inversion/<話者名>
```

- `--init-checkpoint` は **必須**。ベース TTS モデルは凍結して使うため、学習済み重みからの初期化が前提となります（未指定だと `ValueError` で停止）
- 出力ディレクトリには定期的に `checkpoint_<step>.speaker.safetensors` が保存され、最終的に `checkpoint_final.speaker.safetensors` が生成されます
- **これが目的のファイル**。中身は埋め込みテンソルのみで、ベースモデル本体の重みは含まれません

### VRAM が足りない場合

YAML 内で `gradient_checkpointing: true` がすでに有効です。それでも足りない場合は `batch_size` を下げる、`precision: bf16` を維持する、などで対応します。

---

## 手順3: 推論で使う

学習で得た `.speaker.safetensors` を `--ref-embed` で渡します。`--ref-wav` / `--ref-latent` / `--no-ref` とは排他です。

```bash
uv run python infer.py \
  --checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --ref-embed outputs/speaker_inversion/<話者名>/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

Gradio UI (`my/gradio_ref.py`) では `Speaker Embedding Upload` または `Speaker Embedding Path` に同じファイルを指定すれば使えます。

---

## 主要パラメータ

### Speaker Inversion 固有

| YAML / CLI | デフォルト | 役割 |
|------------|-----------|------|
| `speaker_inversion_enabled` / `--speaker-inversion` | `false` | Speaker Inversion モードを有効化（ベースを凍結し、埋め込みのみ学習） |
| `speaker_inversion_tokens` / `--speaker-inversion-tokens` | `16` | 学習する埋め込みトークン数。多いほど表現力が増えるが過学習リスクも上がる |
| `speaker_inversion_init_std` / `--speaker-inversion-init-std` | `0.02` | 埋め込みのランダム初期化時の標準偏差 |
| `speaker_inversion_init_embedding` / `--speaker-inversion-init-embedding` | `None` | 既存の `.speaker.safetensors` から学習を再開／ウォームスタートする場合に指定 |

### 学習挙動に効くパラメータ

| パラメータ | YAML デフォルト | 補足 |
|------------|----------------|------|
| `learning_rate` | `0.01` | 通常の TTS 学習より大きめ。少数トークンだけを更新するため大きな学習率で十分動く |
| `max_steps` | `3000` | データ量・声の複雑さに応じて増減 |
| `batch_size` | `16` | VRAM に応じて調整 |
| `gradient_checkpointing` | `true` | VRAM 削減（速度はやや低下） |
| `text_condition_dropout` / `speaker_condition_dropout` / `caption_condition_dropout` | すべて `0.0` | **必ず 0 のままにする**。埋め込みが「対象話者を無条件で再現する」よう学習させる目的のため、ドロップアウトすると目的と矛盾する |
| `train_mode` | `rf` | Speaker Inversion は RF モードのみ対応（duration_only は不可） |

---

## 制約・注意点

学習開始前に train.py 側でチェックされ、違反すると `ValueError` で停止します。

- **`--init-checkpoint` 必須**: 凍結する本体重みを与えるため
- **`--resume` 不可**: チェックポイントは埋め込みのみで optimizer 状態は持たないため。再開したい場合は代わりに `--speaker-inversion-init-embedding` で前回の `.speaker.safetensors` から続行する
- **LoRA と併用不可**
- **`train_mode` は `rf` のみ**
- **`caption_warmup` 不可**
- **モデルは speaker conditioning 対応である必要がある**（v3 ベースは対応済み）

---

## 既存の埋め込みから続きを学習する

途中で打ち切った学習を再開したり、既存話者を起点に微調整したい場合は `speaker_inversion_init_embedding` を使います。

YAML に書く場合:

```yaml
train:
  speaker_inversion_init_embedding: outputs/speaker_inversion/foo/checkpoint_final.speaker.safetensors
```

または CLI で:

```bash
uv run python train.py \
  --config configs/train_500m_v3_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors \
  --speaker-inversion-init-embedding outputs/speaker_inversion/foo/checkpoint_final.speaker.safetensors \
  --output-dir outputs/speaker_inversion/foo_continue
```

---

## 参照音声方式 (`--ref-wav`) との推論速度比較

リポジトリ内に公式ベンチマーク値はないため定性的な比較になりますが、ソースから読み取れる差は以下のとおりです。

### `--ref-wav` 方式で毎回走る処理（`--ref-embed` ではスキップされる）

[irodori_tts/inference_runtime.py:700-720](../../irodori_tts/inference_runtime.py#L700-L720) の参照音声ロード経路では、推論呼び出しのたびに次が実行されます:

1. 音声ファイル読み込み (`_load_audio`)
2. ラウドネス正規化（デフォルト `-16.0 dB`）
3. **DACVAE エンコード** (`codec.encode_waveform`) — 最大 30 秒分の参照音声をラテント化
4. **Speaker Encoder forward** — 約 750 ステップ程度のラテントを 8 層の speaker transformer に通す

### `--ref-embed` 方式で走る処理

[irodori_tts/speaker_inversion.py:40](../../irodori_tts/speaker_inversion.py#L40) のクラス docstring にもあるとおり、Speaker Inversion 埋め込みは **「reference latent speaker encoder をバイパスする」** ように設計されています。実際の処理は以下のみです:

1. `.speaker.safetensors` のロード（数十 KB のテンソル 1 つ）
2. その埋め込みをそのまま speaker_state として DiT のクロスアテンションに渡す

### 速度に効くポイント

| 観点 | `--ref-embed` が速くなる理由 |
|------|------------------------------|
| **推論前のセットアップ** | DACVAE エンコード + Speaker Encoder forward が丸ごと消える。長い参照音声（30 秒）を使っているほど効果大 |
| **DiT 拡散ループの 1 ステップあたりコスト** | クロスアテンションの K/V トークン数が **約 750 → 16 トークンに激減**（デフォルト `speaker_inversion_tokens: 16`）。理論上アテンション計算量も大幅減 |
| **`context_kv_cache: true`（デフォルト有効）** | speaker K/V は全ステップで使い回されるため、参照音声方式でも K/V を毎ステップ再計算するわけではない。それでも K/V テンソル自体が小さいぶん `--ref-embed` が有利 |
| **音声生成本体（DACVAE デコード等）** | ここは条件に関係なく走るので変わらない |

### 実用上の感覚

- **典型的な 10〜30 秒の音声生成**: 全体時間は拡散ループが支配的なため、Speaker Inversion で短縮されるのは **数百ミリ秒〜1 秒程度** のオーダーが現実的
- **短文を大量にバッチ生成するケース**: 1 回あたりのセットアップコスト削減が累積するため、相対的な効果は大きい
- **ストリーミング・低レイテンシ用途**: 「最初の音が出るまで（TTFB）」の短縮には効きやすい

> 厳密な数値が必要な場合は、同じテキスト/seed で `--ref-wav` と `--ref-embed` を順に実行し、`time` コマンドや Python の `time.perf_counter` で計測してください。

---

## 関連ドキュメント

- 概念的な背景: [my/docs/v2_to_v3_changes.md](v2_to_v3_changes.md) `## 2. Speaker Inversion` セクション
- 全パラメータの英語リファレンス: [docs/parameters.md](../../docs/parameters.md)
- 推論側 (`--ref-embed`) のオプション一覧: [docs/parameters.md](../../docs/parameters.md) `## Inference Parameters` セクション
- 設定ファイル本体: [configs/train_500m_v3_speaker_inversion.yaml](../../configs/train_500m_v3_speaker_inversion.yaml)
- 実装: [irodori_tts/speaker_inversion.py](../../irodori_tts/speaker_inversion.py)
