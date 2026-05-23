# Irodori-TTS — Fork by nizikaze

本家リポジトリ [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のフォークです。
本家の使い方・モデル情報はルートの [README.md](../README.md) を参照してください。

## このフォークについて

本家コードを壊さずに自分の実験・改造を重ねるため、独自コードはすべて `my/` ディレクトリ以下に置いています。

## ディレクトリ構成

```
my/
├── README.md              # このファイル
├── db.py                  # 生成履歴SQLiteモジュール
├── gradio_gen.py          # 独自生成UI（Gradio / VoiceDesign版）
├── gradio_ref.py          # 独自生成UI（Gradio / 参照音声版）
├── streamlit_history.py   # 生成履歴の閲覧・編集UI（Streamlit）
├── run_gen.sh             # 生成UI起動スクリプト (VoiceDesign版)
├── run_ref.sh             # 生成UI起動スクリプト (参照音声版)
├── run_history.sh         # 履歴閲覧UI起動スクリプト
├── data/
│   ├── generations.db     # 生成履歴DB（自動生成）
│   └── outputs/           # 生成した wav ファイル（自動生成）
└── docs/
    ├── fork-workflow.md
    ├── v2_to_v3_changes.md # v2からv3への変更点
    ├── myui-v1/
    │   ├── spec.md        # v1仕様書
    │   └── TODO.md        # v1実装TODO
    └── myui-v2/
        ├── spec.md        # v2仕様書
        └── TODO.md        # v2実装TODO
```

## 運用方針

- 本家ファイルは原則変更しない
- 本家の関数を使いたい場合は `my/` 側でimportしてラップする
- 本家の更新取り込みは `git fetch upstream && git rebase upstream/main`

詳細は [my/docs/fork-workflow.md](docs/fork-workflow.md) を参照。

## 独自UIの起動方法

### 生成UI（VoiceDesign版）

GradioによるVoiceDesignモデル向けのUI。

**簡単起動:** `my/run_gen.sh` を実行 → サーバー起動後にブラウザが自動で開きます（ポート: 7862）。

### 生成UI（参照音声版）

Gradioによる参照音声・Speaker Inversionモデル向けのUI。

**簡単起動:** `my/run_ref.sh` を実行 → サーバー起動後にブラウザが自動で開きます（ポート: 7863）。

コマンドラインから起動する場合:

```bash
# リポジトリルートで実行
python -m my.gradio_gen
# または
python -m my.gradio_ref
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--server-name` | `127.0.0.1` | バインドするホスト名 |
| `--server-port` | `7862`/`7863` | ポート番号 |
| `--share` | OFF | Gradio の公開リンクを生成 |
| `--debug` | OFF | デバッグモードで起動 |

### 閲覧UI（Streamlit）

生成履歴の一覧表示・検索・編集ができるUI。

**簡単起動:** `my/run_history.sh` を実行 → サーバー起動後にブラウザが自動で開きます。

コマンドラインから起動する場合:

```bash
# リポジトリルートで実行
streamlit run my/streamlit_history.py
```

起動後、ブラウザで `http://localhost:8501`（または指定ポート）にアクセスしてください。

---

## v3 (myui-v2) で追加された新機能・おすすめ設定

本家の v3 移行に伴い、独自UI（myui-v2）も新規サンプリングパラメータおよび最新機能に対応しました。

### 1. Sway Sampling
高速かつ高品質な推論を可能にする新しいサンプリングスケジューラです。
* **おすすめ設定 (高品質・高速生成)**:
  * `Num Steps`: `6`
  * `Time Schedule`: `sway`
  * `Sway Coeff`: `-1.0`
* `Time Schedule` を `linear` に切り替えた場合、`Sway Coeff` は自動的に無効化されます。

### 2. Duration Predictor (発話長の自動予測)
* `Seconds (blank=auto)` を空欄（ブランク）のままにすると、モデルの **Duration Predictor** が機能し、入力テキストに基づいて最適な発話長が自動的に計算されます。
* 手動で生成秒数を指定したい場合は、ここに数値を入力します（例: `5.5`）。
* `Duration Scale` スライダー（`0.5` 〜 `1.5`）を調整することで、話速（発話のスピード）を微調整できます（例: `1.0` が等倍、`1.2` でゆっくり、`0.8` で早口）。

### 3. 絵文字パレット
* テキスト入力欄のすぐ下にある絵文字（感情や囁き、笑いなどの音声スタイル調整用）のボタンをクリックするだけで、入力テキストのカーソル位置に自動的に絵文字が挿入されます。
* デフォルトでは折りたたまれており、クリックで開閉できます。

### 4. LoRA アダプタの動的ロード
* `Advanced (Optional)` アコーディオン内にある `LoRA Adapter Directory (optional)` に、学習済みの追加話者スタイル (LoRA) フォルダのパスを指定することで、推論時にそのスタイルを動的に適用できます。

### 5. Speaker Inversion (参照音声版 `gradio_ref.py` のみ)
* `Speaker Embedding (.speaker.safetensors) Upload` もしくは `Speaker Embedding Path (optional)` に学習済みの話者埋め込みファイル (`.safetensors`) を指定することで、その話者の声質で推論が可能になります。
* **注意**: `Reference Audio` と `Speaker Embedding` は排他仕様です。同時に両方を指定することはできません。どちらも未指定の場合は自動的に `no_ref` モードでの推論となります。

### 6. 履歴DB拡張 & 閲覧UI対応
* DB スキーマが自動的に拡張され、上記の v3 新パラメータ（`duration_scale` / `sway_coeff` / `lora_adapter` / `speaker_embedding` など）のほか、myUI のバージョン、推定されたモデル世代 (`v2` / `v3` / `v2-voicedesign`) も記録されます。
* 閲覧UI (Streamlit) の詳細カード内に、これらの追加されたメタ情報が「2段目のメタ情報」として表示されるようになりました。
* **後方互換性**: 古い myUI で生成された履歴（新カラムが `NULL`）についても、`—` や `auto` といった代替表示で崩れなく読み込める設計になっています。

