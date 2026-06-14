# Irodori-TTS 日本語ガイド

[English](README.md) | [日本語](README_JP.md)

**Irodori-TTS** は、Flow Matching ベースの日本語TTSモデルです。v3コードベースでは、通常の参照音声付き音声合成に加えて、**VoiceDesign** チェックポイントによるテキスト＋キャプション指定の音声デザインにも対応しています。

モデルの詳細やサンプル音声は、Hugging Face のモデルカードを参照してください。

- Base model: `Aratako/Irodori-TTS-500M-v3`
- VoiceDesign model: `Aratako/Irodori-TTS-600M-v3-VoiceDesign`

> [!IMPORTANT]
> `main` ブランチは v3 系のコードベースです。v2チェックポイントの推論互換性は維持されていますが、v1チェックポイントやv1時代の前処理とは互換性がありません。

## 主な機能

- 参照音声を使ったゼロショット音声合成
- 参照音声なしの `--no-ref` 推論
- VoiceDesign: テキスト、参照音声、キャプションによる声質・話し方制御
- 絵文字を含むテキストによる表現制御対応
- 自動発話時間予測
- CLI / Gradio Web UI / Hugging Face Hub チェックポイント対応
- 長文ナレーション向けのチャンク分割生成とWAV/MP3出力
- LoRAファインチューニング、Speaker Inversion、分散学習などの研究・開発向け機能

## インストール

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync --extra cu128
```

NVIDIA CUDA 12.8環境では `cu128` を使います。CPU、ROCm、Intel XPU などを使う場合は、英語版READMEの Installation セクションも参照してください。

同期後のコマンド実行では、環境を再同期しないように `uv run --no-sync ...` を使うことを推奨します。


## Windowsユーザー向けクイックスタート

WindowsでVoiceDesignの長文生成を手軽に試したい場合は、`windows/` フォルダ内の補助バッチを使えます。

1. `windows/_IRODORI_LOCAL_CONFIG.example.bat` をコピーして、`windows/_IRODORI_LOCAL_CONFIG.bat` にリネームします。
2. Gradio UIを使う場合は、`windows/_LAUNCH_WebUI_LONG.bat` をダブルクリックします。
3. CLIで長文生成する場合は、UTF-8のテキストファイルを `windows/_LAUNCH_CLI_LONG.bat` にドラッグ＆ドロップします。

バッチファイルは `windows/` 内に置いたまま実行しますが、作業ディレクトリは自動的にリポジトリルートへ移動します。`_vendor`、`_models`、出力フォルダもリポジトリルート側に作成されます。

## 基本的な使い方

### 参照音声ありの推論

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### 参照音声なしの推論

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign 推論

キャプションだけで声質・話し方を指定する例です。

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

参照音声とキャプションを併用することもできます。

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --ref-wav path/to/reference.wav \
  --caption "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。" \
  --output-wav outputs/sample_voice_design_clone.wav
```

## Gradio Web UI

通常モデル向けUI:

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

VoiceDesign向けUI:

```bash
uv run --no-sync python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

ブラウザで `http://localhost:7860` または `http://localhost:7861` を開いて使用します。

## 長文生成

長いナレーションを一度に生成すると、無音化、反復、テンポ崩れなどが起きやすい場合があります。`--long` を使うと、UTF-8テキストファイルを句読点や改行で分割し、各チャンクを順番に生成してから連結できます。

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text-file samples/narration.txt \
  --caption "落ち着いたナレーション調で、自然なテンポで読み上げてください。" \
  --no-ref \
  --long \
  --chunk-max-chars 80 \
  --pause-ms 250 \
  --output-format mp3 \
  --output-dir outputs/long
```

主なオプション:

- `--chunk-max-chars`: 1チャンクあたりのおおよその最大文字数
- `--pause-ms`: チャンク間に挿入する無音時間
- `--output-format wav|mp3`: 最終出力形式
- `--keep-chunk-wavs / --no-keep-chunk-wavs`: 中間WAVを残すかどうか
- `--save-json / --no-save-json`: JSON / JSONL メタデータ保存
- `--write-mp3-tag / --no-write-mp3-tag`: MP3コメントタグへのパラメータ書き込み

MP3出力には `pydub`、`mutagen`、FFmpeg が必要です。

## Windows用補助バッチ

Windowsユーザー向けに、`windows/` フォルダ内の補助バッチを使うこともできます。

```text
windows/_IRODORI_LOCAL_CONFIG.example.bat
windows/_LAUNCH_WebUI_LONG.bat
windows/_LAUNCH_CLI_LONG.bat
```

基本手順:

1. `windows/_IRODORI_LOCAL_CONFIG.example.bat` をコピーします。
2. コピーしたファイルを `windows/_IRODORI_LOCAL_CONFIG.bat` にリネームします。
3. 必要に応じてモデルパス、ポート、出力先、オフライン設定などを書き換えます。
4. Gradioを使う場合は `windows/_LAUNCH_WebUI_LONG.bat` をダブルクリックします。
5. CLI長文生成を使う場合は、UTF-8のテキストファイルを `windows/_LAUNCH_CLI_LONG.bat` にドラッグ＆ドロップします。

バッチファイルは `windows/` 内に置いたまま実行する想定です。ただし、作業ディレクトリは自動的にリポジトリルートへ移動します。`_vendor`、`_models`、出力フォルダは `windows/` の中ではなく、リポジトリルート側に作成されます。

## オフライン利用

完全オフラインで使いたい場合は、Hugging Face repo IDではなく、ローカルの `.safetensors` やcodecファイルへのパスを指定してください。Windows補助バッチでは、`_IRODORI_LOCAL_CONFIG.bat` 内で `IRODORI_TTS_OFFLINE=1` に設定できます。

## トラブルシューティング

- `--seconds` を長くしすぎるより、長文では `--long` による分割生成を推奨します。
- MP3出力に失敗する場合は、FFmpegがPATHに入っているか、または `IRODORI_FFMPEG_EXE` / `--ffmpeg-exe` が正しいか確認してください。
- Windowsバッチを使う場合、入力テキストファイルはUTF-8で保存してください。
- 生成結果の詳細なパラメータを残したい場合は、JSON/JSONLメタデータ保存を有効にしてください。

## 学習・開発向け情報

学習、LoRA、Speaker Inversion、設定ファイルの詳細については、英語版READMEおよび `docs/parameters.md` を参照してください。

## ライセンス

コードはMITライセンスです。モデル重み、学習データ、外部依存ライブラリのライセンスについては、それぞれの配布元を確認してください。
