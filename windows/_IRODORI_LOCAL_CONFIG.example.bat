@echo off
rem ============================================================
rem Irodori-TTS Windows用ローカル設定例
rem
rem 使い方:
rem   1. このファイルは初回起動時に _IRODORI_LOCAL_CONFIG.bat へコピーされます。
rem   2. 必要に応じて _IRODORI_LOCAL_CONFIG.bat の値を書き換えます。
rem   3. _LAUNCH_WebUI_LONG.bat または _LAUNCH_CLI_LONG.bat を実行します。
rem
rem 注意:
rem   このファイルを直接ダブルクリックしても、環境変数はその画面の中だけで有効です。
rem   通常はランチャーbatから読み込ませてください。
rem ============================================================

rem この設定ファイルが置かれている windows フォルダと、リポジトリルートを求めます。
set "IRODORI_WINDOWS_DIR=%~dp0"
for %%I in ("%~dp0..") do set "IRODORI_ROOT=%%~fI"

rem 補助ファイルの保存先です。
rem windows フォルダ配下ではなく、リポジトリルート側に作成します。
set "IRODORI_VENDOR_DIR=%IRODORI_ROOT%\_vendor"
set "IRODORI_LOCAL_MODELS_DIR=%IRODORI_ROOT%\_models"
set "IRODORI_FFMPEG_ROOT=%IRODORI_VENDOR_DIR%\ffmpeg"
set "IRODORI_FFMPEG_EXE=%IRODORI_FFMPEG_ROOT%\bin\ffmpeg.exe"
set "IRODORI_FFMPEG_ZIP=%IRODORI_VENDOR_DIR%\ffmpeg-release-essentials.zip"
set "IRODORI_FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

rem 1 にすると Hugging Face へアクセスしないローカル/オフライン運用を想定します。
rem 既定では _models 以下のローカルファイルを読みます。
rem モデル/Codec が無い場合のみ、ランチャーまたは _DOWNLOAD_MODELS.bat がオンライン取得するか確認します。
set "IRODORI_TTS_OFFLINE=1"

rem uv sync で使う追加依存グループです。
rem NVIDIA GPU 環境では cu128、CPUのみの場合は cpu に変更してください。
set "IRODORI_UV_EXTRA=cu128"

rem モデル/Codec のダウンロード元です。
rem _DOWNLOAD_MODELS.bat が _models 以下へ保存します。
set "IRODORI_TTS_SOURCE_REPO=Aratako/Irodori-TTS-600M-v3-VoiceDesign"
set "IRODORI_CODEC_SOURCE_REPO=Aratako/Semantic-DACVAE-Japanese-32dim"
set "IRODORI_TOKENIZER_REPO=llm-jp/llm-jp-3-150m"

rem 既定では _models 以下のローカルファイルを読み込みます。
rem ファイルが無い場合はランチャーが _DOWNLOAD_MODELS.bat を呼び出します。
set "IRODORI_TTS_CHECKPOINT=%IRODORI_LOCAL_MODELS_DIR%\Irodori-TTS-600M-v3-VoiceDesign\model.safetensors"
set "IRODORI_CODEC_REPO=%IRODORI_LOCAL_MODELS_DIR%\Semantic-DACVAE-Japanese-32dim\weights.pth"

rem Gradioサーバー設定です。
set "IRODORI_SERVER_NAME=127.0.0.1"
set "IRODORI_SERVER_PORT=7861"

rem Gradio Long Generation の出力先です。
set "IRODORI_OUTPUT_DIR=%IRODORI_ROOT%\gradio_outputs_voicedesign"

rem Gradio UI の Long Generation / Export 初期値です。
set "IRODORI_LONG_CHUNK_MAX_CHARS=80"
set "IRODORI_LONG_PAUSE_MS=250"
set "IRODORI_LONG_OUTPUT_FORMAT=mp3"
set "IRODORI_LONG_KEEP_FINAL_WAV_WHEN_MP3=0"
set "IRODORI_LONG_KEEP_CHUNK_WAVS=1"
set "IRODORI_LONG_SAVE_JSON=1"
set "IRODORI_LONG_WRITE_MP3_TAG=1"

rem CLI用の細かい既定値はここには置きません。
rem _LAUNCH_CLI_LONG.bat にテキストファイルをドラッグ&ドロップするか、
rem コマンドラインから infer.py --long を直接指定してください。

if not defined IRODORI_CONFIG_CALLED_BY_LAUNCHER (
    echo.
    echo [info] Irodori-TTS Windows用ローカル設定例を表示しました。
    echo [info] このファイルを直接実行しても、設定はこの画面の中だけで有効です。
    echo [info] 通常は _LAUNCH_WebUI_LONG.bat または _LAUNCH_CLI_LONG.bat を実行してください。
    echo.
    echo IRODORI_ROOT=%IRODORI_ROOT%
    echo IRODORI_TTS_CHECKPOINT=%IRODORI_TTS_CHECKPOINT%
    echo IRODORI_CODEC_REPO=%IRODORI_CODEC_REPO%
    echo IRODORI_OUTPUT_DIR=%IRODORI_OUTPUT_DIR%
    echo.
    pause
)
