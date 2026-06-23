@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Irodori-TTS VoiceDesign Gradio Long Generation 起動用
rem このbatは windows フォルダ内から、リポジトリルートへ移動して実行します。
rem ============================================================

cd /d "%~dp0.."

if not exist "windows\_IRODORI_LOCAL_CONFIG.bat" (
    echo [setup] windows\_IRODORI_LOCAL_CONFIG.bat が見つかりません。
    echo [setup] 設定例をコピーします。必要に応じて編集してください。
    copy "windows\_IRODORI_LOCAL_CONFIG.example.bat" "windows\_IRODORI_LOCAL_CONFIG.bat" >nul
)

set "IRODORI_CONFIG_CALLED_BY_LAUNCHER=1"
call "windows\_IRODORI_LOCAL_CONFIG.bat"
set "IRODORI_CONFIG_CALLED_BY_LAUNCHER="

if /I "%IRODORI_TTS_OFFLINE%"=="1" (
    set "HF_HUB_OFFLINE=1"
    set "TRANSFORMERS_OFFLINE=1"
    set "HF_HUB_DISABLE_TELEMETRY=1"
)
set "UV_LINK_MODE=copy"

if not exist "%IRODORI_VENDOR_DIR%" mkdir "%IRODORI_VENDOR_DIR%"

if not exist "%IRODORI_FFMPEG_EXE%" (
    echo [setup] ローカルffmpegが見つかりません。ffmpeg-release-essentials.zip をダウンロードします...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%IRODORI_FFMPEG_URL%' -OutFile '%IRODORI_FFMPEG_ZIP%'"
    if errorlevel 1 (
        echo [error] ffmpegのダウンロードに失敗しました。
        echo [error] 手動で以下に ffmpeg.exe を配置してください:
        echo         %IRODORI_FFMPEG_EXE%
        pause
        exit /b 1
    )

    echo [setup] ffmpegを展開しています...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $tmp='%IRODORI_VENDOR_DIR%\ffmpeg_extract'; if(Test-Path $tmp){Remove-Item $tmp -Recurse -Force}; Expand-Archive -Path '%IRODORI_FFMPEG_ZIP%' -DestinationPath $tmp -Force; $bin=Get-ChildItem -Path $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1; if(-not $bin){throw 'ffmpeg.exe not found in archive'}; $root=$bin.Directory.Parent.FullName; if(Test-Path '%IRODORI_FFMPEG_ROOT%'){Remove-Item '%IRODORI_FFMPEG_ROOT%' -Recurse -Force}; New-Item -ItemType Directory -Path '%IRODORI_FFMPEG_ROOT%' | Out-Null; Copy-Item -Path (Join-Path $root '*') -Destination '%IRODORI_FFMPEG_ROOT%' -Recurse -Force; Remove-Item $tmp -Recurse -Force"
    if errorlevel 1 (
        echo [error] ffmpegの展開に失敗しました。
        pause
        exit /b 1
    )
)

set "PATH=%IRODORI_FFMPEG_ROOT%\bin;%PATH%"

uv run --no-sync python -c "import gradio" >nul 2>nul
if errorlevel 1 (
    echo [setup] Python依存関係がまだインストールされていません。
    echo [setup] uv sync --extra %IRODORI_UV_EXTRA% を実行します...
    uv sync --extra %IRODORI_UV_EXTRA%
    if errorlevel 1 (
        echo [error] uv sync に失敗しました。
        pause
        exit /b 1
    )
)

rem モデル/Codecが無ければ _models 以下へ取得するか確認します。
if not exist "%IRODORI_TTS_CHECKPOINT%" (
    echo [setup] ローカルモデルが見つかりません。_models 以下へ取得するか確認します...
    call "windows\_DOWNLOAD_MODELS.bat" --no-pause
    if errorlevel 1 (
        echo [error] モデルの準備に失敗しました。
        pause
        exit /b 1
    )
)
if not exist "%IRODORI_CODEC_REPO%" (
    echo [setup] ローカルCodecが見つかりません。_models 以下へ取得するか確認します...
    call "windows\_DOWNLOAD_MODELS.bat" --no-pause
    if errorlevel 1 (
        echo [error] Codecの準備に失敗しました。
        pause
        exit /b 1
    )
)

rem 古い設定ファイルでHF repo IDが残っている場合も、_modelsに保存済みならローカルファイルを優先します。
if exist "%IRODORI_LOCAL_MODELS_DIR%\Irodori-TTS-600M-v3-VoiceDesign\model.safetensors" set "IRODORI_TTS_CHECKPOINT=%IRODORI_LOCAL_MODELS_DIR%\Irodori-TTS-600M-v3-VoiceDesign\model.safetensors"
if exist "%IRODORI_LOCAL_MODELS_DIR%\Semantic-DACVAE-Japanese-32dim\weights.pth" set "IRODORI_CODEC_REPO=%IRODORI_LOCAL_MODELS_DIR%\Semantic-DACVAE-Japanese-32dim\weights.pth"

echo [launch] Irodori-TTS VoiceDesign Gradio を起動します...
echo [config] server: http://%IRODORI_SERVER_NAME%:%IRODORI_SERVER_PORT%
echo [config] output: %IRODORI_OUTPUT_DIR%

uv run --no-sync python gradio_app_voicedesign.py --server-name "%IRODORI_SERVER_NAME%" --server-port %IRODORI_SERVER_PORT%
pause
