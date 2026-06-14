@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Irodori-TTS Long Generation CLI 起動用
rem 使い方:
rem   _LAUNCH_CLI_LONG.bat path\to\text.txt
rem   または、このbatへUTF-8のテキストファイルをドラッグ&ドロップします。
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

set "CLI_TEXT_FILE=%~1"
if not defined CLI_TEXT_FILE (
    echo [error] 入力テキストファイルが指定されていません。
    echo [hint] このbatへUTF-8のテキストファイルをドラッグ&ドロップするか、
    echo [hint] コマンドラインから第1引数として指定してください。
    echo.
    echo 例: windows\_LAUNCH_CLI_LONG.bat samples\narration.txt
    pause
    exit /b 1
)

if not exist "%CLI_TEXT_FILE%" (
    echo [error] 入力テキストファイルが見つかりません:
    echo         %CLI_TEXT_FILE%
    pause
    exit /b 1
)

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

uv run --no-sync python -c "import pydub, mutagen" >nul 2>nul
if errorlevel 1 (
    echo [setup] Long Generation用の任意パッケージ pydub / mutagen をインストールします...
    uv pip install pydub mutagen
    if errorlevel 1 (
        echo [error] pydub / mutagen のインストールに失敗しました。
        pause
        exit /b 1
    )
)

set "CHECKPOINT_ARG=--hf-checkpoint"
if exist "%IRODORI_TTS_CHECKPOINT%" set "CHECKPOINT_ARG=--checkpoint"

rem CLI補助batの最小既定値です。細かく調整したい場合は infer.py --long を直接実行してください。
set "CLI_OUTPUT_DIR=%IRODORI_ROOT%\cli_outputs"
set "CLI_OUTPUT_FORMAT=mp3"
set "CLI_CHUNK_MAX_CHARS=80"
set "CLI_PAUSE_MS=250"

echo [launch] Irodori-TTS Long Generation CLI を起動します...
echo [config] text: %CLI_TEXT_FILE%
echo [config] output: %CLI_OUTPUT_DIR%

uv run --no-sync python infer.py ^
  --long ^
  %CHECKPOINT_ARG% "%IRODORI_TTS_CHECKPOINT%" ^
  --codec-repo "%IRODORI_CODEC_REPO%" ^
  --text-file "%CLI_TEXT_FILE%" ^
  --output-dir "%CLI_OUTPUT_DIR%" ^
  --output-format "%CLI_OUTPUT_FORMAT%" ^
  --chunk-max-chars %CLI_CHUNK_MAX_CHARS% ^
  --pause-ms %CLI_PAUSE_MS% ^
  --model-device cuda ^
  --model-precision bf16 ^
  --codec-device cuda ^
  --codec-precision fp32 ^
  --num-steps 40 ^
  --duration-scale 1.0 ^
  --ffmpeg-exe "%IRODORI_FFMPEG_EXE%" ^
  --no-ref ^
  --no-keep-final-wav-when-mp3 ^
  --keep-chunk-wavs ^
  --save-json ^
  --write-mp3-tag
pause
