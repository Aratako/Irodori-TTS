@echo off
setlocal EnableExtensions

rem ============================================================
rem Irodori-TTS モデル/Codec ダウンロード用
rem _models 以下へ model.safetensors / weights.pth を保存します。
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

if not defined IRODORI_UV_EXTRA set "IRODORI_UV_EXTRA=cu128"
if not defined IRODORI_LOCAL_MODELS_DIR set "IRODORI_LOCAL_MODELS_DIR=%CD%\_models"
if not defined IRODORI_TTS_SOURCE_REPO set "IRODORI_TTS_SOURCE_REPO=Aratako/Irodori-TTS-600M-v3-VoiceDesign"
if not defined IRODORI_CODEC_SOURCE_REPO set "IRODORI_CODEC_SOURCE_REPO=Aratako/Semantic-DACVAE-Japanese-32dim"
if not defined IRODORI_TOKENIZER_REPO set "IRODORI_TOKENIZER_REPO=llm-jp/llm-jp-3-150m"
if not defined IRODORI_TTS_CHECKPOINT set "IRODORI_TTS_CHECKPOINT=%IRODORI_LOCAL_MODELS_DIR%\Irodori-TTS-600M-v3-VoiceDesign\model.safetensors"
if not defined IRODORI_CODEC_REPO set "IRODORI_CODEC_REPO=%IRODORI_LOCAL_MODELS_DIR%\Semantic-DACVAE-Japanese-32dim\weights.pth"

set "UV_LINK_MODE=copy"
set "HF_HUB_DISABLE_TELEMETRY=1"

if /I "%IRODORI_TTS_OFFLINE%"=="1" (
    if exist "%IRODORI_TTS_CHECKPOINT%" (
        if exist "%IRODORI_CODEC_REPO%" (
            echo [models] offline mode: local model files already exist.
            if /I not "%~1"=="--no-pause" pause
            exit /b 0
        )
    )

    echo [setup] IRODORI_TTS_OFFLINE=1 ですが、ローカルモデル/Codecが見つかりません。
    echo [setup] model : %IRODORI_TTS_CHECKPOINT%
    echo [setup] codec : %IRODORI_CODEC_REPO%
    echo.
    echo [confirm] オンラインに切り替えて _models 以下へダウンロードしますか？
    choice /C YN /N /M "[Y/N]: "
    if errorlevel 2 (
        echo [cancel] ダウンロードをキャンセルしました。
        if /I not "%~1"=="--no-pause" pause
        exit /b 1
    )

    echo [setup] このダウンロード処理の間だけオンラインモードに切り替えます。
    set "IRODORI_TTS_OFFLINE=0"
    set "HF_HUB_OFFLINE="
    set "TRANSFORMERS_OFFLINE="
)

if not exist "%IRODORI_LOCAL_MODELS_DIR%" mkdir "%IRODORI_LOCAL_MODELS_DIR%"

echo [setup] Python依存関係を確認します...
uv run --no-sync python -c "import huggingface_hub, transformers" >nul 2>nul
if errorlevel 1 (
    echo [setup] uv sync --extra %IRODORI_UV_EXTRA% を実行します...
    uv sync --extra %IRODORI_UV_EXTRA%
    if errorlevel 1 (
        echo [error] uv sync に失敗しました。
        if /I not "%~1"=="--no-pause" pause
        exit /b 1
    )
)

echo [models] model/codec/tokenizer を確認・ダウンロードします...
uv run --no-sync python windows\_download_models.py
if errorlevel 1 (
    echo [error] model/codec/tokenizer のダウンロードに失敗しました。
    if /I not "%~1"=="--no-pause" pause
    exit /b 1
)

echo [models] 完了しました。
if /I not "%~1"=="--no-pause" pause
