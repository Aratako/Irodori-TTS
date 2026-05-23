"""
Irodori-TTS 独自生成UI（Gradio）— 参照音声(Reference Audio)版。

本家 gradio_app.py をベースに、以下の変更を加えた生成画面:
- 候補グリッド（32枠）・num_candidates スライダーを廃止（常に1件生成）
- ファイル名規則を {YYYYMMDD_HHMMSS}_{seed}.wav に変更
- 直近5件の履歴表示（Audio×5）
- autoplay ON/OFF トグル
- Generate Forever / Cancel Forever ボタンによる連続生成制御
- 生成完了時に DB へ書き込み
- キュー再生（再生中に新しい音声が来ても中断せずキューに積む）

gradio_gen.py との主な違い:
- caption (スタイルプロンプト) の代わりに uploaded_audio (参照音声) を使用
- cfg_scale_caption の代わりに cfg_scale_speaker を使用
- speaker_kv_scale / speaker_kv_min_t / speaker_kv_max_layers パラメータがある
- max_text_len / max_caption_len パラメータがない
- チェックポイントのデフォルト候補に caption/voice_design フィルタがない

Usage:
    python -m my.gradio_ref [--server-name 127.0.0.1] [--server-port 7863]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import gradio as gr

# --------------------------------------------------------------------------- #
#  絵文字パレットコンポーネントのインポート
#
#  Why: 音声生成時のテキストに感情やニュアンスを表す絵文字（耳元の音、ため息など）を
#       簡単に挿入できるようにするため、本家の絵文字パレットUIを再利用する。
# --------------------------------------------------------------------------- #
from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette


# --------------------------------------------------------------------------- #
#  本家 gradio_app.py からユーティリティ関数をインポート
#
#  Why: 同じロジックの重複実装を避けるため、パラメータ解析・ランタイム管理
#       などの純粋なユーティリティは本家から再利用する。
#       build_ui() と _run_generation() は本ファイルで独自に再実装する。
# --------------------------------------------------------------------------- #
from gradio_app import (
    _build_runtime_key,
    _clear_runtime_cache,
    _default_checkpoint,
    _default_codec_device,
    _default_model_device,
    _format_timings,
    _load_model,
    _on_codec_device_change,
    _on_model_device_change,
    _on_t_schedule_mode_change,
    _parse_optional_float,
    _parse_optional_int,
    _parse_optional_str,
    _precision_choices_for_device,
    _resolve_ref_wav,
    _resolve_speaker_embedding,
)
from irodori_tts.inference_runtime import (
    SamplingRequest,
    get_cached_runtime,
    list_available_runtime_devices,
    save_wav,
)

# 自前の DB モジュール
from my import MY_UI_VERSION
from my.db import init_db, insert_generation, guess_model_version

# 直近履歴の最大表示件数
_MAX_HISTORY = 5

# キューの最大サイズ（これ以上溜まると古いものから破棄）
_MAX_QUEUE_SIZE = 10

# グローバルな停止フラグ管理用（session_hash -> 停止要求）
_session_stop_flags: dict[str, bool] = {}

# --------------------------------------------------------------------------- #
#  セッションごとの autoplay フラグ管理
#
#  Why: Gradio のジェネレータ関数は、inputs を関数呼び出し時に一度だけ評価する。
#       そのため Generate Forever のループ中に UI 上の autoplay チェックボックスを
#       変更しても、ジェネレータ内の autoplay 引数は最初の値のまま変わらない。
#       この問題を解決するため、autoplay の変更イベントでグローバル変数を更新し、
#       ジェネレータのループ内からそのフラグをリアルタイムに参照する。
# --------------------------------------------------------------------------- #
_session_autoplay_flags: dict[str, bool] = {}

# --------------------------------------------------------------------------- #
#  セッションごとの Live Update（プロンプト変更反映）管理
#
#  Why: autoplay と同様、Gradio のジェネレータ関数は inputs を関数呼び出し時に
#       一度だけ評価する。Generate Forever ループ中に text を変更しても、
#       ジェネレータ内の引数は最初の値のまま変わらない。
#       Live Update ON の場合のみ、change イベントでセッション変数を更新し、
#       ジェネレータのループ内でその最新値を参照する。
#       参照音声版では text のみが対象（uploaded_audio はファイルのため対象外）。
# --------------------------------------------------------------------------- #
_session_live_update_flags: dict[str, bool] = {}
_session_live_text: dict[str, str] = {}

# 生成した wav を保存するディレクトリ
# Why: gradio_gen.py と同じ出力先（my/data/outputs）に統一する。
#      参照音声版と VoiceDesign 版の生成結果を同じ場所に集約し、
#      Streamlit の履歴閲覧 UI から一覧で見られるようにするため。
_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "outputs"

# --------------------------------------------------------------------------- #
#  直近の生成設定を保存・読み込みする機構
#
#  Why: UI起動時に前回の設定（テキストやサンプリングパラメータなど）を
#       復元し、ユーザーが毎回入力し直す手間を省くため。
#       gradio_gen.py とは別のファイルに保存する（パラメータ体系が異なるため）。
# --------------------------------------------------------------------------- #
_SETTINGS_PATH = Path(__file__).resolve().parent / "data" / "last_settings_ref.json"


def load_last_settings() -> dict:
    """前回の生成設定をJSONファイルから読み込む。ファイルがなければ空dictを返す。"""
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[my-ref] Failed to load settings: {e}")
    return {}


def save_last_settings(settings: dict) -> None:
    """生成設定をJSONファイルに保存する。"""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[my-ref] Failed to save settings: {e}")


# --------------------------------------------------------------------------- #
#  Generate Forever 稼働中に表示するアニメーションスピナーの HTML/CSS
#
#  Why: Generate Forever が動いていることをユーザーに視覚的に伝えるため。
#       CSSアニメーション（回転）で動くスピナーを生成ボタンの近くに表示する。
#       Gradio の gr.HTML で注入し、visible の切り替えで表示/非表示を制御する。
# --------------------------------------------------------------------------- #
_FOREVER_SPINNER_HTML = """
<div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
    <div style="
        width: 20px;
        height: 20px;
        border: 3px solid rgba(124, 58, 237, 0.2);
        border-top-color: #7c3aed;
        border-radius: 50%;
        animation: forever-spin 0.8s linear infinite;
    "></div>
    <span style="
        font-size: 14px;
        font-weight: 600;
        color: #7c3aed;
    ">Generate Forever 実行中…</span>
</div>
<style>
    @keyframes forever-spin {
        to { transform: rotate(360deg); }
    }
</style>
"""

# --------------------------------------------------------------------------- #
#  キュー再生用カスタム JavaScript
#
#  Why: gr.Audio は HTML5 の <audio> 要素の ended イベントを Python 側に
#       通知する仕組みを持っていない。そのため、再生中に新しい音声が来たとき
#       「現在の再生を中断せずキューに積み、再生完了後に次を再生する」という
#       キュー再生を実現するには、ブラウザ側の JavaScript で制御する必要がある。
#       また Gradio の UI は頻繁に DOM や src を書き換えるため、それとは独立した
#       専用の <audio> プレイヤーを用いることで安定した再生を実現する。
# --------------------------------------------------------------------------- #
_QUEUE_PLAYBACK_JS = f"""
<script>
(function() {{
    'use strict';

    window._queueAudioList = [];
    window._isQueuePlaying = false;
    window._queueForcePaused = false;
    window._currentPlayingItem = null; // 現在再生中のアイテムを保持
    const MAX_QUEUE_SIZE = {_MAX_QUEUE_SIZE};

    /**
     * Python 側から発行された新しいオーディオURLをキューに追加する。
     */
    window.enqueueAudio = function(url) {{
        if (!url) return;

        let name = "Audio";
        try {{
            // URLデコードし、スラッシュ(/)とバックスラッシュ(\\)の両方で分割して最後の部分を取得する
            let decodedUrl = decodeURIComponent(url.split("?")[0]);
            // バックスラッシュをスラッシュに変換してから分割する
            let normalizedUrl = decodedUrl.split("\\\\").join("/");
            let parts = normalizedUrl.split("/");
            name = parts[parts.length - 1] || "Audio";
        }} catch(e) {{}}

        // 新しいものを先頭に追加 (生成順の降順)
        window._queueAudioList.unshift({{ url: url, name: name }});
        console.log('[queue-playback] queued:', name, 'size:', window._queueAudioList.length);
        window.updateQueueUI();

        // If not currently playing and not forcibly paused by user, start now
        if (!window._isQueuePlaying && !window._queueForcePaused) {{
            window.playNextInQueue();
        }}
    }};

    /**
     * キューリストから指定のアイテムを再生する。
     */
    window.playSpecificQueueItem = function(index) {{
        const audioEl = document.getElementById('queue-audio');
        if (!audioEl) return;

        if (index >= 0 && index < window._queueAudioList.length) {{
            let item = window._queueAudioList.splice(index, 1)[0];

            window._isQueuePlaying = true;
            window._queueForcePaused = false;
            window._currentPlayingItem = item; // 現在の再生アイテムに設定
            audioEl.src = item.url;
            audioEl.play().catch(function(e) {{
                console.warn('[queue-playback] play() blocked:', e.message);
                window._isQueuePlaying = false;
                window._currentPlayingItem = null;
                window.updateQueueUI();
            }});
            console.log('[queue-playback] playing specific:', item.name);
        }}
        window.updateQueueUI();
    }};

    /**
     * 次のキューを再生する。
     */
    window.playNextInQueue = function() {{
        const audioEl = document.getElementById('queue-audio');
        if (!audioEl) return;

        if (window._queueAudioList.length > 0) {{
            // unshift で先頭に追加しているため、一番古いものは末尾(pop)
            let item = window._queueAudioList.pop();

            // max queue size limit
            while (window._queueAudioList.length > MAX_QUEUE_SIZE) {{
                window._queueAudioList.pop(); // 古いものから捨てる
            }}

            window._isQueuePlaying = true;
            window._queueForcePaused = false;
            window._currentPlayingItem = item; // 現在の再生アイテムに設定
            audioEl.src = item.url;
            audioEl.play().catch(function(e) {{
                console.warn('[queue-playback] play() blocked:', e.message);
                window._isQueuePlaying = false;
                window._currentPlayingItem = null;
                window.updateQueueUI();
            }});
            console.log('[queue-playback] playing next:', item.name);
        }} else {{
            window._isQueuePlaying = false;
            window._currentPlayingItem = null; // キューが空になったらクリア
            console.log('[queue-playback] queue empty, idle');
        }}
        window.updateQueueUI();
    }};

    /**
     * キューをすべてクリアする
     */
    window.clearQueue = function() {{
        window._queueAudioList = [];
        window._isQueuePlaying = false;
        window._currentPlayingItem = null;
        window._queueForcePaused = false;

        const audioEl = document.getElementById('queue-audio');
        if (audioEl) {{
            audioEl.pause();
            audioEl.removeAttribute('src'); // src=" " だとリロードが走るブラウザがあるため
        }}

        console.log('[queue-playback] queue cleared');
        window.updateQueueUI();
    }};

    /**
     * キューのUI状態を更新する。
     */
    window.updateQueueUI = function() {{
        const container = document.getElementById('queue-player-container');
        const badge = document.getElementById('queue-count-badge');
        const queueListContainer = document.getElementById('queue-list-container');
        const queueUl = document.getElementById('queue-list');
        const currentPlayingInfo = document.getElementById('current-playing-info');
        const currentPlayingName = document.getElementById('current-playing-name');

        if (!container || !badge) return;

        if (window._isQueuePlaying || window._queueForcePaused || window._queueAudioList.length > 0 || window._currentPlayingItem) {{
            container.style.display = 'block';
        }} else {{
            container.style.display = 'none';
        }}

        // 再生中のファイル名表示を更新
        if (window._currentPlayingItem && currentPlayingInfo && currentPlayingName) {{
            currentPlayingInfo.style.display = 'block';
            currentPlayingName.innerText = window._currentPlayingItem.name;
        }} else if (currentPlayingInfo) {{
            currentPlayingInfo.style.display = 'none';
        }}

        if (window._queueAudioList.length > 0) {{
            badge.style.display = 'inline-block';
            badge.innerText = window._queueAudioList.length;

            if (queueListContainer && queueUl) {{
                queueListContainer.style.display = 'block';
                queueUl.innerHTML = '';

                window._queueAudioList.forEach(function(item, index) {{
                    let li = document.createElement('li');
                    li.style.padding = '5px 8px';
                    li.style.borderBottom = '1px solid var(--border-color-primary, #e5e7eb)';
                    li.style.display = 'flex';
                    li.style.justifyContent = 'space-between';
                    li.style.alignItems = 'center';
                    li.style.transition = 'background-color 0.2s';
                    li.onmouseover = function() {{ this.style.backgroundColor = 'var(--background-fill-secondary, #f3f4f6)'; }};
                    li.onmouseout = function() {{ this.style.backgroundColor = 'transparent'; }};

                    let nameSpan = document.createElement('span');
                    nameSpan.innerText = item.name;
                    nameSpan.style.wordBreak = 'break-all';
                    nameSpan.style.marginRight = '10px';

                    let playBtn = document.createElement('button');
                    playBtn.innerText = '再生';
                    playBtn.style.padding = '4px 12px';
                    playBtn.style.backgroundColor = '#7c3aed';
                    playBtn.style.color = 'white';
                    playBtn.style.border = 'none';
                    playBtn.style.borderRadius = '4px';
                    playBtn.style.cursor = 'pointer';
                    playBtn.style.fontSize = '12px';
                    playBtn.style.fontWeight = 'bold';
                    playBtn.style.whiteSpace = 'nowrap';
                    playBtn.onclick = function() {{
                        window.playSpecificQueueItem(index);
                    }};

                    li.appendChild(nameSpan);
                    li.appendChild(playBtn);
                    queueUl.appendChild(li);
                }});
            }}
        }} else {{
            badge.style.display = 'none';
            if (queueListContainer) queueListContainer.style.display = 'none';
            if (queueUl) queueUl.innerHTML = '';
        }}
    }};
}})()
</script>

<script>
// --------------------------------------------------------------------------- //
//  全 <audio> 要素の初期音量を管理し、ユーザーの音量変更を記憶する仕組み
//
//  Why: Gradio の gr.Audio コンポーネントには音量を制御するパラメータがない。
//       HTML5 の <audio> 要素は JavaScript で volume プロパティを設定すれば
//       音量を変えられるが、Gradio は動的に <audio> 要素を生成・差し替えるため、
//       MutationObserver で新しい <audio> が追加されるたびに音量を設定する必要がある。
//       さらに、ユーザーが音量スライダーで変更した値をグローバル変数に記憶し、
//       次に生成される <audio> 要素にもその音量を引き継ぐ。
// --------------------------------------------------------------------------- //
(function() {{
    'use strict';

    // 初期音量（0.0〜1.0）。0.3 = 30%
    // ユーザーが音量を変更するとこの値が更新される
    window._currentVolume = 0.3;

    /**
     * 指定の <audio> 要素に音量を設定し、音量変更のリスナーを登録する。
     * すでに設定済みのものには _volumeInitialized フラグで二重設定を防ぐ。
     */
    function initAudioVolume(audioEl) {{
        if (audioEl && !audioEl._volumeInitialized) {{
            // グローバル変数に記憶された音量を適用
            audioEl.volume = window._currentVolume;
            audioEl._volumeInitialized = true;

            // ユーザーが音量スライダーを操作したらグローバル変数を更新
            // Why: volumechange イベントでユーザーが設定した音量を記憶し、
            //      次に生成される <audio> 要素にも同じ音量を引き継ぐ。
            audioEl.addEventListener('volumechange', function() {{
                window._currentVolume = audioEl.volume;
            }});

            console.log('[volume-init] set volume to', window._currentVolume, 'for', audioEl.id || 'unnamed audio');
        }}
    }}

    // ページ読み込み時に既存の <audio> 要素すべてに音量を設定
    document.addEventListener('DOMContentLoaded', function() {{
        document.querySelectorAll('audio').forEach(initAudioVolume);
    }});

    // MutationObserver: DOM に新しい <audio> が追加されるたびに音量を設定
    // Why: Gradio は生成のたびに <audio> 要素を動的に差し替えるため、
    //      初回ロード時だけ設定しても不十分。変更を監視して常に音量を設定する。
    const observer = new MutationObserver(function(mutations) {{
        mutations.forEach(function(mutation) {{
            mutation.addedNodes.forEach(function(node) {{
                if (node.nodeType === 1) {{
                    // 追加されたノード自体が <audio> の場合
                    if (node.tagName === 'AUDIO') {{
                        initAudioVolume(node);
                    }}
                    // 追加されたノードの子孫に <audio> がある場合
                    node.querySelectorAll && node.querySelectorAll('audio').forEach(initAudioVolume);
                }}
            }});
        }});
    }});
    observer.observe(document.body || document.documentElement, {{
        childList: true,
        subtree: true
    }});
}})()
</script>
"""


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text: str,
    uploaded_audio: str | None,
    uploaded_speaker_embedding: object,
    speaker_embedding_path_raw: str,
    num_steps: int,
    seed_raw: str,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    # Why: v3 で追加された新サンプリングパラメータを UI から受け取るため、引数を追加する。
    seconds_raw: str,
    duration_scale: float,
    t_schedule_mode: str,
    sway_coeff: float,
    lora_adapter_raw: str,
    autoplay: bool,
    forever: bool,
    history_paths: list[str],
    request: gr.Request = None,
) -> Generator[tuple, None, None]:
    """
    音声を1件生成し、履歴・DB を更新する（参照音声版）。

    forever=True の場合はジェネレータとして繰り返し生成を行い、
    Gradio 側で yield ごとにUIが更新される。
    forever=False の場合でもジェネレータとして1回だけ yield する
    （Gradio の .click() で統一的に扱うため）。

    gradio_gen.py の _run_generation との主な違い:
    - caption の代わりに uploaded_audio（参照音声）を使用
    - cfg_scale_caption の代わりに cfg_scale_speaker を使用
    - speaker_kv_scale / speaker_kv_min_t / speaker_kv_max_layers パラメータあり
    - max_text_len / max_caption_len パラメータなし
    - caption conditioning チェックなし（参照音声モデルのため）

    Args:
        checkpoint〜speaker_kv_max_layers_raw: サンプリング関連パラメータ
        seconds_raw〜lora_adapter_raw: v3 新サンプリングパラメータ
        autoplay:      最新音声を自動再生するかどうか（初回値、以降はセッション変数を参照）
        forever:       連続生成モードかどうか（ボタンに応じて固定値が渡される）
        history_paths: 直近5件の wav パスリスト（gr.State から受け取る）
        request:       Gradio のリクエスト情報（セッション管理用）

    Yields:
        tuple: (Audio×5 の更新, ログ文字列, タイミング文字列,
                更新後の history_paths, スピナー表示状態, キュー用ファイルパス)
    """

    def stdout_log(msg: str) -> None:
        """標準出力にログを書き出すヘルパー"""
        print(msg, flush=True)

    # --- パラメータのバリデーションとパース ---
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )

    text_value = str(text).strip()

    if text_value == "":
        raise ValueError("text（テキスト）は必須です。")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    # Why: v3 で追加されたオプショナルパラメータのパースを行う。
    manual_seconds = _parse_optional_float(seconds_raw, "seconds")
    lora_adapter = _parse_optional_str(lora_adapter_raw)

    # 参照音声の解決
    # Why: uploaded_audio が None や空文字の場合は no-reference モードで推論する。
    #      参照音声がある場合はそのパスを使い、話者の声質を真似て生成する。
    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)

    # Speaker Embedding の解決
    # Why: 本家で定義されている _resolve_speaker_embedding を用いて、
    #      アップロードされたファイルまたは直接入力されたパスから embedding ファイルを解決する。
    speaker_embedding = _resolve_speaker_embedding(
        uploaded_embedding=uploaded_speaker_embedding,
        speaker_embedding_path_raw=speaker_embedding_path_raw,
    )

    # 排他チェック
    # Why: 参照音声(ref_wav)と Speaker Embedding(safetensors) は排他仕様であり、
    #      同時に指定された場合はエラーにする。どちらも未指定の場合は no_ref モードとなる。
    if ref_wav is not None and speaker_embedding is not None:
        raise ValueError("参照音声と Speaker Embedding は同時に指定できません。")

    no_ref = ref_wav is None and speaker_embedding is None

    # 出力ディレクトリを確保
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # DB を初期化（テーブルが無ければ作る）
    init_db()

    # セッションIDを取得・フラグをリセット
    session_id = request.session_hash if request else "default"
    _session_stop_flags[session_id] = False

    # autoplay の初期値をセッション変数にも反映
    # Why: ジェネレータ起動前のチェックボックス状態を保持する
    _session_autoplay_flags[session_id] = autoplay

    # Live Update 用にテキストの初期値をセッション変数に保存
    # Why: Live Update ON 時にループ内で最新値を参照するための初期値
    _session_live_text[session_id] = text_value

    # forever モード開始時にスピナーを表示
    if forever:
        stdout_log(f"[my-ref] Generate Forever started (session: {session_id})")

    # --- 直近の生成設定を保存 ---
    settings_to_save = {
        "checkpoint": str(checkpoint),
        "model_device": str(model_device),
        "model_precision": str(model_precision),
        "codec_device": str(codec_device),
        "codec_precision": str(codec_precision),
        "text": text_value,
        "speaker_embedding_path_raw": str(speaker_embedding_path_raw),
        "num_steps": int(num_steps) if str(num_steps).isdigit() else 40,
        "seed_raw": str(seed_raw),
        "cfg_guidance_mode": str(cfg_guidance_mode),
        "cfg_scale_text": float(cfg_scale_text),
        "cfg_scale_speaker": float(cfg_scale_speaker),
        "cfg_scale_raw": str(cfg_scale_raw),
        "cfg_min_t": float(cfg_min_t),
        "cfg_max_t": float(cfg_max_t),
        "context_kv_cache": bool(context_kv_cache),
        "truncation_factor_raw": str(truncation_factor_raw),
        "rescale_k_raw": str(rescale_k_raw),
        "rescale_sigma_raw": str(rescale_sigma_raw),
        "speaker_kv_scale_raw": str(speaker_kv_scale_raw),
        "speaker_kv_min_t_raw": str(speaker_kv_min_t_raw),
        "speaker_kv_max_layers_raw": str(speaker_kv_max_layers_raw),
        "seconds_raw": str(seconds_raw),
        "duration_scale": float(duration_scale),
        "t_schedule_mode": str(t_schedule_mode),
        "sway_coeff": float(sway_coeff),
        "lora_adapter_raw": str(lora_adapter_raw),
        "autoplay": bool(autoplay),
    }
    save_last_settings(settings_to_save)

    # --- 生成ループ（forever=False なら1回で終了） ---
    # Why: while True + break で「最低1回は実行」を保証しつつ、
    #      forever フラグで連続生成に対応する
    iteration = 0
    while True:
        # 停止要求があれば生成開始前にBreak
        if _session_stop_flags.get(session_id, False):
            stdout_log(f"[my-ref] Generation stopped by user (session: {session_id})")
            break

        iteration += 1

        runtime, reloaded = get_cached_runtime(runtime_key)

        stdout_log(
            f"[my-ref] runtime: {'reloaded' if reloaded else 'reused'} (iteration {iteration})"
        )

        # --- Live Update: プロンプトの最新値を取得 ---
        # Why: Generate Forever ループ中にユーザーが text を変更した場合、
        #      Live Update ON ならセッション変数から最新値を使用する。
        #      OFF の場合は Generate Forever 開始時の値がそのまま使われる。
        #      参照音声版では text のみが対象（uploaded_audio はファイルのため対象外）。
        if forever and _session_live_update_flags.get(session_id, False):
            live_text = _session_live_text.get(session_id, text_value)
            if live_text != text_value:
                stdout_log(
                    f"[my-ref] Live Update: text を最新値に更新 (iteration {iteration})"
                )
                text_value = live_text

        result = runtime.synthesize(
            SamplingRequest(
                text=text_value,
                ref_wav=ref_wav,
                ref_latent=None,
                ref_embed=speaker_embedding,
                no_ref=bool(no_ref),
                ref_normalize_db=-16.0,
                ref_ensure_max=True,
                num_candidates=1,  # 常に1件だけ生成（候補グリッド廃止）
                decode_mode="sequential",
                # Why: 秒数を自動予測 (Duration Predictor) もしくは手動指定 (manual_seconds) に対応させる。
                seconds=manual_seconds,
                duration_scale=float(duration_scale),
                max_ref_seconds=30.0,
                max_text_len=None,
                num_steps=int(num_steps),
                seed=None if seed is None else int(seed),
                cfg_guidance_mode=str(cfg_guidance_mode),
                cfg_scale_text=float(cfg_scale_text),
                cfg_scale_speaker=float(cfg_scale_speaker),
                cfg_scale=cfg_scale,
                cfg_min_t=float(cfg_min_t),
                cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers,
                # Why: 時間スケジュールモード、その係数、LoRAアコーディオンの指定値を追加する。
                t_schedule_mode=str(t_schedule_mode),
                sway_coeff=float(sway_coeff),
                lora_adapter=lora_adapter,
                trim_tail=True,
            ),
            log_fn=stdout_log,
        )

        # --- ファイル保存 ---
        # ファイル名規則: {YYYYMMDD_HHMMSS}_{seed}.wav
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        used_seed = result.used_seed
        out_filename = f"{stamp}_{used_seed}.wav"
        out_path = save_wav(
            _OUTPUT_DIR / out_filename,
            result.audios[0].float(),
            result.sample_rate,
        )
        out_path_str = str(out_path)

        stdout_log(f"[my-ref] saved: {out_path_str}")

        # --- DB に書き込み ---
        # Why: 参照音声版では caption（スタイルプロンプト）を使わないが、
        #      caption カラムに参照音声のファイル名を記録しておくことで、
        #      DB 上でどの参照音声を使って生成したかを追跡できるようにする。
        #      参照音声がない場合は "none" を記録する。
        ref_caption = Path(ref_wav).name if ref_wav else "none"
        insert_generation(
            text=text_value,
            caption=ref_caption,  # 参照音声のファイル名、なければ "none"
            seed=used_seed,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=None,  # 参照音声版では cfg_scale_speaker を使うが DB スキーマに列がないため省略
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_guidance_mode=str(cfg_guidance_mode),
            checkpoint=str(checkpoint).strip(),
            file_path=out_path_str,
            filename=out_filename,
            duration_scale=locals().get("duration_scale"),
            seconds=locals().get("manual_seconds"),
            t_schedule_mode=locals().get("t_schedule_mode"),
            sway_coeff=locals().get("sway_coeff"),
            lora_adapter=locals().get("lora_adapter"),
            speaker_embedding=locals().get("speaker_embedding"),
            ui_version=MY_UI_VERSION,
            model_version=guess_model_version(checkpoint),
        )
        stdout_log("[my-ref] DB insert 完了")

        # --- 履歴を更新 ---
        # 先頭に追加して5件に切り詰める
        history_paths = [out_path_str] + history_paths[: _MAX_HISTORY - 1]

        # --- ログ文字列を組み立て ---
        runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
        detail_lines = [
            runtime_msg,
            f"iteration: {iteration}",
            f"seed_used: {used_seed}",
            f"ref_audio: {'あり' if ref_wav else 'なし (no-reference)'}",
            f"saved: {out_path_str}",
            *result.messages,
        ]
        detail_text = "\n".join(detail_lines)
        timing_text = _format_timings(result.stage_timings, result.total_to_decode)

        # --- autoplay をリアルタイムに参照 ---
        # Why: Generate Forever ループ中に autoplay チェックボックスが変更された場合、
        #      セッション変数から最新の値を取得する。
        current_autoplay = _session_autoplay_flags.get(session_id, autoplay)

        # キュー再生用: Autoplay ON ならファイルパスを、OFF なら None を返す
        new_queue_path = out_path_str if current_autoplay else None

        # --- Audio×5 の更新を組み立て ---
        # Why: キュー再生用に独立したオーディオプレイヤーが全てを処理するため、
        #      履歴一覧の Audio コンポーネントは autoplay=False で固定する。
        #      これにより、Gradio 側が勝手に再生を開始して再生音が重複するのを防ぐ。
        audio_updates: list[gr.Audio] = []
        for i in range(_MAX_HISTORY):
            if i < len(history_paths):
                audio_updates.append(
                    gr.Audio(
                        value=history_paths[i],
                        visible=True,
                        autoplay=False,
                    )
                )
            else:
                audio_updates.append(gr.Audio(value=None, visible=False))

        # スピナーの表示状態を決定
        spinner_visible = forever

        yield (
            *audio_updates,
            detail_text,
            timing_text,
            history_paths,
            gr.update(visible=spinner_visible),
            new_queue_path,
        )

        # forever=False なら1回で終了
        if not forever:
            break

        # 停止要求があれば1回出力した後でもBreak
        if _session_stop_flags.get(session_id, False):
            stdout_log(f"[my-ref] Generation stopped by user after yield (session: {session_id})")
            break

        # forever モード時は seed=None にしてランダム化
        # Why: 同じ seed で繰り返しても同じ音声が生成されるだけなので、
        #      連続生成時はランダムシードに切り替える
        seed = None

    # --- ループ終了後: スピナーを確実に非表示にする ---
    # Why: 停止ボタンやエラーでループを抜けた場合でも、スピナーを非表示にする。
    #      最後の yield でスピナーを非表示にすることで、UIが正しい状態に戻る。
    if forever:
        stdout_log(f"[my-ref] Generate Forever ended (session: {session_id})")
        # 最後に Audio は変更せず、スピナーだけ非表示にする更新を yield
        # 各 Audio は現在の状態を維持（gr.update() で何も変えない）
        no_change_audios = [gr.update() for _ in range(_MAX_HISTORY)]
        yield (
            *no_change_audios,
            "Generate Forever が終了しました。",
            gr.update(),
            history_paths,
            gr.update(visible=False),
            None,
        )


def build_ui() -> gr.Blocks:
    """
    Gradio UI を構築して返す（参照音声版）。

    本家 gradio_app.py build_ui() との主な差分:
    - 候補グリッド（32枠）を廃止し、直近5件の履歴表示に置き換え
    - autoplay トグルを追加
    - Generate Forever / Cancel Forever ボタンで連続生成を制御
    - Generate Forever 中にスピナーアイコンを表示
    - num_candidates スライダーを削除

    gradio_gen.py build_ui() との主な差分:
    - caption テキストボックスの代わりに uploaded_audio (参照音声) を表示
    - cfg_scale_caption の代わりに cfg_scale_speaker を表示
    - speaker_kv_* パラメータのAdvanced設定がある
    - max_text_len / max_caption_len がない
    """
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    # 前回の設定を読み込む
    last_settings = load_last_settings()

    # 必須の選択肢を検証しながら復元
    saved_model_device = last_settings.get("model_device", default_model_device)
    if saved_model_device not in device_choices:
        saved_model_device = default_model_device

    saved_codec_device = last_settings.get("codec_device", default_codec_device)
    if saved_codec_device not in device_choices:
        saved_codec_device = default_codec_device

    # head: HTML の <head> セクションに注入するカスタム JavaScript
    # キュー再生用の JS をここで読み込む
    with gr.Blocks(title="Irodori-TTS My Reference", head=_QUEUE_PLAYBACK_JS) as demo:
        gr.Markdown("# Irodori-TTS 独自生成UI（参照音声版）")
        gr.Markdown(
            "参照音声モデル向けの独自UIです。"
            "Reference Audio を指定するとその声質を真似た音声を生成し、"
            "空欄なら no-reference モードで推論します。"
        )

        # --- モデル設定行 ---
        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=last_settings.get("checkpoint", default_checkpoint),
                scale=4,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=saved_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=last_settings.get("model_precision", model_precision_choices[0]),
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=saved_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=last_settings.get("codec_precision", codec_precision_choices[0]),
                scale=1,
            )
            # ウォーターマークは常にOFF（gr.State で非表示管理）
            enable_watermark = gr.State(False)

        # --- モデル読み込み/解放ボタン ---
        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        # --- テキスト入力 ---
        text = gr.Textbox(label="Text", lines=4, value=last_settings.get("text", ""))

        # Why: ユーザーが特定の感情表現や音響効果（囁き、笑いなど）を表す絵文字を
        #      クリックだけでテキストのカーソル位置に挿入できるようにするため、
        #      テキスト入力エリアの直下に絵文字パレットを配置する。初期状態は閉じた状態(open=False)とする。
        build_emoji_palette(text, open=False)

        # --- 参照音声 / Speaker Embedding 入力欄 ---
        # Why: 参照音声アップロードと Speaker Embedding は排他仕様（どちらか一方のみ指定）である。
        #      左右の Column に並べて表示することで、ユーザーがそれぞれの状態を視覚的に
        #      比較・確認しやすくする。
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    uploaded_audio_file = gr.File(
                        label="Reference Audio Upload (ここに音声ファイルをドラッグ&ドロップ)",
                        type="filepath",
                        file_types=["audio"],
                        file_count="single",
                    )
                
                with gr.Row():
                    ref_filename_display = gr.Textbox(
                        label="現在のリファレンス音声",
                        value="(no-reference mode)",
                        interactive=False,
                        scale=3
                    )
                    clear_ref_btn = gr.Button("クリア (No Reference)", variant="stop", scale=1)
                
                uploaded_audio_player = gr.Audio(
                    label="Reference Audio Playback",
                    type="filepath",
                    interactive=False,
                )

            with gr.Column():
                uploaded_speaker_embedding = gr.File(
                    label="Speaker Embedding (.speaker.safetensors) Upload",
                    type="filepath",
                    file_types=[".safetensors"],
                    file_count="single",
                )
                speaker_embedding_path_raw = gr.Textbox(
                    label="Speaker Embedding Path (optional, alt to upload)",
                    value=last_settings.get("speaker_embedding_path_raw", ""),
                )
            
        # 内部パス保持用の State コンポーネント。この State が _make_inputs で参照される。
        uploaded_audio = gr.State(None)

        def _handle_upload(new_file, current_state):
            # new_file にパスが入っている（ファイルがドロップされた）場合
            if new_file is not None:
                from pathlib import Path
                name = Path(new_file).name
                # Fileコンポーネントには None を返して常に空のドロップゾーンを維持する
                return None, new_file, name, new_file
            
            # Fileコンポーネントが空になった場合は現在の状態を維持
            if current_state is not None:
                from pathlib import Path
                return None, current_state, Path(current_state).name, current_state
                
            return None, None, "(no-reference mode)", None
            
        def _clear_ref():
            return None, "(no-reference mode)", None

        uploaded_audio_file.change(
            fn=_handle_upload,
            inputs=[uploaded_audio_file, uploaded_audio],
            outputs=[uploaded_audio_file, uploaded_audio, ref_filename_display, uploaded_audio_player]
        )

        clear_ref_btn.click(
            fn=_clear_ref,
            inputs=[],
            outputs=[uploaded_audio, ref_filename_display, uploaded_audio_player]
        )
        # --- サンプリング設定 ---
        # Why: num_candidates は常に1なのでスライダーを削除
        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Slider(
                    label="Num Steps",
                    minimum=1,
                    maximum=120,
                    value=last_settings.get("num_steps", 40),
                    step=1,
                )
                seed_raw = gr.Textbox(
                    label="Seed (blank=random)", value=last_settings.get("seed_raw", "")
                )
                # Why: v3 の Duration Predictor に合わせ、明示的な秒数指定および話速（Duration Scale）を調整可能にする。
                seconds_raw = gr.Textbox(
                    label="Seconds (blank=auto)", value=last_settings.get("seconds_raw", "")
                )
                duration_scale = gr.Slider(
                    label="Duration Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=last_settings.get("duration_scale", 1.0),
                    step=0.01,
                )

            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule",
                    choices=["linear", "sway"],
                    value=last_settings.get("t_schedule_mode", "linear"),
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff",
                    minimum=-1.0,
                    maximum=1.5,
                    value=last_settings.get("sway_coeff", -1.0),
                    step=0.1,
                    interactive=False, # Time Schedule で sway を選択した時のみ有効化
                )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value=last_settings.get("cfg_guidance_mode", "independent"),
                )
                # Why: 参照音声版では cfg_scale_text のデフォルトが 3.0（VoiceDesign版は 2.0）
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=last_settings.get("cfg_scale_text", 3.0),
                    step=0.1,
                )
                # Why: VoiceDesign版の cfg_scale_caption に相当するが、
                #      参照音声版では話者(Speaker)の類似度を制御するため名前が異なる。
                #      デフォルト値は 5.0（VoiceDesign版の caption は 4.0）。
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=last_settings.get("cfg_scale_speaker", 5.0),
                    step=0.1,
                )

        # --- Advanced 設定 ---
        # Why: 参照音声版には speaker_kv_* パラメータがある一方、
        #      max_text_len / max_caption_len はない。
        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(
                label="CFG Scale Override (optional)", value=last_settings.get("cfg_scale_raw", "")
            )
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=last_settings.get("cfg_min_t", 0.5))
                cfg_max_t = gr.Number(label="CFG Max t", value=last_settings.get("cfg_max_t", 1.0))
                context_kv_cache = gr.Checkbox(
                    label="Context KV Cache", value=last_settings.get("context_kv_cache", True)
                )
            with gr.Row():
                truncation_factor_raw = gr.Textbox(
                    label="Truncation Factor (optional)",
                    value=last_settings.get("truncation_factor_raw", ""),
                )
                rescale_k_raw = gr.Textbox(
                    label="Rescale k (optional)", value=last_settings.get("rescale_k_raw", "")
                )
                rescale_sigma_raw = gr.Textbox(
                    label="Rescale sigma (optional)",
                    value=last_settings.get("rescale_sigma_raw", ""),
                )
            # --- Speaker KV 関連パラメータ ---
            # Why: 参照音声から抽出した話者特徴量の注入強度を微調整するための
            #      Advanced パラメータ。VoiceDesign版にはないが、参照音声版には必要。
            #      speaker_kv_scale: 話者KVの注入スケール（省略時はモデルデフォルト）
            #      speaker_kv_min_t: 話者KVを注入する最小ステップ比率（通常0.9程度）
            #      speaker_kv_max_layers: 話者KVを注入するTransformer層数の上限
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(
                    label="Speaker KV Scale (optional)",
                    value=last_settings.get("speaker_kv_scale_raw", ""),
                )
                speaker_kv_min_t_raw = gr.Textbox(
                    label="Speaker KV Min t (optional)",
                    value=last_settings.get("speaker_kv_min_t_raw", "0.9"),
                )
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)",
                    value=last_settings.get("speaker_kv_max_layers_raw", ""),
                )
            # Why: 学習済みの追加話者スタイル (LoRA) を適用可能にするため、アダプタディレクトリのパス入力欄を設ける。
            lora_adapter_raw = gr.Textbox(
                label="LoRA Adapter Directory (optional)",
                value=last_settings.get("lora_adapter_raw", ""),
            )

        # --- 生成制御 ---
        # Why: EasyReforge 風に Generate / Generate Forever / Cancel Forever の
        #      ボタン3つで制御する。旧 Forever チェックボックス + Stop ボタンを廃止。
        #      Generate Forever 中はスピナーアイコンを表示して稼働中であることを示す。
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", scale=3)
            generate_forever_btn = gr.Button("Generate Forever", variant="secondary", scale=2)
            # autoplay: 最新の1件を自動再生するかの切り替え
            autoplay = gr.Checkbox(
                label="Autoplay", value=last_settings.get("autoplay", True), scale=1
            )
            # Live Update: Generate Forever 中にプロンプトの変更を反映するかどうか
            # Why: デフォルト OFF にすることで、意図しないプロンプト変更が
            #      連続生成に影響しないようにする。ON にすると、text の
            #      変更が次のイテレーションから反映される。
            live_update = gr.Checkbox(
                label="Live Update", value=False, scale=1
            )
            cancel_forever_btn = gr.Button("Cancel Forever", variant="stop", scale=1)

        # --- Generate Forever 稼働中のスピナー表示 ---
        # Why: Generate Forever が動いていることを視覚的に分かりやすくするため、
        #      生成ボタンの近くにアニメーション付きスピナーを配置する。
        #      初期状態では非表示で、Generate Forever 開始時に visible=True にする。
        forever_spinner = gr.HTML(
            value=_FOREVER_SPINNER_HTML,
            visible=False,
        )

        # --- forever フラグ用の固定値 State ---
        # Why: Generate ボタンと Generate Forever ボタンで同じ _run_generation 関数を
        #      使い回すため、forever パラメータを gr.State の固定値として渡す。
        #      Generate ボタン → forever=False（1回生成）
        #      Generate Forever ボタン → forever=True（連続生成）
        forever_false = gr.State(False)
        forever_true = gr.State(True)

        # --- キュー用 UI ---
        # Why: 独立した再生プレイヤーでキューを管理するため、コンテナごと表示する。
        #      直近の生成結果より上に配置することで、ユーザーの目に留まりやすくする。
        gr.HTML("""
        <div id="queue-player-container" style="display:none; padding: 10px; background: var(--background-fill-secondary, #f9fafb); border-radius: 8px; border: 1px solid var(--border-color-primary, #e5e7eb); margin-bottom: 15px;">
            <div style="font-size:14px; font-weight:bold; margin-bottom:10px; color: var(--body-text-color, #374151); display: flex; justify-content: space-between; align-items: center;">
                <div>▶️ 連続再生プレイヤー <span id="queue-count-badge" style="background:#7c3aed; color:white; border-radius:10px; padding:2px 8px; font-size:12px; margin-left:5px; display:none;">0</span></div>
                <button onclick="if(window.clearQueue) window.clearQueue()" style="font-size:12px; padding: 4px 12px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">全削除</button>
            </div>
            <div id="current-playing-info" style="font-size:13px; margin-bottom:8px; display:none; padding: 6px; background-color: var(--background-fill-primary, #ffffff); border-radius: 4px; border-left: 4px solid #7c3aed;">
                <span style="color: var(--body-text-color-sub, #6b7280);">現在再生中:</span>
                <span id="current-playing-name" style="font-weight:bold; color: var(--body-text-color, #374151); word-break: break-all; margin-left: 4px;"></span>
            </div>
            <audio id="queue-audio" controls style="width: 100%; margin-bottom: 10px;"
                onended="if(window.playNextInQueue) window.playNextInQueue()"
                onplay="window._isQueuePlaying = true; window._queueForcePaused = false; if(window.updateQueueUI) window.updateQueueUI()"
                onpause="if(!this.ended){ window._queueForcePaused = true; window._isQueuePlaying = false; if(window.updateQueueUI) window.updateQueueUI(); }"
            ></audio>
            <div id="queue-list-container" style="display:none; max-height: 180px; overflow-y: auto; background: var(--background-fill-primary, #ffffff); border: 1px solid var(--border-color-primary, #d1d5db); border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);">
                <ul id="queue-list" style="list-style: none; padding: 0; margin: 0; font-size: 13px; color: var(--body-text-color, #374151);"></ul>
            </div>
        </div>
        """)

        # --- キュー処理用隠し File ---
        # 生成完了のたびにこのコンポーネントに音声パスが渡され、JS(enqueueAudio)にURL(Token付)が発火する
        queue_new_item = gr.File(visible=False, elem_id="queue-new-item")
        queue_new_item.change(
            fn=None,
            inputs=[queue_new_item],
            js="""
            function(fileObj) {
                if (fileObj && fileObj.url && typeof window.enqueueAudio === 'function') {
                    window.enqueueAudio(fileObj.url);
                }
            }
            """,
        )

        # --- 直近5件の履歴表示 ---
        # Why: 候補グリッド（32枠）を廃止し、直近5件の生成結果を縦に並べて表示する。
        #      最新が上に来るため、生成のたびに新しい音声がすぐ確認できる。
        gr.Markdown("### 直近の生成結果")
        # gr.State: Gradio のセッション内でデータを保持する仕組み
        # ここでは直近5件の wav ファイルパスを配列で管理する
        history_state = gr.State(value=[])

        out_audios: list[gr.Audio] = []
        for i in range(_MAX_HISTORY):
            # elem_id: カスタム JS から DOM 要素を特定するために使用
            # 最新の1件 (i=0) は "audio-0" という ID を付与し、
            # JS のキュー再生ロジックがこの要素を監視する
            out_audios.append(
                gr.Audio(
                    label=f"#{i + 1}",
                    type="filepath",
                    interactive=False,
                    visible=(False),  # 初期状態では非表示（生成後に表示される）
                    elem_id=f"audio-{i}",
                )
            )

        # --- ログ出力 ---
        out_log = gr.Textbox(label="Run Log", lines=6)
        out_timing = gr.Textbox(label="Timing", lines=6)

        # --- 共通の入力リスト ---
        # Why: Generate ボタンと Generate Forever ボタンで共通する入力パラメータを
        #      まとめて定義し、コードの重複を避ける。
        #      forever パラメータだけがボタンごとに異なる。
        def _make_inputs(forever_state: gr.State) -> list:
            """
            ボタンごとに異なる forever State を含む入力リストを生成する。

            Args:
                forever_state: gr.State(False) または gr.State(True)

            Returns:
                list: _run_generation に渡す入力コンポーネントのリスト
            """
            return [
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
                text,
                uploaded_audio,
                uploaded_speaker_embedding,
                speaker_embedding_path_raw,
                num_steps,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
                # Why: v3 新パラメータを _run_generation 引数の順序に合わせて追加する。
                seconds_raw,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                lora_adapter_raw,
                autoplay,
                forever_state,
                history_state,
            ]

        # 出力リスト（Audio×5 + ログ + タイミング + 履歴State + スピナー + 新規キューパス）
        gen_outputs = [
            *out_audios,
            out_log,
            out_timing,
            history_state,
            forever_spinner,
            queue_new_item,
        ]

        # --- イベントバインド ---

        # Generate ボタン: 1回だけ生成（forever=False）
        # Why: _run_generation はジェネレータ関数なので、Gradio は yield ごとに
        #      UI を逐次更新する。forever=False なら1回で終了する。
        generate_btn.click(
            _run_generation,
            inputs=_make_inputs(forever_false),
            outputs=gen_outputs,
        )

        # Generate Forever ボタン: 連続生成（forever=True）
        # Why: forever=True の固定値を渡すことで、ジェネレータが停止要求まで
        #      ループし続ける。スピナーは _run_generation 内で制御される。
        generate_forever_btn.click(
            _run_generation,
            inputs=_make_inputs(forever_true),
            outputs=gen_outputs,
        )

        def _handle_stop(req: gr.Request):
            """
            停止フラグを立てて現在の処理が終わり次第ループを抜ける。
            Cancel Forever ボタンから呼ばれる。
            """
            sid = req.session_hash if req else "default"
            _session_stop_flags[sid] = True
            return "※停止要求を受け付けました。現在の生成が完了すると停止します。"

        # Cancel Forever ボタン: Generate Forever のジェネレータループを中断する
        # Why: queue=False にすることで順次処理キューをバイパスし、即座にフラグを立てる。
        #      cancels=[] を使うと UI 全体がエラー表示になってしまう問題を回避するため。
        cancel_forever_btn.click(fn=_handle_stop, inputs=[], outputs=[out_log], queue=False)

        def _handle_autoplay_change(value: bool, req: gr.Request):
            """
            autoplay チェックボックスの変更をセッション変数に反映する。

            Why: Generate Forever ループ中に autoplay を OFF にした場合、
                 次の生成から autoplay 属性を付与しないようにするため。
                 ジェネレータの inputs は起動時に一度だけ評価されるので、
                 ループ中の変更はこのイベントハンドラ経由でしか取得できない。
            """
            sid = req.session_hash if req else "default"
            _session_autoplay_flags[sid] = value
            stdout_msg = "ON" if value else "OFF"
            print(f"[my-ref] autoplay changed to {stdout_msg} (session: {sid})", flush=True)

        # autoplay チェックボックスの変更イベント
        # Why: queue=False で即座に反映し、Generate Forever ループ中でも
        #      次の生成から autoplay の ON/OFF が反映されるようにする。
        autoplay.change(
            fn=_handle_autoplay_change,
            inputs=[autoplay],
            outputs=[],
            queue=False,
        )

        def _handle_live_update_change(value: bool, req: gr.Request):
            """
            Live Update チェックボックスの変更をセッション変数に反映する。

            Why: Generate Forever ループ中にプロンプト変更を反映するかどうかを
                 リアルタイムに切り替えるため。
            """
            sid = req.session_hash if req else "default"
            _session_live_update_flags[sid] = value
            stdout_msg = "ON" if value else "OFF"
            print(f"[my-ref] live update changed to {stdout_msg} (session: {sid})", flush=True)

        # Live Update チェックボックスの変更イベント
        live_update.change(
            fn=_handle_live_update_change,
            inputs=[live_update],
            outputs=[],
            queue=False,
        )

        def _handle_text_change(value: str, req: gr.Request):
            """
            Text テキストボックスの変更をセッション変数に反映する。

            Why: Live Update ON 時に Generate Forever ループ内から
                 最新のテキストを参照できるようにするため。
            """
            sid = req.session_hash if req else "default"
            _session_live_text[sid] = str(value).strip()

        # text の変更イベント
        # Why: queue=False で即座にセッション変数を更新する。
        #      Live Update OFF でも変数は更新しておき、ON に切り替えた瞬間から
        #      最新値を参照できるようにする。
        text.change(
            fn=_handle_text_change,
            inputs=[text],
            outputs=[],
            queue=False,
        )

        # デバイス変更時に精度選択肢を動的更新
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )

        # Time Schedule 変更時に sway_coeff の有効/無効を切り替える
        # Why: linear のときは sway_coeff を無効 (interactive=False) にし、
        #      sway のときのみ有効 (interactive=True) に制御するため。
        t_schedule_mode.change(
            _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
        )

        # モデル読み込み/解放
        # Why: 参照音声版は _load_model を使用（VoiceDesign版の _describe_runtime とは異なり、
        #      caption conditioning チェックを行わないシンプルな読み込み関数）
        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

        def _load_settings_for_ui():
            """
            ページリロード時に最新の設定JSONを読み込み、各UIコンポーネントの初期値として返す。
            Why: 別タブで変更した設定や、ブラウザのリロード時にも最新の前回設定を反映するため。
            """
            s = load_last_settings()

            # 必須の選択肢を検証しながら復元
            m_device = s.get("model_device", default_model_device)
            if m_device not in device_choices:
                m_device = default_model_device

            c_device = s.get("codec_device", default_codec_device)
            if c_device not in device_choices:
                c_device = default_codec_device

            # Why: リロード時に t_schedule_mode が sway であれば sway_coeff を interactive=True に、
            #      linear であれば interactive=False に設定する。
            t_mode = s.get("t_schedule_mode", "linear")
            is_sway = str(t_mode).strip().lower() == "sway"

            return [
                s.get("checkpoint", default_checkpoint),
                m_device,
                s.get("model_precision", model_precision_choices[0]),
                c_device,
                s.get("codec_precision", codec_precision_choices[0]),
                s.get("text", ""),
                s.get("speaker_embedding_path_raw", ""),
                s.get("num_steps", 40),
                s.get("seed_raw", ""),
                s.get("cfg_guidance_mode", "independent"),
                s.get("cfg_scale_text", 3.0),
                s.get("cfg_scale_speaker", 5.0),
                s.get("cfg_scale_raw", ""),
                s.get("cfg_min_t", 0.5),
                s.get("cfg_max_t", 1.0),
                s.get("context_kv_cache", True),
                s.get("truncation_factor_raw", ""),
                s.get("rescale_k_raw", ""),
                s.get("rescale_sigma_raw", ""),
                s.get("speaker_kv_scale_raw", ""),
                s.get("speaker_kv_min_t_raw", "0.9"),
                s.get("speaker_kv_max_layers_raw", ""),
                # Why: v3 サンプリングパラメータを UI 復元リストに追加する。
                s.get("seconds_raw", ""),
                s.get("duration_scale", 1.0),
                t_mode,
                gr.update(value=s.get("sway_coeff", -1.0), interactive=is_sway),
                s.get("lora_adapter_raw", ""),
                s.get("autoplay", True),
            ]

        # --- UIリロード時の最新設定の読み込み ---
        demo.load(
            fn=_load_settings_for_ui,
            inputs=None,
            outputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                text,
                speaker_embedding_path_raw,
                num_steps,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
                # Why: UI リロード時に復元する v3 新パラメータを outputs リストに追加する。
                seconds_raw,
                duration_scale,
                t_schedule_mode,
                sway_coeff,
                lora_adapter_raw,
                autoplay,
            ],
        )

    return demo


def main() -> None:
    """エントリーポイント: コマンドライン引数を解析して Gradio サーバーを起動"""
    parser = argparse.ArgumentParser(
        description="独自生成UI for Irodori-TTS reference-audio checkpoints."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    # concurrency_limit=1: GPU を使うため、同時に1件しか処理しない
    demo.queue(default_concurrency_limit=1)
    # Why: 絵文字パレットのボタン配置やスクロールエリアのスタイルを適切に適用するため、
    #      起動時の CSS 引数に EMOJI_PALETTE_CSS を注入する。
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        inbrowser=True,
        allowed_paths=[str(_OUTPUT_DIR)],
        css=EMOJI_PALETTE_CSS,
    )


if __name__ == "__main__":
    main()