"""
TTS生成履歴の閲覧・編集UI（Streamlit版）。

DBに保存された生成データをカード形式で一覧表示し、
フィルター・ソート・編集（レーティング/お気に入り/メモ）機能を提供する。

起動方法:
    streamlit run my/streamlit_history.py

Why Streamlit:
    Gradioは生成UIに向いているが、一覧表示・フィルタリング・編集のような
    CRUD的な操作にはStreamlitの方がレイアウトの自由度が高い。
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  sys.path にプロジェクトルートを追加
#
#  Why: Streamlitのスクリプトランナーはスクリプトのあるディレクトリ（my/）を
#       sys.path に追加するが、プロジェクトルートは追加しない。
#       そのため「from my.db import ...」のように my パッケージを参照する
#       インポートが ModuleNotFoundError になる。
#       このファイル（my/streamlit_history.py）の親の親 = プロジェクトルート を
#       sys.path の先頭に追加することで解決する。
#       PYTHONPATH 環境変数に頼る方法は Windows(MINGW64) 環境で不安定なため、
#       Python 内で直接 sys.path を操作する。
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

from my.db import init_db, select_generations, update_generation  # noqa: E402

# --------------------------------------------------------------------------- #
#  Streamlit バージョン互換ヘルパー
# --------------------------------------------------------------------------- #


def _safe_rerun() -> None:
    """
    Streamlitのバージョンに応じて画面を再実行（rerun）する。

    Why:
        st.rerun() は Streamlit 1.27.0 で追加された正式 API。
        それ以前のバージョン（例: 1.22.0）では st.experimental_rerun() を
        使う必要がある。どちらのバージョンでも動作するよう、
        hasattr で分岐して呼び分ける。
    """
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        # Streamlit < 1.27.0 用のフォールバック
        st.experimental_rerun()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  定数
# --------------------------------------------------------------------------- #

# ソート順の表示名とDB側のorder_byキーの対応
_SORT_OPTIONS: dict[str, str] = {
    "新しい順": "created_at_desc",
    "古い順": "created_at_asc",
    "レーティング順": "rating_desc",
    "お気に入り優先": "favorite_desc",
}

# レーティングの選択肢（0=未評価、1〜5=評価値）
_RATING_OPTIONS: list[str] = ["未評価", "★1", "★2", "★3", "★4", "★5"]


# --------------------------------------------------------------------------- #
#  ページ設定
#  - Streamlitの最初の呼び出しは set_page_config でなければならない
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="TTS生成履歴",
    page_icon="🎵",
    layout="wide",
)

# --------------------------------------------------------------------------- #
#  カスタムCSS
#  - Streamlitのデフォルトスタイルに加え、カード形式の見た目を整える
#  - st.markdown(unsafe_allow_html=True) で注入する
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
    /* カードコンテナ: 各生成レコードを囲むボックス */
    .gen-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.02);
        transition: box-shadow 0.2s ease;
    }
    .gen-card:hover {
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
    }

    /* メタ情報のラベルスタイル */
    .meta-label {
        font-size: 0.75rem;
        color: rgba(128, 128, 128, 0.8);
        margin-bottom: 0.1rem;
    }
    .meta-value {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        word-break: break-all;
    }

    /* テキスト表示を見やすく */
    .gen-text {
        font-size: 1.0rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .gen-caption {
        font-size: 0.85rem;
        color: rgba(128, 128, 128, 0.9);
        font-style: italic;
        margin-bottom: 0.5rem;
    }

    /* お気に入りバッジ */
    .fav-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    /* ファイルなし警告 */
    .file-missing {
        color: #e74c3c;
        font-size: 0.85rem;
        padding: 0.3rem 0;
    }

    /* カード内のセパレーター */
    .card-divider {
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        margin: 0.8rem 0;
    }

    /* フローティング上に戻るボタン
       Why: 生成履歴はレコード数（最大50件以上）が多く縦に長いページになるため、
            どのスクロール位置からでもワンクリックで最上部に戻れるようにする。
            Streamlitのiframe環境やスクロールコンテナが異なる場合を考慮し、
            window自体と.mainコンテナの双方に対してスムーズスクロールを行う。 */
    .scroll-to-top-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        font-size: 20px;
        border: none;
        outline: none;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .scroll-to-top-btn:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 6px 20px rgba(238, 90, 36, 0.4);
        background: linear-gradient(135deg, #ee5a24, #ff6b6b);
    }
    .scroll-to-top-btn:active {
        transform: translateY(-2px) scale(0.98);
    }
    </style>

    <!-- 上に戻るボタンHTML
         Why: スクリプトの再実行（rerun）のたびに複雑なJSイベントリスナーを登録するのを防ぎ、
              シンプルかつ堅牢に動作させるため、インラインのonclick属性でスクロールを実行する。 -->
    <button onclick="window.scrollTo({top: 0, behavior: 'smooth'}); const main = document.querySelector('.main'); if(main){main.scrollTo({top: 0, behavior: 'smooth'});}" class="scroll-to-top-btn" title="上に戻る">
        ▲
    </button>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
#  DB初期化
#  - テーブルが未作成の場合に備えて起動時に1回だけ呼ぶ
# --------------------------------------------------------------------------- #

init_db()


# --------------------------------------------------------------------------- #
#  サイドバー: フィルター・検索・ソート
# --------------------------------------------------------------------------- #

st.sidebar.title("🔍 フィルター")

# Why: 
#   以前は、チェックボックスやお気に入り、ソート順などのUI操作のたびに
#   DBアクセスや画面の再描画（Streamlitの再実行）が走るのを防ぐために、
#   st.sidebar.form を使用して「適用する」ボタンが押されるまで実行を保留していました。
#   しかし、これによって「キーワード検索のテキストボックスでEnterキーを押しても、
#   明示的に適用ボタンをクリックするまで検索が反映されない」という不便な挙動になっていました。
#   SQLiteへのDBアクセスは非常に高速（ミリ秒単位で完了）であり、リアクティブな操作性の方が
#   利便性が高いため、フォーム（st.sidebar.form）を廃止し、Enterキーの押下やチェックボックス等の
#   変更によって即座に検索が適用されるように変更しました。
#   これにより、テキストボックスにキーワードを入力してEnterを押すだけで検索が実行されます。

# キーワード検索
# text / caption のどちらかに部分一致すれば表示する
keyword = st.sidebar.text_input(
    "キーワード検索",
    value="",
    placeholder="text / caption で検索...",
    help="生成テキストまたはキャプション(スタイルプロンプト)を部分一致で検索します。",
)

# お気に入りのみ表示
favorite_only = st.sidebar.checkbox(
    "⭐ お気に入りのみ表示",
    value=False,
)

# ソート順
sort_label = st.sidebar.selectbox(
    "📊 ソート順",
    options=list(_SORT_OPTIONS.keys()),
    index=0,  # デフォルト: 新しい順
)

# 表示ラベルからDB側のキーに変換
sort_key = _SORT_OPTIONS[sort_label]

# フィルターのいずれかの状態（キーワード、お気に入り、ソート順）が変更された場合、
# 表示上限数（display_limit）を初期値（50）に戻すためのリセット処理を行います。
#
# Why:
#   「もっと読み込む」ボタン等をクリックして display_limit が100や150に増えている状態で、
#   ユーザーが別の検索条件を入力した場合、新しい検索条件に適合する上位50件から表示するのが自然だからです。
#   Streamlitは状態が変化するとスクリプトを最初から再実行する仕組みのため、
#   現在のフィルター値と前回のフィルター値を st.session_state（セッション情報を保持する仕組み）に
#   保存して比較し、差分があれば display_limit をリセットします。
filter_state = (keyword, favorite_only, sort_key)
if "prev_filter_state" not in st.session_state:
    st.session_state["prev_filter_state"] = filter_state

if st.session_state["prev_filter_state"] != filter_state:
    st.session_state["display_limit"] = 50
    st.session_state["prev_filter_state"] = filter_state

st.sidebar.markdown("---")
st.sidebar.caption("🔧 更新設定")

# 専用の更新ボタン
if st.sidebar.button("🔄 今すぐ最新化", use_container_width=True):
    _safe_rerun()

# 自動更新の間隔（0でオフ）
refresh_interval = st.sidebar.number_input(
    "自動更新間隔 (秒)",
    min_value=0,
    max_value=3600,
    value=0,
    step=5,
    help="指定した秒数ごとに自動で履歴を再取得します。0を設定するとオフになります。",
)

if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, limit=None, key="history_auto_refresh")

st.sidebar.markdown("---")
st.sidebar.caption("TTS生成履歴ブラウザ v1.0")


# --------------------------------------------------------------------------- #
#  データ取得
#  - select_generations() でDB検索し結果をリストで取得
# --------------------------------------------------------------------------- #

# セッションステートに表示上限がなければ初期化
if "display_limit" not in st.session_state:
    st.session_state["display_limit"] = 50

current_limit = st.session_state["display_limit"]

rows, total_count = select_generations(
    keyword=keyword if keyword else None,
    favorite_only=favorite_only,
    order_by=sort_key,
    limit=current_limit,
)


# --------------------------------------------------------------------------- #
#  メインエリア: ヘッダー
# --------------------------------------------------------------------------- #

st.title("🎵 TTS生成履歴")
st.caption(f"全 {total_count} 件")

if not rows:
    st.info("条件に一致するデータがありません。")
    st.stop()


# --------------------------------------------------------------------------- #
#  ヘルパー関数
# --------------------------------------------------------------------------- #


def _format_checkpoint(checkpoint: str | None) -> str:
    """
    チェックポイント名を短縮表示する。

    長いHuggingFaceのrepo IDやローカルパスを省略して見やすくする。
    例: "Aratako/Irodori-TTS-500M-v2-VoiceDesign" → "Irodori-TTS-500M-v2-VoiceDesign"
    """
    if not checkpoint:
        return "不明"
    # '/' が含まれていれば最後の部分だけ表示（HF repo IDの場合）
    if "/" in checkpoint:
        return checkpoint.rsplit("/", 1)[-1]
    # ローカルパスの場合はファイル名部分のみ
    return Path(checkpoint).stem if "\\" in checkpoint or "/" in checkpoint else checkpoint


def _format_datetime(iso_str: str | None) -> str:
    """
    ISO 8601形式の文字列を人間が読みやすい形式に変換する。

    例: "2026-04-06T19:00:00+09:00" → "2026/04/06 19:00:00"
    """
    if not iso_str:
        return "不明"
    # ISO 8601のパース（タイムゾーン付きに対応）
    try:
        # 'T'を空白に置換し、先頭の19文字（YYYY-MM-DD HH:MM:SS）だけを取得する
        return iso_str.replace("T", " ")[:19]
    except Exception:
        return iso_str


def _rating_to_index(rating: int | None) -> int:
    """
    DB上のレーティング値（None / 1〜5）を、
    _RATING_OPTIONS リストのインデックス（0〜5）に変換する。

    None（未評価）→ 0, 1 → 1, ..., 5 → 5
    """
    if rating is None or rating == 0:
        return 0
    return rating


def _index_to_rating(index: int) -> int | None:
    """
    _RATING_OPTIONS のインデックスをDB上のレーティング値に変換する。

    0（未評価）→ None, 1 → 1, ..., 5 → 5
    """
    if index == 0:
        return None
    return index


# --------------------------------------------------------------------------- #
#  メインエリア: カード一覧表示
#  - 各レコードを1枚のカードとして表示
#  - 音声再生、情報表示、編集機能を含む
# --------------------------------------------------------------------------- #

for row in rows:
    gen_id: int = row["id"]
    file_path_str: str = row["file_path"]
    file_path = Path(file_path_str)

    # ---------- カード開始 ----------
    # st.container で1レコード分をグルーピング
    with st.container():
        st.markdown('<div class="gen-card">', unsafe_allow_html=True)

        # --- 上段: テキスト + お気に入りバッジ ---
        fav_badge = '<span class="fav-badge">⭐ お気に入り</span>' if row["favorite"] else ""
        st.markdown(
            f'<div class="gen-text">{row["text"]}{fav_badge}</div>',
            unsafe_allow_html=True,
        )

        # キャプション（スタイルプロンプト）
        if row["caption"]:
            st.markdown(
                f'<div class="gen-caption">🎨 {row["caption"]}</div>',
                unsafe_allow_html=True,
            )

        # ファイル名（DBまたはファイルパスから抽出）
        filename_display = row.get("filename") or file_path.name
        st.markdown(
            f'<div class="gen-caption" style="color: #888;">📁 {filename_display}</div>',
            unsafe_allow_html=True,
        )

        # --- 音声再生 ---
        if file_path.exists():
            st.audio(str(file_path))
        else:
            st.markdown(
                '<div class="file-missing">⚠️ 音声ファイルが見つかりません</div>',
                unsafe_allow_html=True,
            )

        # --- メタ情報（横並び表示） ---
        # st.columnsで各項目の幅を調整（seedや日時は長いため広く取る）
        meta_cols = st.columns([1.5, 1, 2.5, 2, 1.5])
        with meta_cols[0]:
            st.markdown('<div class="meta-label">🌱 Seed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-value">{row["seed"] if row["seed"] is not None else "—"}</div>',
                unsafe_allow_html=True,
            )
        with meta_cols[1]:
            st.markdown('<div class="meta-label">🔄 Steps</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-value">{row["num_steps"] if row["num_steps"] is not None else "—"}</div>',
                unsafe_allow_html=True,
            )
        with meta_cols[2]:
            st.markdown('<div class="meta-label">🏷️ Checkpoint</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-value">{_format_checkpoint(row["checkpoint"])}</div>',
                unsafe_allow_html=True,
            )
        with meta_cols[3]:
            st.markdown('<div class="meta-label">📅 生成日時</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-value">{_format_datetime(row["created_at"])}</div>',
                unsafe_allow_html=True,
            )
        with meta_cols[4]:
            st.markdown('<div class="meta-label">⭐ レーティング</div>', unsafe_allow_html=True)
            current_rating = row["rating"]
            if current_rating is not None:
                st.markdown(
                    f'<div class="meta-value">{"★" * current_rating}{"☆" * (5 - current_rating)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="meta-value">未評価</div>',
                    unsafe_allow_html=True,
                )

        # --- メタ情報2段目（新パラメータ等） ---
        # 既存レコード（旧スキーマ）との互換性のため、すべて row.get() を使って安全に取得し、
        # カラムが存在しない・値が設定されていない場合は適切な代替表記にフォールバックさせる。
        meta2_cols = st.columns([1.5, 1.5, 2.5, 2.5, 1])
        with meta2_cols[0]:
            st.markdown('<div class="meta-label">⏱️ Duration Scale / Sec</div>', unsafe_allow_html=True)
            dur = row.get("duration_scale")
            sec = row.get("seconds")
            dur_str = f"{dur:.2f}x" if dur is not None else "—"
            sec_str = f"{sec:.1f}s" if sec is not None else "auto"
            st.markdown(
                f'<div class="meta-value">{dur_str} / {sec_str}</div>',
                unsafe_allow_html=True,
            )
        with meta2_cols[1]:
            st.markdown('<div class="meta-label">📈 Schedule / Sway</div>', unsafe_allow_html=True)
            tsched = row.get("t_schedule_mode")
            sway = row.get("sway_coeff")
            tsched_str = str(tsched) if tsched is not None else "—"
            sway_str = f"{sway:.1f}" if sway is not None else "—"
            st.markdown(
                f'<div class="meta-value">{tsched_str} (coeff: {sway_str})</div>',
                unsafe_allow_html=True,
            )
        with meta2_cols[2]:
            st.markdown('<div class="meta-label">🎧 Speaker (CFG / Embed)</div>', unsafe_allow_html=True)
            spk_cfg = row.get("cfg_scale_speaker")
            spk_embed = row.get("speaker_embedding")
            spk_cfg_str = f"{spk_cfg:.1f}" if spk_cfg is not None else "—"
            spk_embed_str = Path(spk_embed).name if spk_embed else "—"
            st.markdown(
                f'<div class="meta-value">CFG: {spk_cfg_str} / {spk_embed_str}</div>',
                unsafe_allow_html=True,
            )
        with meta2_cols[3]:
            st.markdown('<div class="meta-label">🎨 LoRA Adapter</div>', unsafe_allow_html=True)
            lora = row.get("lora_adapter")
            lora_str = Path(lora).name if lora else "—"
            st.markdown(
                f'<div class="meta-value">{lora_str}</div>',
                unsafe_allow_html=True,
            )
        with meta2_cols[4]:
            st.markdown('<div class="meta-label">🏷️ UI / Model Ver</div>', unsafe_allow_html=True)
            ui_ver = row.get("ui_version") or "—"
            model_ver = row.get("model_version") or "—"
            st.markdown(
                f'<div class="meta-value">{ui_ver} / {model_ver}</div>',
                unsafe_allow_html=True,
            )

        # --- 区切り線 ---
        st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)

        # --- 編集エリア ---
        # st.expander で折りたたみにして、通常時はすっきり表示
        with st.expander("✏️ 編集", expanded=False):
            edit_cols = st.columns([1, 1, 2])

            # レーティング（1〜5, 未評価=0）
            with edit_cols[0]:
                new_rating_idx = st.selectbox(
                    "レーティング",
                    options=range(len(_RATING_OPTIONS)),
                    format_func=lambda i: _RATING_OPTIONS[i],
                    index=_rating_to_index(row["rating"]),
                    key=f"rating_{gen_id}",
                )
                new_rating = _index_to_rating(new_rating_idx)

            # お気に入りトグル
            with edit_cols[1]:
                new_favorite = st.checkbox(
                    "⭐ お気に入り",
                    value=bool(row["favorite"]),
                    key=f"fav_{gen_id}",
                )

            # メモ入力
            with edit_cols[2]:
                new_note = st.text_area(
                    "メモ",
                    value=row["note"] or "",
                    height=80,
                    key=f"note_{gen_id}",
                    placeholder="自由にメモを記入...",
                )

            # 保存ボタン
            # ボタン押下時にDB更新し、画面をリロードして反映する
            #
            # Why tryの外でrerun:
            #   st.rerun() は内部的に RerunException という特殊な例外をraiseして
            #   スクリプト実行を中断→再実行する仕組みになっている。
            #   try ブロック内で st.rerun() を呼ぶと、この RerunException が
            #   except 節に捕捉されたり、Streamlit側でエラー表示されてしまう。
            #   そのため、DB更新は try 内で行い、rerun は try の外で呼ぶ。
            #
            # Why session_state で成功メッセージを管理:
            #   st.rerun() を呼ぶとスクリプトが先頭から再実行されるため、
            #   rerun 前に st.success() を表示しても即座に消えてしまう。
            #   session_state にフラグを保存し、再実行後にメッセージを表示する。
            save_succeeded = False
            if st.button("💾 保存", key=f"save_{gen_id}"):
                try:
                    update_generation(
                        gen_id,
                        favorite=1 if new_favorite else 0,
                        rating=new_rating,
                        note=new_note if new_note else None,
                    )
                    save_succeeded = True
                except ValueError as e:
                    st.error(f"保存に失敗しました: {e}")

            # rerun は try の外で呼ぶ（RerunException が except に捕捉されるのを防ぐ）
            if save_succeeded:
                # session_state に保存成功のメッセージを記録
                st.session_state[f"save_msg_{gen_id}"] = "保存しました！"
                _safe_rerun()

            # 前回の保存成功メッセージがあれば表示し、表示後にクリアする
            _save_msg_key = f"save_msg_{gen_id}"
            if _save_msg_key in st.session_state:
                st.success(st.session_state[_save_msg_key])
                del st.session_state[_save_msg_key]

        # カード終了タグ
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  フッター: もっと見るボタン
# --------------------------------------------------------------------------- #

# 表示されていないアイテムがあれば、ボタンを表示
if total_count > len(rows):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔽 さらに読み込む", use_container_width=True):
            st.session_state["display_limit"] += 50
            _safe_rerun()