"""
生成履歴を管理するSQLiteデータベースモジュール。

TTS音声生成のパラメータ・ファイルパス・メタ情報をSQLiteに永続化する。
生成UI（Gradio）からの書き込みと、閲覧UI（Streamlit）からの読み取りの
両方で共通利用される。

DBファイルは my/data/generations.db に配置される。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# DBファイルのデフォルトパス
# このモジュールの親ディレクトリ（my/）の下にある data/ に配置する
_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "generations.db"

# --------------------------------------------------------------------------- #
#  テーブル定義SQL
#  - generations: 生成1件ごとのパラメータ・ファイル情報・評価を格納
#  - tags / generation_tags: Phase 4 で使うタグ機能（先行定義のみ）
# --------------------------------------------------------------------------- #

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS generations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,  -- ISO 8601 形式（例: 2026-04-06T19:00:00+09:00）
    text              TEXT    NOT NULL,  -- 生成に使ったテキスト
    caption           TEXT,             -- スタイルプロンプト（空の場合は NULL）
    seed              INTEGER,          -- 生成に使ったシード値
    num_steps         INTEGER,          -- サンプリングステップ数
    cfg_scale_text    REAL,             -- CFG Scale（テキスト側）
    cfg_scale_caption REAL,             -- CFG Scale（キャプション側）
    cfg_guidance_mode TEXT,             -- CFG ガイダンスモード（independent / joint / alternating）
    checkpoint        TEXT,             -- チェックポイントのパスまたは HF repo ID
    file_path         TEXT    NOT NULL, -- 生成された wav ファイルのパス
    filename          TEXT,             -- 生成された wav ファイルの名前（拡張子含む）
    favorite          INTEGER DEFAULT 0,-- お気に入りフラグ（0=なし, 1=あり）
    rating            INTEGER,          -- レーティング（1〜5, 未評価は NULL）
    note              TEXT,             -- ユーザーメモ（自由記述）
    duration_scale    REAL,             -- 音声長の伸縮倍率（v3パラメータ）
    seconds           REAL,             -- 指定秒数（v3パラメータ, 空欄時自動）
    t_schedule_mode   TEXT,             -- 時間スケジューリングモード（v3パラメータ, linear/sway）
    sway_coeff        REAL,             -- sway係数（v3パラメータ）
    lora_adapter      TEXT,             -- LoRAアダプタのディレクトリパス（v3パラメータ）
    speaker_embedding TEXT,             -- Speaker Inversion の safetensors パス（v3パラメータ）
    cfg_scale_speaker REAL,             -- Speaker 側の CFG Scale（v3パラメータ）
    ref_wav           TEXT,             -- 参照音声ファイル名（v3パラメータ, 空の場合は NULL）
    ui_version        TEXT,             -- 生成したUIのバージョン
    model_version     TEXT              -- 推定されたモデルのバージョン
);

CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE  -- タグ名（重複不可）
);

CREATE TABLE IF NOT EXISTS generation_tags (
    generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    tag_id        INTEGER NOT NULL REFERENCES tags(id)        ON DELETE CASCADE,
    PRIMARY KEY (generation_id, tag_id)
);
"""


def _get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """
    SQLite接続を取得するヘルパー関数。

    - WAL（Write-Ahead Logging）モードを有効化して、
      読み取りと書き込みの同時実行を可能にする。
      （生成UIが書き込み中でも閲覧UIが読み取れる）
    - 外部キー制約を有効化する（タグ関連テーブルの整合性のため）

    Args:
        db_path: DBファイルのパス。None の場合は _DEFAULT_DB_PATH を使用。

    Returns:
        sqlite3.Connection: データベース接続オブジェクト
    """
    path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    # 親ディレクトリが存在しない場合は作成する
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    # WALモード: 読み書きの並行性を向上させる
    conn.execute("PRAGMA journal_mode=WAL;")
    # 外部キー制約を有効化（SQLiteはデフォルトで無効）
    conn.execute("PRAGMA foreign_keys=ON;")
    # SELECT結果をdictライクに扱えるようにする
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    """
    指定したテーブルにカラムが存在しない場合、ALTER TABLE を用いてカラムを追加する。
    これにより、すでにDBが存在している環境であっても、スキーマ変更時にエラーを発生させずに
    安全に新パラメータ用の列を追加（マイグレーション）できる。
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def guess_model_version(checkpoint: str) -> str | None:
    """
    使用したチェックポイント名（パスやハギングフェイスのリポジトリID）から、
    モデルのバージョン（v2, v2-voicedesign, v3, v3-voicedesignなど）を推定して文字列で返す。
    履歴画面でどのバージョンのモデルで生成されたか判別するために利用する。
    """
    s = str(checkpoint).lower()
    # v3 の VoiceDesign モデルを優先して判定する
    if "v3-voicedesign" in s or ("v3" in s and ("voicedesign" in s or "voice_design" in s)):
        return "v3-voicedesign"
    if "v3" in s:
        return "v3"
    if "v2-voicedesign" in s or "voice_design" in s or "voicedesign" in s:
        return "v2-voicedesign"
    if "v2" in s:
        return "v2"
    return None


def init_db(db_path: Path | str | None = None) -> Path:
    """
    DBファイルを初期化し、テーブルが存在しなければ作成する。

    初回起動時や、DBファイルが未作成の場合に呼ぶ。
    すでにテーブルが存在する場合は、新規カラム（v3用およびバージョン情報用）が
    存在するか確認し、なければ自動で追加（マイグレーション）する。

    Args:
        db_path: DBファイルのパス。None の場合はデフォルトパスを使用。

    Returns:
        Path: 使用したDBファイルの絶対パス
    """
    resolved_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    conn = _get_connection(resolved_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        
        # マイグレーション: filename カラムが存在しない場合は追加し、既存データに値を詰める
        # 競合を防ぐため、ALTER TABLE を try/except で囲む（並行起動時の安全性）
        try:
            conn.execute("ALTER TABLE generations ADD COLUMN filename TEXT;")
        except sqlite3.OperationalError as e:
            # カラムがすでに存在する場合はエラーを無視して続行
            if "duplicate column name" not in str(e).lower():
                raise

        # v3 新サンプリングパラメータおよびUI/モデルバージョン列を安全（冪等）に追加
        # 既存データベースがある場合でも、この処理によりスキーマが自動拡張される
        _ensure_column(conn, "generations", "duration_scale", "REAL")
        _ensure_column(conn, "generations", "seconds", "REAL")
        _ensure_column(conn, "generations", "t_schedule_mode", "TEXT")
        _ensure_column(conn, "generations", "sway_coeff", "REAL")
        _ensure_column(conn, "generations", "lora_adapter", "TEXT")
        _ensure_column(conn, "generations", "speaker_embedding", "TEXT")
        _ensure_column(conn, "generations", "cfg_scale_speaker", "REAL")
        _ensure_column(conn, "generations", "ref_wav", "TEXT")
        _ensure_column(conn, "generations", "ui_version", "TEXT")
        _ensure_column(conn, "generations", "model_version", "TEXT")

        # 既存レコードの file_path からファイル名を抽出して filename カラムに反映する
        # filenameが未格納（NULL）のレコードのみを対象にすることで無駄な更新を防ぐ
        rows = conn.execute("SELECT id, file_path FROM generations WHERE filename IS NULL").fetchall()
        for row in rows:
            row_id = row["id"]
            file_path_str = row["file_path"]
            # プラットフォームに依存せずスラッシュとバックスラッシュの両方で分割してファイル名を取得
            fname = file_path_str.replace("\\", "/").split("/")[-1]
            conn.execute(
                "UPDATE generations SET filename = ? WHERE id = ?",
                (fname, row_id)
            )

        conn.commit()
    finally:
        conn.close()
    return resolved_path.resolve()


def insert_generation(
    *,
    text: str,
    file_path: str,
    filename: str | None = None,
    caption: str | None = None,
    seed: int | None = None,
    num_steps: int | None = None,
    cfg_scale_text: float | None = None,
    cfg_scale_caption: float | None = None,
    cfg_guidance_mode: str | None = None,
    checkpoint: str | None = None,
    created_at: str | None = None,
    duration_scale: float | None = None,
    seconds: float | None = None,
    t_schedule_mode: str | None = None,
    sway_coeff: float | None = None,
    lora_adapter: str | None = None,
    speaker_embedding: str | None = None,
    cfg_scale_speaker: float | None = None,
    ref_wav: str | None = None,
    ui_version: str | None = None,
    model_version: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """
    生成結果を1件INSERTする。

    生成UI（Gradio）から生成完了時に呼び出される想定。
    キーワード引数のみを受け付ける（呼び出し側で引数名を明示させるため）。

    Args:
        text:              生成に使ったテキスト（必須）
        file_path:         生成された wav ファイルのパス（必須）
        filename:          生成された wav ファイルのファイル名
        caption:           スタイルプロンプト（省略可）
        seed:              シード値
        num_steps:         サンプリングステップ数
        cfg_scale_text:    CFG Scale（テキスト側）
        cfg_scale_caption: CFG Scale（キャプション側）
        cfg_guidance_mode: CFG ガイダンスモード
        checkpoint:        チェックポイントのパスまたは HF repo ID
        created_at:        生成日時（ISO 8601形式）。省略時は現在時刻。
        duration_scale:    音声長の伸縮倍率（v3パラメータ）
        seconds:           指定秒数（v3パラメータ）
        t_schedule_mode:   時間スケジューリングモード（v3パラメータ）
        sway_coeff:        sway係数（v3パラメータ）
        lora_adapter:      LoRAアダプタのディレクトリパス（v3パラメータ）
        speaker_embedding: Speaker Inversion の safetensors パス（v3パラメータ）
        cfg_scale_speaker: Speaker 側の CFG Scale（v3パラメータ）
        ref_wav:           参照音声ファイル名（v3パラメータ）
        ui_version:        生成したUIのバージョン
        model_version:     推定されたモデルのバージョン
        db_path:           DBファイルのパス。None の場合はデフォルト。

    Returns:
        int: 挿入された行の id（AUTOINCREMENT で採番された値）
    """
    if created_at is None:
        # タイムゾーン付きの現在時刻をISO 8601形式で生成
        created_at = datetime.now(timezone.utc).astimezone().isoformat()

    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO generations
                (created_at, text, caption, seed, num_steps,
                 cfg_scale_text, cfg_scale_caption, cfg_guidance_mode,
                 checkpoint, file_path, filename, duration_scale,
                 seconds, t_schedule_mode, sway_coeff, lora_adapter,
                 speaker_embedding, cfg_scale_speaker, ref_wav, ui_version, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                text,
                caption,
                seed,
                num_steps,
                cfg_scale_text,
                cfg_scale_caption,
                cfg_guidance_mode,
                checkpoint,
                file_path,
                filename,
                duration_scale,
                seconds,
                t_schedule_mode,
                sway_coeff,
                lora_adapter,
                speaker_embedding,
                cfg_scale_speaker,
                ref_wav,
                ui_version,
                model_version,
            ),
        )
        conn.commit()
        # lastrowid: INSERT直後に自動採番されたid
        row_id = cursor.lastrowid
        assert row_id is not None, "INSERT後にlastrowidがNoneになるのは想定外"
        return row_id
    finally:
        conn.close()


def select_generations(
    *,
    keyword: str | None = None,
    favorite_only: bool = False,
    order_by: str = "created_at_desc",
    limit: int | None = None,
    offset: int = 0,
    db_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    生成履歴を検索・取得する。

    閲覧UI（Streamlit）や生成UI（Gradio）の履歴表示で使用。
    フィルター・ソート・ページングに対応。

    Args:
        keyword:       text/caption のキーワード検索（部分一致）。
                       None または空文字の場合はフィルターなし。
        favorite_only: True の場合、お気に入りのみに絞り込む。
        order_by:      ソート順。以下のいずれか:
                       - "created_at_desc" (新しい順, デフォルト)
                       - "created_at_asc"  (古い順)
                       - "rating_desc"     (レーティング高い順)
                       - "favorite_desc"   (お気に入り優先)
        limit:         取得件数の上限。None で全件。
        offset:        取得開始位置（ページング用）。
        db_path:       DBファイルのパス。None の場合はデフォルト。

    Returns:
        tuple[list[dict], int]: (各行を辞書化したリスト, フィルター適用後の総件数)
    """
    # ソート順のマッピング（SQLインジェクション防止のためホワイトリスト方式）
    _ORDER_MAP: dict[str, str] = {
        "created_at_desc": "created_at DESC",
        "created_at_asc": "created_at ASC",
        "rating_desc": "rating DESC NULLS LAST",
        "favorite_desc": "favorite DESC, created_at DESC",
    }
    order_clause = _ORDER_MAP.get(order_by)
    if order_clause is None:
        raise ValueError(
            f"order_by に不正な値が指定されました: {order_by!r}。"
            f"使用可能な値: {list(_ORDER_MAP.keys())}"
        )

    # WHERE句とパラメータを動的に組み立てる
    conditions: list[str] = []
    params: list[Any] = []

    if keyword:
        # text または caption または ref_wav に部分一致する行を検索
        conditions.append("(text LIKE ? OR caption LIKE ? OR ref_wav LIKE ?)")
        like_pattern = f"%{keyword}%"
        params.extend([like_pattern, like_pattern, like_pattern])

    if favorite_only:
        conditions.append("favorite = 1")

    where_sql = ""
    if conditions:
        where_sql = "WHERE " + " AND ".join(conditions)

    # SQLiteは "NULLS LAST" をネイティブサポートしないが、
    # "rating DESC NULLS LAST" は CASE式で代替する
    if "NULLS LAST" in order_clause:
        # NULL を最後尾に送るための CASE式に書き換え
        # 例: "rating DESC NULLS LAST"
        #   → "CASE WHEN rating IS NULL THEN 1 ELSE 0 END, rating DESC"
        col = order_clause.split()[0]  # "rating"
        direction = order_clause.split()[1]  # "DESC"
        order_clause = f"CASE WHEN {col} IS NULL THEN 1 ELSE 0 END, {col} {direction}"

    conn = _get_connection(db_path)
    try:
        # 総件数を取得（フィルター適用後、limit/offset適用前）
        count_query = f"SELECT COUNT(*) FROM generations {where_sql}"
        total_count = conn.execute(count_query, params[: len(params)]).fetchone()[0]

        # データを取得（limit/offset適用）
        query = f"SELECT * FROM generations {where_sql} ORDER BY {order_clause}"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset > 0:
            # limitなしでもoffsetを使いたい場合は大きなlimitを指定
            query += " LIMIT -1 OFFSET ?"
            params.append(offset)

        rows = conn.execute(query, params).fetchall()
        # sqlite3.Row を普通のdictに変換して返す
        # （sqlite3.Row はJSON化やDataFrame化で扱いにくいため）
        return [dict(row) for row in rows], total_count
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
#  特定IDの generation を更新する（お気に入り・レーティング・メモ）
# --------------------------------------------------------------------------- #

# _SENTINEL: 「引数が指定されなかった」ことを判別するための特別な値。
# None を「値をクリアする（NULLにする）」意味で使いたいため、
# デフォルト値として None は使えない。そこで別のオブジェクトで区別する。
_SENTINEL = object()


def update_generation(
    generation_id: int,
    *,
    favorite: int | object = _SENTINEL,
    rating: int | None | object = _SENTINEL,
    note: str | None | object = _SENTINEL,
    db_path: Path | str | None = None,
) -> None:
    """
    指定IDの生成レコードを部分更新する。

    閲覧UI（Streamlit）での編集操作時に呼び出される想定。
    指定されたフィールドのみUPDATEし、指定されなかったフィールドは変更しない。

    _SENTINEL という特別な値で「引数が渡されなかった」ことを判別している。
    これは、None を「値をクリアする（NULL にリセットする）」意味で使いたいため。
    例: rating=None → レーティングを未評価に戻す

    Args:
        generation_id: 更新対象の行ID（必須）
        favorite:      お気に入りフラグ（0=なし, 1=あり）
        rating:        レーティング（1〜5）。None で未評価にリセット。
        note:          ユーザーメモ。None でクリア。
        db_path:       DBファイルのパス。None の場合はデフォルト。

    Raises:
        ValueError: 更新対象のフィールドが1つも指定されなかった場合
        ValueError: 指定IDのレコードが存在しない場合
    """
    # 更新対象のカラムと値を動的に組み立てる
    updates: list[str] = []
    params: list[Any] = []

    if favorite is not _SENTINEL:
        updates.append("favorite = ?")
        params.append(favorite)

    if rating is not _SENTINEL:
        updates.append("rating = ?")
        params.append(rating)

    if note is not _SENTINEL:
        updates.append("note = ?")
        params.append(note)

    if not updates:
        raise ValueError(
            "update_generation() には少なくとも1つの更新フィールド"
            "（favorite, rating, note）を指定してください。"
        )

    # WHERE句のパラメータ（IDで絞り込み）
    params.append(generation_id)

    sql = f"UPDATE generations SET {', '.join(updates)} WHERE id = ?"

    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(sql, params)
        conn.commit()
        # rowcount: UPDATEで影響を受けた行数。0の場合はIDが存在しない
        if cursor.rowcount == 0:
            raise ValueError(
                f"ID={generation_id} のレコードが見つかりません。"
                "すでに削除されている可能性があります。"
            )
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
#  動作確認用: python -m my.db で初期化テスト
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    db_file = init_db()
    print(f"DB初期化完了: {db_file}")

    # テスト用のデータをINSERT
    test_id = insert_generation(
        text="テスト用テキスト",
        caption="明るく元気な女性の声",
        seed=42,
        num_steps=40,
        cfg_scale_text=2.0,
        cfg_scale_caption=4.0,
        cfg_guidance_mode="independent",
        checkpoint="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        file_path="my/data/test_20260406_190000_42.wav",
        filename="test_20260406_190000_42.wav",
    )
    print(f"INSERT成功: id={test_id}")

    # 全件取得テスト
    all_rows, total_count = select_generations()
    print(f"全件取得: {len(all_rows)} 件 (総件数: {total_count})")
    for row in all_rows:
        print(f"  id={row['id']}, text={row['text']!r}, seed={row['seed']}")

    # キーワード検索テスト
    keyword_rows, keyword_count = select_generations(keyword="テスト")
    print(f"キーワード検索 'テスト': {len(keyword_rows)} 件 (総件数: {keyword_count})")

    print("全テスト完了")