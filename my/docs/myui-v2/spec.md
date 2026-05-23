# myui-v2 仕様書

## 概要

myui-v1（`gradio_app_voicedesign.py` / `gradio_app.py` をベースにした独自UI）を、
本家 Irodori-TTS の v2 → v3 アップグレードに追従させた版。

v1 からの差分は **追加・追従のみ**。既存の生成UI/閲覧UI/DBの基本構造は維持する。

v3 本体側の変更点の詳細は [v2_to_v3_changes.md](../v2_to_v3_changes.md) を参照。

---

## v1 からの主な変更点

### 互換性復旧（必須）
本家の以下の破壊的変更に追従する：

| シンボル/シグネチャ | v2 | v3 |
|---|---|---|
| `FIXED_SECONDS` 定数 | あり | **削除**（Duration Predictor で代替） |
| `_build_runtime_key(...)` の `enable_watermark` 引数 | あり | **削除**（`silentcipher` パッケージ有無で自動判定） |
| `_describe_runtime(...)` の `enable_watermark` 引数 | あり | **削除** |
| `_load_model(...)` の `enable_watermark` 引数（参照版） | あり | **削除** |
| `SamplingRequest.seconds` のデフォルト | `30.0` 想定 | `None`（自動） |

### 新規UI要素（v3 機能の露出）

#### Sampling アコーディオン
- `Seconds (blank=auto)` — Duration Predictor を使うか手動指定するか
- `Duration Scale` — 0.5〜1.5 / 0.01 / 1.0 — 発話長スケール
- `Time Schedule` — `linear` / `sway` — Sway Sampling 切替
- `Sway Coeff` — -1.0〜1.5 / 0.1 / -1.0 — Sway 係数（`linear` 時は非activate）

#### Advanced アコーディオン
- `LoRA Adapter Directory (optional)` — 動的 LoRA ロード用パス

#### テキスト入力
- **絵文字パレット**（`build_emoji_palette` / `EMOJI_PALETTE_CSS`）を text 入力直下に追加
- Live Update と整合：パレットからの絵文字挿入も `text.change` を発火するため追加実装不要

#### gradio_ref.py（参照音声版）のみ
- **Speaker Embedding (.speaker.safetensors) アップロード欄**を追加
- Reference Audio との同時指定は `ValueError` で排他
- 両方未指定なら `no_ref` モード

### デフォルトチェックポイント
| UI | デフォルト | 備考 |
|---|---|---|
| `my/gradio_gen.py`（VoiceDesign版） | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | **v2 のまま維持**。v3 VoiceDesign 未リリース |
| `my/gradio_ref.py`（参照音声版） | `Aratako/Irodori-TTS-500M-v3` | v3 へ更新 |

本家 `_default_checkpoint()` をそのまま import で再利用すれば、上記がそのまま反映される。

### 設定永続化（`last_settings_*.json`）
- 新パラメータ（`seconds_raw`, `duration_scale`, `t_schedule_mode`, `sway_coeff`, `lora_adapter_raw`）を保存・復元対象に追加
- **下位互換**：`.get(key, default)` で全キー安全フォールバック。既存ユーザーの JSON は壊さない
- リロード時に `t_schedule_mode` を復元したら `sway_coeff` の `interactive` 状態もそれに合わせる

---

## ファイル構成（v1 から無変更）

| ファイル | フレームワーク | 役割 |
|---|---|---|
| `my/gradio_gen.py` | Gradio | 生成UI（VoiceDesign版）→ SQLite書き込み |
| `my/gradio_ref.py` | Gradio | 生成UI（参照音声版）→ SQLite書き込み |
| `my/streamlit_history.py` | Streamlit | SQLite読み取り・閲覧・編集 |
| `my/db.py` | - | SQLite共通ロジック |
| `my/data/generations.db` | SQLite | DBファイル本体 |

---

## DBスキーマ（v2 拡張）

v1 のスキーマを **追加のみ** で拡張する。`ALTER TABLE ADD COLUMN` で冪等に行う。

### 既存カラム（v1 から維持）
```sql
CREATE TABLE generations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    text              TEXT NOT NULL,
    caption           TEXT,
    seed              INTEGER,
    num_steps         INTEGER,
    cfg_scale_text    REAL,
    cfg_scale_caption REAL,
    cfg_guidance_mode TEXT,
    checkpoint        TEXT,
    file_path         TEXT NOT NULL,
    favorite          INTEGER DEFAULT 0,
    rating            INTEGER,
    note              TEXT,
    filename          TEXT
);
```

### v2 で追加するカラム

| カラム名 | 型 | nullable | 例 | 用途 |
|---|---|---|---|---|
| `duration_scale` | REAL | yes | `1.0` | Duration Predictor のスケール |
| `seconds` | REAL | yes | `NULL`=auto | 手動秒数指定（NULL = Duration Predictor に委譲） |
| `t_schedule_mode` | TEXT | yes | `linear`/`sway` | タイムステップスケジュール |
| `sway_coeff` | REAL | yes | `-1.0` | Sway Sampling 係数 |
| `lora_adapter` | TEXT | yes | path or `NULL` | LoRA アダプタディレクトリ |
| `speaker_embedding` | TEXT | yes | path or `NULL` | Speaker Inversion 埋め込みファイル（参照版のみ使用） |
| `cfg_scale_speaker` | REAL | yes | `5.0` | 参照版の CFG Scale Speaker（v1 では未保存） |
| `ui_version` | TEXT | yes | `v3-myui-2` | myUI 自体のバージョン（互換性追跡用） |
| `model_version` | TEXT | yes | `v2` / `v3` / `v2-voicedesign` | チェックポイント由来のモデル世代 |

### マイグレーション設計

- **冪等性**：`init_db()` 内で `PRAGMA table_info(generations)` から既存カラム集合を取得し、無いものだけ `ALTER TABLE ADD COLUMN` する
- **既存レコードの扱い**：すべて `NULL` で初期化。**遡及的に `model_version='v2'` などを埋めない**（実際に v2 で生成したかどうか判別不能なため）
- **`DROP COLUMN` は使わない**（SQLite バージョン互換のため）
- **共通ヘルパ**を `my/db.py` に追加：
  ```python
  def _ensure_column(conn, table: str, col: str, ddl: str) -> None:
      cur = conn.execute(f"PRAGMA table_info({table})")
      cols = {row[1] for row in cur.fetchall()}
      if col not in cols:
          conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
  ```

### `ui_version` と `model_version` の使い分け

| 列 | 何を表すか | 取得元 |
|---|---|---|
| `ui_version` | myUI 自体のバージョン。Phase 完了時に上げる | `my/__init__.py` の定数 `MY_UI_VERSION` |
| `model_version` | 使ったチェックポイントの世代 | チェックポイントパス/HF repo id から文字列マッチで簡易推定。失敗時は `NULL` |

両者を分けることで「v3 myUI で v2 ckpt を使った履歴」も後から区別できる。

### モデルバージョン推定ロジック（参考実装）
```python
def _guess_model_version(checkpoint: str) -> str | None:
    s = str(checkpoint).lower()
    if "v3" in s:
        return "v3"
    if "v2-voicedesign" in s or "voice_design" in s or "voicedesign" in s:
        return "v2-voicedesign"
    if "v2" in s:
        return "v2"
    return None  # 不明
```

---

## 生成UI 共通仕様（v1 から維持 + v2 追加）

### v1 から維持する機能
- `generate_forever` モード / Generate Forever / Cancel Forever ボタン
- Autoplay トグル（セッション変数でループ中も切り替え可）
- 直近5件の履歴表示
- 候補グリッドは廃止
- ファイル名規則 `{YYYYMMDD_HHMMSS}_{seed}.wav`
- キュー再生（最大10件・最新の生成結果より上に配置）
- 初期音量30%・ユーザー音量変更を記憶
- Live Update（text / caption / text のみの参照版）

### v2 で追加される挙動
- **新パラメータは Live Update の対象外**。Generate Forever 起動時のスナップショットで固定される。`seconds_raw / duration_scale / t_schedule_mode / sway_coeff / lora_adapter` などはループ中に変更しても次イテレーションに反映されない（仕様）
- `t_schedule_mode` Dropdown と `sway_coeff` Slider の連動制御
- 絵文字パレットを text 入力直下に展開（デフォルト折りたたみ）

### 入力パラメータ全体像（v2 完成時）

#### gradio_gen.py
1. checkpoint, model_device, model_precision, codec_device, codec_precision
2. text, caption
3. **Sampling**: num_steps, seed_raw, **seconds_raw, duration_scale**, **t_schedule_mode, sway_coeff**, cfg_guidance_mode, cfg_scale_text, cfg_scale_caption
4. **Advanced**: cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache, max_text_len_raw, max_caption_len_raw, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw, **lora_adapter_raw**
5. autoplay, forever, live_update

#### gradio_ref.py
1. checkpoint, model_device, model_precision, codec_device, codec_precision
2. text, uploaded_audio, **uploaded_speaker_embedding, speaker_embedding_path_raw**
3. **Sampling**: num_steps, seed_raw, **seconds_raw, duration_scale**, **t_schedule_mode, sway_coeff**, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker
4. **Advanced**: cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw, speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw, **lora_adapter_raw**
5. autoplay, forever, live_update

太字が v2 で追加される要素。

---

## 閲覧UI（Streamlit）拡張

### v1 から維持
- 一覧表示・カード形式
- フィルター / 検索 / ソート / レーティング / お気に入り / メモ

### v2 で追加
- **新カラムの表示**：カード詳細部に `duration_scale / seconds / t_schedule_mode / sway_coeff / lora_adapter / speaker_embedding / cfg_scale_speaker / ui_version / model_version` を表示
- 古いレコード（新カラムが `NULL`）は **「(不明)」または「-」表示**で無理に埋めない
- **フィルター追加（任意）**：`model_version` でフィルタ、`t_schedule_mode='sway'` だけ表示など。実装優先度は低い

---

## バージョニング規約

### `MY_UI_VERSION` 定数
- `my/__init__.py` に `MY_UI_VERSION: str = "v3-myui-1"` のように定義
- 命名規則：`{upstream_version}-myui-{my_iteration}`
  - 例：`v3-myui-1`（v3 対応の myUI 第1版）/ `v3-myui-2`（不具合修正後）
- Phase 完了時に上げるかどうかは PR レビューで判断

### `model_version` の文字列規約
- `v2`、`v3`、`v2-voicedesign` の3パターン基本
- 推定不能は `NULL`（無理に埋めない）

---

## 後方互換性ポリシー

- **既存 `last_settings_*.json`**：未知キーは `.get(key, default)` で安全フォールバック
- **既存 DB**：起動時に冪等に `ALTER TABLE ADD COLUMN`、既存レコードは新カラム `NULL` で残す
- **Phase A だけ適用された状態でも v3 で動く**ことを保証
- **古い myUI（v1）で書かれたレコードは新閲覧UIで「(不明)」表示で読める**

---

## 検証観点（受け入れ条件）

| 観点 | 期待動作 |
|---|---|
| 起動 | `python -m my.gradio_gen` / `python -m my.gradio_ref` が v3 環境で例外なく起動 |
| 既存 v2 ckpt | v2 チェックポイントで生成成功（Duration Predictor 未搭載 → 30秒フォールバック）|
| 新 v3 ckpt（参照版） | `Aratako/Irodori-TTS-500M-v3` で `seconds` 空欄 → 自動長で生成成功 |
| Sway Sampling | `num_steps=6, t_schedule_mode=sway, sway_coeff=-1.0` で品質劣化が少ないこと |
| Duration Scale | `1.2` で長め、`0.8` で短めになること |
| LoRA | 既存 LoRA アダプタディレクトリで声質切替成功 |
| 絵文字パレット | text 入力に挿入され、Live Update 経由でも反映 |
| Speaker Inversion | `.speaker.safetensors` をアップロード → 学習済み話者で生成 |
| Speaker Inversion 排他 | Reference Audio と同時指定で明示的にエラー |
| DB マイグレーション | 旧 DB を持つ環境で起動 → カラム自動追加・既存レコード保持 |
| 閲覧UI 表示 | 旧レコード「(不明)」、新レコード新カラム表示 |
| Generate Forever | 新パラメータがスナップショット固定で動く（途中変更反映しない） |
| 設定永続化 | `last_settings_*.json` に新キーが入り、再起動時に復元される |

---

## 参考リンク

- 本家 v2→v3 差分まとめ: [../v2_to_v3_changes.md](../v2_to_v3_changes.md)
- v1 仕様: [../myui-v1/spec.md](../myui-v1/spec.md)
- v1 TODO: [../myui-v1/TODO.md](../myui-v1/TODO.md)
- 本家 VoiceDesign UI: [gradio_app_voicedesign.py](../../../gradio_app_voicedesign.py)
- 本家 参照版 UI: [gradio_app.py](../../../gradio_app.py)
- 推論ランタイム: [irodori_tts/inference_runtime.py](../../../irodori_tts/inference_runtime.py)
- 絵文字パレット: [irodori_tts/gradio_emoji_palette.py](../../../irodori_tts/gradio_emoji_palette.py)
- Speaker Inversion: [irodori_tts/speaker_inversion.py](../../../irodori_tts/speaker_inversion.py)
- Duration Predictor: [irodori_tts/duration.py](../../../irodori_tts/duration.py)
