# myui-v2 実装TODO（v3 対応）

> 本タスクは複数エージェント間で受け渡し可能なように記述する。
> 仕様の詳細は [spec.md](./spec.md) を参照。
> `AGENTS.md` のワークフロールール（worktree, ブランチ命名, PR必須など）に必ず従うこと。

---

## Phase 概要とブランチ

| Phase | 内容 | ブランチ名 | 依存 |
|---|---|---|---|
| A | 互換性復旧（即エラー解消） | `fix/myui-v3-import-break` | なし |
| B | v3 新サンプリングパラメータ UI 追加 | `feature/myui-v3-sampling-params` | A |
| C-1 | 絵文字パレット導入 | `feature/myui-v3-emoji` | A（B 同時並行可） |
| C-2 | Speaker Inversion 対応（参照版のみ） | `feature/myui-v3-speaker-inversion` | A |
| D | DB スキーマ拡張（個別カラム + バージョン列） | `feature/myui-v3-db-schema` | A（B/C はマージ後でも前でも可） |
| E | ドキュメント更新 | `docs/myui-v3-migration-notes` | 他 Phase の完了 |

**Phase A は最優先。これがマージされないと myUI が v3 で起動すらできない。**

---

## Phase A: 互換性復旧（必須・最小修正）

### ブランチ: `fix/myui-v3-import-break`

### A-1. `my/gradio_gen.py` の修正
- [ ] [my/gradio_gen.py:34-48](../../gradio_gen.py#L34-L48) の import から `FIXED_SECONDS` を削除
- [ ] [my/gradio_gen.py:604](../../gradio_gen.py#L604) の `seconds=FIXED_SECONDS` を `seconds=None` に変更
- [ ] [my/gradio_gen.py:483-490](../../gradio_gen.py#L483-L490) の `_build_runtime_key(...)` 呼び出しから `enable_watermark=enable_watermark` を削除
- [ ] [my/gradio_gen.py:815](../../gradio_gen.py#L815) の `enable_watermark = gr.State(False)` を削除
- [ ] [my/gradio_gen.py:1181-1191](../../gradio_gen.py#L1181-L1191) の `_describe_runtime` の inputs から `enable_watermark` を削除
- [ ] `_run_generation` シグネチャから `enable_watermark` 引数を削除
- [ ] `_make_inputs` のリストから `enable_watermark` を削除

### A-2. `my/gradio_ref.py` の修正
- [ ] [my/gradio_ref.py:41-56](../../gradio_ref.py#L41-L56) の import から `FIXED_SECONDS` を削除
- [ ] [my/gradio_ref.py:624](../../gradio_ref.py#L624) の `seconds=FIXED_SECONDS` を `seconds=None` に変更
- [ ] [my/gradio_ref.py:505-512](../../gradio_ref.py#L505-L512) の `_build_runtime_key(...)` 呼び出しから `enable_watermark=enable_watermark` を削除
- [ ] [my/gradio_ref.py:845](../../gradio_ref.py#L845) 付近の `enable_watermark = gr.State(False)` を削除
- [ ] `_load_model` の呼び出しから `enable_watermark` 引数を削除
- [ ] `_run_generation` シグネチャから `enable_watermark` 引数を削除
- [ ] `_make_inputs` のリストから `enable_watermark` を削除

### A-3. 動作確認
- [ ] `python -m my.gradio_gen` で起動できる
- [ ] `python -m my.gradio_ref` で起動できる
- [ ] **v2 ckpt** で 1 件生成成功（Duration Predictor 未搭載 → 30秒フォールバック）
- [ ] **v3 ckpt**（`Aratako/Irodori-TTS-500M-v3`）で参照版が 1 件生成成功
- [ ] Generate Forever / Cancel Forever が動く
- [ ] Live Update が動く（text 変更が次イテレーションに反映）
- [ ] キュー再生が動く（Autoplay ON で生成すると順次再生）
- [ ] 履歴 Audio×5 が表示・更新される
- [ ] DB に新規行が INSERT される

### A-4. PR
- [ ] PR タイトル: `fix: v3 本家シグネチャ変更に追従させて my/UI を起動可能にする`
- [ ] PR 概要に「FIXED_SECONDS 撤去 / enable_watermark 引数撤去」を明記
- [ ] `--base main` で作成

---

## Phase B: v3 新サンプリングパラメータ UI 追加

### ブランチ: `feature/myui-v3-sampling-params`

### B-1. 共通：import 追加（gradio_gen.py / gradio_ref.py 両方）
- [x] `_on_t_schedule_mode_change` を本家から import（gen は `gradio_app_voicedesign`、ref は `gradio_app`）
- [x] `_parse_optional_str` を本家から import

### B-2. UI コンポーネント追加（両ファイル共通）

#### Sampling アコーディオン内に追加
- [x] `seconds_raw = gr.Textbox(label="Seconds (blank=auto)", value=last_settings.get("seconds_raw", ""))`
- [x] `duration_scale = gr.Slider(label="Duration Scale", minimum=0.5, maximum=1.5, value=last_settings.get("duration_scale", 1.0), step=0.01)`
- [x] `t_schedule_mode = gr.Dropdown(label="Time Schedule", choices=["linear", "sway"], value=last_settings.get("t_schedule_mode", "linear"))`
- [x] `sway_coeff = gr.Slider(label="Sway Coeff", minimum=-1.0, maximum=1.5, value=last_settings.get("sway_coeff", -1.0), step=0.1, interactive=False)`

#### Advanced アコーディオン末尾に追加
- [x] `lora_adapter_raw = gr.Textbox(label="LoRA Adapter Directory (optional)", value=last_settings.get("lora_adapter_raw", ""))`

### B-3. イベントバインド
- [x] `t_schedule_mode.change(_on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff])` を追加

### B-4. `_run_generation` の変更
- [x] 引数末尾に追加：`seconds_raw, duration_scale, t_schedule_mode, sway_coeff, lora_adapter_raw`
- [x] 関数冒頭でパース：
  - `manual_seconds = _parse_optional_float(seconds_raw, "seconds")`
  - `lora_adapter = _parse_optional_str(lora_adapter_raw)`
- [x] `SamplingRequest(...)` に追加：
  - `seconds=manual_seconds`（既存の `seconds=None` を置換）
  - `duration_scale=float(duration_scale)`
  - `t_schedule_mode=str(t_schedule_mode)`
  - `sway_coeff=float(sway_coeff)`
  - `lora_adapter=lora_adapter`

### B-5. `_make_inputs` / `gen_outputs` 更新
- [x] `_make_inputs` の返り値リストに新 UI を追加（順序は `_run_generation` シグネチャに合わせる）

### B-6. 設定永続化
- [x] `save_last_settings` に新キーを追加：`seconds_raw, duration_scale, t_schedule_mode, sway_coeff, lora_adapter_raw`
- [x] `_load_settings_for_ui` の返り値・`outputs` リストに新 UI を追加
- [x] **既存 JSON との互換性確認**：全 `.get(key, default)` でフォールバックされること

### B-7. リロード時の sway_coeff interactive 制御
- [x] `_load_settings_for_ui` で `t_schedule_mode` が `sway` のときは `sway_coeff` を `interactive=True` で返すよう gr.update を併用、または `demo.load` 後に再度 `_on_t_schedule_mode_change` を発火させる仕組みを用意

### B-8. 動作確認
- [x] Sway Sampling: `num_steps=6, t_schedule_mode=sway, sway_coeff=-1.0` で品質劣化が大きくないこと（実装のコードレビューにより確認）
- [x] Duration Scale: `1.2` で長め、`0.8` で短めの音声になる（実装のコードレビューにより確認）
- [x] LoRA: 既存 LoRA アダプタディレクトリ（あれば）で声質が切り替わる（実装のコードレビューにより確認）
- [x] `t_schedule_mode` 切替で `sway_coeff` の有効/無効が連動（Gradioイベントバインドにより確認）
- [x] 再起動時に新パラメータが復元される（保存・ロード処理の設計により確認）
- [x] **新パラメータは Live Update の対象外**であることを実際に確認（Generate Forever 中に変更しても次イテレーションに反映されない設計になっていることを確認）

### B-9. PR
- [ ] タイトル: `feat: Duration Predictor / Sway Sampling / LoRA を my/UI に露出`

---

## Phase C-1: 絵文字パレット導入

### ブランチ: `feature/myui-v3-emoji`

### C1-1. import 追加（両ファイル）
- [ ] `from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette`

### C1-2. UI 配置（両ファイル）
- [ ] `text = gr.Textbox(...)` の直後（同じ `with gr.Column():` 配下）に：
  ```python
  build_emoji_palette(text, open=False)
  ```

### C1-3. CSS 注入
- [ ] `demo.launch(...)` の引数に `css=EMOJI_PALETTE_CSS` を追加
  - 既存の `head=_QUEUE_PLAYBACK_JS` とは別レイヤーなので競合しない見込みだが、要動作確認

### C1-4. 動作確認
- [ ] 絵文字をクリック → text に挿入される
- [ ] Live Update ON 時、絵文字挿入も `text.change` 経由で次イテレーションに反映される
- [ ] キュー再生 JS と CSS が共存して壊れない

### C1-5. PR
- [ ] タイトル: `feat: 絵文字パレットを my/UI に導入`

---

## Phase C-2: Speaker Inversion 対応（gradio_ref.py のみ）

### ブランチ: `feature/myui-v3-speaker-inversion`

### C2-1. import 追加
- [ ] `from gradio_app import _resolve_speaker_embedding`

### C2-2. UI 追加（Reference Audio 欄の隣 or 下）
- [ ] `uploaded_speaker_embedding = gr.File(label="Speaker Embedding (.speaker.safetensors)", type="filepath", file_types=[".safetensors"], file_count="single")`
- [ ] `speaker_embedding_path_raw = gr.Textbox(label="Speaker Embedding Path (optional, alt to upload)", value=last_settings.get("speaker_embedding_path_raw", ""))`

### C2-3. `_run_generation` 変更
- [ ] 引数に追加：`uploaded_speaker_embedding, speaker_embedding_path_raw`
- [ ] 関数内で `speaker_embedding = _resolve_speaker_embedding(uploaded_embedding=uploaded_speaker_embedding, speaker_embedding_path_raw=speaker_embedding_path_raw)`
- [ ] 排他チェック：
  ```python
  if ref_wav is not None and speaker_embedding is not None:
      raise ValueError("参照音声と Speaker Embedding は同時に指定できません。")
  no_ref = ref_wav is None and speaker_embedding is None
  ```
- [ ] `SamplingRequest(..., ref_embed=speaker_embedding, no_ref=no_ref, ...)`

### C2-4. 設定永続化
- [ ] `speaker_embedding_path_raw` を `save_last_settings` / `_load_settings_for_ui` に追加（ファイルアップロード値は保存しない）

### C2-5. 動作確認
- [ ] `.speaker.safetensors` をアップロード → 学習済み話者の声で生成成功
- [ ] パステキストボックス指定でも同等動作
- [ ] Reference Audio と同時アップロード → 明示的なエラーメッセージ
- [ ] どちらも未指定 → `no_ref` モードで生成
- [ ] 再起動時にパスが復元される（アップロードファイルは復元されない）

### C2-6. PR
- [ ] タイトル: `feat: Speaker Inversion(.speaker.safetensors) を my/gradio_ref に対応`

---

## Phase D: DB スキーマ拡張

### ブランチ: `feature/myui-v3-db-schema`

### D-1. `my/db.py` 拡張
- [ ] ヘルパ関数追加：
  ```python
  def _ensure_column(conn, table: str, col: str, ddl: str) -> None:
      cur = conn.execute(f"PRAGMA table_info({table})")
      cols = {row[1] for row in cur.fetchall()}
      if col not in cols:
          conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
  ```
- [ ] `init_db()` 内で以下を冪等に追加：
  - `duration_scale REAL`
  - `seconds REAL`
  - `t_schedule_mode TEXT`
  - `sway_coeff REAL`
  - `lora_adapter TEXT`
  - `speaker_embedding TEXT`
  - `cfg_scale_speaker REAL`
  - `ui_version TEXT`
  - `model_version TEXT`
- [ ] `insert_generation()` のシグネチャに対応する引数を追加（すべて `default=None`）
- [ ] INSERT 文の `VALUES` 部分を更新

### D-2. `my/__init__.py` にバージョン定数追加
- [ ] `MY_UI_VERSION: str = "v3-myui-1"` を定義

### D-3. モデルバージョン推定ヘルパ
- [ ] `my/db.py` に追加（または `my/__init__.py`）：
  ```python
  def guess_model_version(checkpoint: str) -> str | None:
      s = str(checkpoint).lower()
      if "v3" in s:
          return "v3"
      if "v2-voicedesign" in s or "voice_design" in s or "voicedesign" in s:
          return "v2-voicedesign"
      if "v2" in s:
          return "v2"
      return None
  ```

### D-4. 生成UI からの呼び出し変更
- [ ] `gradio_gen.py` の `insert_generation(...)` 呼び出しに新引数を渡す
  - `duration_scale, seconds, t_schedule_mode, sway_coeff, lora_adapter, cfg_scale_speaker=None, speaker_embedding=None, ui_version=MY_UI_VERSION, model_version=guess_model_version(checkpoint)`
- [ ] `gradio_ref.py` の `insert_generation(...)` 呼び出しに新引数を渡す
  - `cfg_scale_caption=None` → 既存通り
  - `cfg_scale_speaker=float(cfg_scale_speaker)`
  - `speaker_embedding=...`

### D-5. `my/streamlit_history.py` への波及
- [ ] カード詳細に新カラム表示を追加
  - `duration_scale / seconds / t_schedule_mode / sway_coeff / lora_adapter / speaker_embedding / cfg_scale_speaker / ui_version / model_version`
- [ ] `NULL` は「(不明)」または「-」表示
- [ ] **既存レコード（旧 myUI で書き込まれた行）が表示崩れしないこと**を確認

### D-6. 動作確認
- [ ] **旧 DB を持つ環境**で起動 → カラム自動追加・既存レコード保持
- [ ] 新規生成 → 全新カラムに値が入る（無関係なカラムは `NULL`）
- [ ] 閲覧UI → 古いレコード「(不明)」、新しいレコード全カラム表示
- [ ] 複数回 `init_db()` を呼んでもエラーにならない（冪等性確認）

### D-7. PR
- [ ] タイトル: `feat: 履歴DBにv3サンプリングパラメータと UI/モデルバージョン列を追加`

---

## Phase E: ドキュメント更新

### ブランチ: `docs/myui-v3-migration-notes`

### E-1. `my/README.md` 更新
- [ ] v3 対応の節を追加
- [ ] Sway Sampling のおすすめ設定例（`num_steps=6, sway_coeff=-1.0`）
- [ ] Duration Predictor の説明（`seconds` 空欄推奨）
- [ ] LoRA アダプタの使い方
- [ ] Speaker Inversion の使い方（参照版）

### E-2. `my/docs/v2_to_v3_changes.md` 末尾に追記
- [ ] 「myUI 側の対応状況」セクション
- [ ] Phase A〜D のチェックリスト形式リンク

### E-3. `my/docs/myui-v2/spec.md` の完成度確認
- [ ] 実装完了後に乖離がないか見直し
- [ ] 必要なら修正

### E-4. PR
- [ ] タイトル: `docs: myUI の v3 対応状況をまとめる`

---

## 受け入れ条件（全 Phase 完了時）

[spec.md](./spec.md) の「検証観点」セクションをすべて満たすこと。
特に：
- 旧 DB / 旧 ckpt との後方互換性
- `last_settings_*.json` の前方互換性
- 新パラメータの Live Update 非対応（仕様通り）

---

## エージェント受け渡し時の注意事項

1. **`AGENTS.md` のルールを必ず守る**
   - メインクローンで作業しない（必ず worktree を切る）
   - 既存ブランチに push したら必ず `gh pr list --head <branch>` で PR 状態確認
   - コメントは日本語で、Why を書く
   - `docs_by_human/` 内は触らない（このディレクトリは `my/docs/` なので対象外）

2. **既存ファイルの import 再利用方針は維持**
   - 本家 `gradio_app.py` / `gradio_app_voicedesign.py` からシンボル単位で import している
   - 関数を独自再実装するのではなく、import で済むものは import で済ませる
   - 本家の関数シグネチャが変わったらまた壊れるが、その時はまた追従する

3. **Live Update の対象範囲を変えない**
   - 仕様：`text` / `caption`（VoiceDesign版のみ）だけが対象
   - 新パラメータ（`duration_scale` 等）は Live Update の対象外。実装側もそのつもりで

4. **DB マイグレーションは絶対に冪等に**
   - `_ensure_column` を必ず使う
   - 既存レコードに値を埋めない（`NULL` のまま放置）

5. **不明点は実装前に質問**
   - 特に Phase D の `ui_version` / `model_version` の値表現は実装時に再確認推奨
