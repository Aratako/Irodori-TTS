#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from character_profile_loader import load_character_config
from conversation_engine import CharacterProfile, ConversationTurn
from llm_config import LLMConfig
from openai_conversation_engine import OpenAIConversationEngine
from openai_transcription_engine import OpenAITranscriptionEngine
from voice_engine import VoiceEngine


BASE_DIR = Path(__file__).resolve().parent

CHARACTER_PROFILE_PATH = BASE_DIR / "character_profile.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradio_character_chat"


ChatMessage = dict[str, str]
HistoryItem = dict[str, str]
SessionSettings = dict[str, str]

SUPPORTED_REFERENCE_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


@dataclass
class AppResources:
    llm_config: LLMConfig
    voice_engine: VoiceEngine
    transcription_engine: OpenAITranscriptionEngine
    initial_settings: SessionSettings


def _create_session_settings(
    profile: CharacterProfile,
    reference_audio: str | Path,
) -> SessionSettings:
    return {
        "name": profile.name,
        "first_person": profile.first_person,
        "personality": profile.personality,
        "speaking_style": profile.speaking_style,
        "reference_audio_path": str(Path(reference_audio).resolve()),
    }


def _load_resources() -> AppResources:
    character_config = load_character_config(CHARACTER_PROFILE_PATH)
    llm_config = LLMConfig.from_environment()

    voice_engine = VoiceEngine(
        reference_audio=character_config.voice.reference_audio,
        output_dir=OUTPUT_DIR,
    )
    voice_engine.load()

    return AppResources(
        llm_config=llm_config,
        voice_engine=voice_engine,
        transcription_engine=OpenAITranscriptionEngine(llm_config),
        initial_settings=_create_session_settings(
            character_config.profile,
            character_config.voice.reference_audio,
        ),
    )


def _settings_state_or_initial(
    settings: SessionSettings | None,
    resources: AppResources,
) -> SessionSettings:
    if not isinstance(settings, dict):
        return dict(resources.initial_settings)

    normalized = dict(resources.initial_settings)

    for key in normalized:
        value = settings.get(key)

        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()

    return normalized


def _session_settings_to_profile(settings: SessionSettings) -> CharacterProfile:
    return CharacterProfile(
        name=settings["name"],
        first_person=settings["first_person"],
        personality=settings["personality"],
        speaking_style=settings["speaking_style"],
    )


def _audio_value_to_path(reference_audio: object) -> Path:
    if reference_audio is None:
        raise ValueError("参照音声ファイルを指定してください。")

    if isinstance(reference_audio, str | Path):
        audio_path_text = str(reference_audio)

    elif isinstance(reference_audio, dict):
        audio_path_text = str(
            reference_audio.get("path")
            or reference_audio.get("name")
            or ""
        )

    elif hasattr(reference_audio, "name"):
        audio_path_text = str(reference_audio.name)

    else:
        audio_path_text = str(reference_audio)

    audio_path_text = audio_path_text.strip()

    if not audio_path_text:
        raise ValueError("参照音声ファイルを指定してください。")

    return Path(audio_path_text)


def _validate_session_settings(
    name: str,
    first_person: str,
    personality: str,
    speaking_style: str,
    reference_audio: object,
) -> SessionSettings:
    cleaned_values = {
        "name": str(name or "").strip(),
        "first_person": str(first_person or "").strip(),
        "personality": str(personality or "").strip(),
        "speaking_style": str(speaking_style or "").strip(),
    }
    empty_fields = [
        label
        for key, label in (
            ("name", "キャラクター名"),
            ("first_person", "一人称"),
            ("personality", "性格"),
            ("speaking_style", "話し方"),
        )
        if not cleaned_values[key]
    ]

    if empty_fields:
        raise ValueError(
            "未入力の設定があります: " + ", ".join(empty_fields)
        )

    reference_audio_path = _audio_value_to_path(reference_audio)

    if not reference_audio_path.exists():
        raise FileNotFoundError(
            "参照音声ファイルが見つかりません。\n"
            f"確認する場所: {reference_audio_path}"
        )

    if not reference_audio_path.is_file():
        raise ValueError(
            "参照音声には通常のファイルを指定してください。\n"
            f"確認する場所: {reference_audio_path}"
        )

    if reference_audio_path.stat().st_size == 0:
        raise ValueError("空の参照音声ファイルは使用できません。")

    audio_extension = reference_audio_path.suffix.lower()

    if audio_extension not in SUPPORTED_REFERENCE_AUDIO_EXTENSIONS:
        supported_extensions = ", ".join(
            sorted(SUPPORTED_REFERENCE_AUDIO_EXTENSIONS)
        )
        raise ValueError(
            "参照音声の形式が対応していません。\n"
            f"対応拡張子: {supported_extensions}"
        )

    return {
        **cleaned_values,
        "reference_audio_path": str(reference_audio_path.resolve()),
    }


def _turns_to_history_state(history: list[ConversationTurn]) -> list[HistoryItem]:
    return [
        {
            "user_text": turn.user_text,
            "character_text": turn.character_text,
        }
        for turn in history
    ]


def _history_state_to_turns(history: list[HistoryItem] | None) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []

    for item in history or []:
        user_text = str(item.get("user_text", "")).strip()
        character_text = str(item.get("character_text", "")).strip()

        if user_text and character_text:
            turns.append(
                ConversationTurn(
                    user_text=user_text,
                    character_text=character_text,
                )
            )

    return turns


def _build_chat_messages(history: list[ConversationTurn]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []

    for turn in history:
        messages.append(
            {
                "role": "user",
                "content": turn.user_text,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": turn.character_text,
            }
        )

    return messages


def _create_conversation_engine(
    resources: AppResources,
    profile: CharacterProfile,
    history: list[ConversationTurn],
) -> OpenAIConversationEngine:
    engine = OpenAIConversationEngine(
        profile=profile,
        config=resources.llm_config,
    )
    engine._history = list(history)
    return engine


def _submit_message(
    user_text: str,
    history: list[HistoryItem] | None,
    settings: SessionSettings | None,
    resources: AppResources,
):
    current_history = _history_state_to_turns(history)
    current_settings = _settings_state_or_initial(settings, resources)
    cleaned_text = str(user_text or "").strip()

    if not cleaned_text:
        yield (
            "",
            _build_chat_messages(current_history),
            _turns_to_history_state(current_history),
            gr.update(value=None),
            "文章を入力してください。",
        )
        return

    processing_messages = [
        *_build_chat_messages(current_history),
        {
            "role": "user",
            "content": cleaned_text,
        },
    ]
    yield (
        cleaned_text,
        processing_messages,
        _turns_to_history_state(current_history),
        gr.update(value=None),
        "OpenAIの返答を生成しています...",
    )

    try:
        profile = _session_settings_to_profile(current_settings)
        conversation_engine = _create_conversation_engine(
            resources,
            profile,
            current_history,
        )
        reply = conversation_engine.generate_reply(cleaned_text)

    except Exception as error:
        yield (
            cleaned_text,
            _build_chat_messages(current_history),
            _turns_to_history_state(current_history),
            gr.update(value=None),
            f"OpenAIの返答生成に失敗しました: {error}",
        )
        return

    updated_history = list(conversation_engine.history)
    updated_history_state = _turns_to_history_state(updated_history)
    chat_messages = _build_chat_messages(updated_history)

    yield (
        "",
        chat_messages,
        updated_history_state,
        gr.update(value=None),
        "返答を生成しました。音声を生成しています...",
    )

    try:
        voice_result = resources.voice_engine.generate(
            reply.text,
            reference_audio=current_settings["reference_audio_path"],
        )

    except Exception as error:
        yield (
            "",
            chat_messages,
            updated_history_state,
            gr.update(value=None),
            f"返答は生成できましたが、音声生成に失敗しました: {error}",
        )
        return

    status = (
        "返答と音声を生成しました。\n"
        f"音声生成時間: {voice_result.generation_seconds:.3f}秒\n"
        f"使用Seed: {voice_result.used_seed}"
    )

    yield (
        "",
        chat_messages,
        updated_history_state,
        gr.update(value=str(voice_result.output_path)),
        status,
    )


def _clear_conversation() -> tuple[
    list[ChatMessage],
    list[HistoryItem],
    object,
    object,
    str,
]:
    return (
        [],
        [],
        gr.update(value=None),
        gr.update(value=None),
        "会話履歴をクリアしました。",
    )


def _transcribe_microphone_audio(
    microphone_audio: str | Path | None,
    current_user_text: str,
    resources: AppResources,
) -> tuple[str, str]:
    previous_text = str(current_user_text or "")

    try:
        transcribed_text = resources.transcription_engine.transcribe(microphone_audio)

    except Exception as error:
        return (
            previous_text,
            f"文字起こしに失敗しました: {error}",
        )

    return (
        transcribed_text,
        "文字起こししました。内容を確認・修正してから送信してください。",
    )


def _transcribe_microphone_audio_with_status(
    microphone_audio: str | Path | None,
    current_user_text: str,
    resources: AppResources,
) -> Iterator[tuple[str, str]]:
    yield (
        str(current_user_text or ""),
        "文字起こし中...",
    )

    yield _transcribe_microphone_audio(
        microphone_audio,
        current_user_text,
        resources,
    )


def _apply_session_settings(
    name: str,
    first_person: str,
    personality: str,
    speaking_style: str,
    reference_audio: object,
    settings: SessionSettings | None,
    history: list[HistoryItem] | None,
    resources: AppResources,
) -> tuple[
    SessionSettings,
    list[ChatMessage],
    list[HistoryItem],
    object,
    object,
    str,
]:
    current_settings = _settings_state_or_initial(settings, resources)
    current_history = _history_state_to_turns(history)

    try:
        updated_settings = _validate_session_settings(
            name=name,
            first_person=first_person,
            personality=personality,
            speaking_style=speaking_style,
            reference_audio=reference_audio,
        )

    except Exception as error:
        return (
            current_settings,
            _build_chat_messages(current_history),
            _turns_to_history_state(current_history),
            gr.update(),
            gr.update(),
            f"設定を反映できませんでした: {error}",
        )

    return (
        updated_settings,
        [],
        [],
        gr.update(value=None),
        gr.update(value=None),
        "設定を反映しました。会話履歴と生成音声をクリアしました。",
    )


def build_ui(resources: AppResources) -> gr.Blocks:
    initial_status = (
        "キャラクターとの会話を開始できます。\n"
        "送信するとOpenAIの返答を生成し、その返答をIrodori-TTSで音声化します。"
    )

    with gr.Blocks(title="OpenAI Character Chat - Irodori-TTS") as demo:
        gr.Markdown("# OpenAI Character Chat")

        chat_history = gr.State([])
        session_settings = gr.State(dict(resources.initial_settings))

        def submit_message(
            user_text: str,
            history: list[HistoryItem] | None,
            settings: SessionSettings | None,
        ):
            yield from _submit_message(user_text, history, settings, resources)

        def apply_session_settings(
            name: str,
            first_person: str,
            personality: str,
            speaking_style: str,
            reference_audio: object,
            settings: SessionSettings | None,
            history: list[HistoryItem] | None,
        ):
            return _apply_session_settings(
                name=name,
                first_person=first_person,
                personality=personality,
                speaking_style=speaking_style,
                reference_audio=reference_audio,
                settings=settings,
                history=history,
                resources=resources,
            )

        def transcribe_microphone_audio(
            microphone_audio: str | Path | None,
            current_user_text: str,
        ) -> Iterator[tuple[str, str]]:
            yield from _transcribe_microphone_audio_with_status(
                microphone_audio,
                current_user_text,
                resources,
            )

        with gr.Accordion("キャラクター設定", open=True):
            with gr.Row():
                character_name = gr.Textbox(
                    label="キャラクター名",
                    value=resources.initial_settings["name"],
                    scale=2,
                )
                first_person = gr.Textbox(
                    label="一人称",
                    value=resources.initial_settings["first_person"],
                    scale=1,
                )

            personality = gr.Textbox(
                label="性格",
                value=resources.initial_settings["personality"],
                lines=2,
            )
            speaking_style = gr.Textbox(
                label="話し方",
                value=resources.initial_settings["speaking_style"],
                lines=2,
            )
            reference_audio = gr.Audio(
                label="参照音声ファイル",
                value=resources.initial_settings["reference_audio_path"],
                sources=["upload"],
                type="filepath",
                interactive=True,
            )
            apply_settings_button = gr.Button("設定を反映")

        chatbot = gr.Chatbot(
            label="キャラクターとの会話",
            height=420,
        )

        with gr.Row():
            user_input = gr.Textbox(
                label="あなたの文章",
                placeholder="キャラクターに話しかける文章を入力してください",
                lines=3,
                scale=5,
            )
            send_button = gr.Button("送信", variant="primary", scale=1)

        with gr.Row():
            microphone_audio = gr.Audio(
                label="マイク録音",
                sources=["microphone"],
                type="filepath",
                interactive=True,
                scale=5,
            )
            transcribe_button = gr.Button("文字起こし", scale=1)

        generated_audio = gr.Audio(
            label="生成音声",
            type="filepath",
            interactive=False,
            autoplay=True,
        )
        status = gr.Textbox(
            label="状態",
            value=initial_status,
            lines=4,
            interactive=False,
        )
        clear_button = gr.Button("会話クリア")

        send_button.click(
            submit_message,
            inputs=[
                user_input,
                chat_history,
                session_settings,
            ],
            outputs=[
                user_input,
                chatbot,
                chat_history,
                generated_audio,
                status,
            ],
        )
        user_input.submit(
            submit_message,
            inputs=[
                user_input,
                chat_history,
                session_settings,
            ],
            outputs=[
                user_input,
                chatbot,
                chat_history,
                generated_audio,
                status,
            ],
        )
        apply_settings_button.click(
            apply_session_settings,
            inputs=[
                character_name,
                first_person,
                personality,
                speaking_style,
                reference_audio,
                session_settings,
                chat_history,
            ],
            outputs=[
                session_settings,
                chatbot,
                chat_history,
                generated_audio,
                microphone_audio,
                status,
            ],
        )
        transcribe_button.click(
            transcribe_microphone_audio,
            inputs=[
                microphone_audio,
                user_input,
            ],
            outputs=[
                user_input,
                status,
            ],
        )
        microphone_audio.stop_recording(
            transcribe_microphone_audio,
            inputs=[
                microphone_audio,
                user_input,
            ],
            outputs=[
                user_input,
                status,
            ],
        )
        clear_button.click(
            _clear_conversation,
            outputs=[
                chatbot,
                chat_history,
                generated_audio,
                microphone_audio,
                status,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio app for OpenAI character chat with Irodori-TTS voice output."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    resources = _load_resources()
    demo = build_ui(resources)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
