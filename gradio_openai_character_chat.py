#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from character_profile_loader import load_character_config
from conversation_engine import CharacterProfile, ConversationTurn
from llm_config import LLMConfig
from openai_conversation_engine import OpenAIConversationEngine
from voice_engine import VoiceEngine


BASE_DIR = Path(__file__).resolve().parent

CHARACTER_PROFILE_PATH = BASE_DIR / "character_profile.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradio_character_chat"


ChatMessage = dict[str, str]
HistoryItem = dict[str, str]


@dataclass
class AppResources:
    profile: CharacterProfile
    llm_config: LLMConfig
    voice_engine: VoiceEngine


def _load_resources() -> AppResources:
    character_config = load_character_config(CHARACTER_PROFILE_PATH)
    profile = character_config.profile
    llm_config = LLMConfig.from_environment()

    voice_engine = VoiceEngine(
        reference_audio=character_config.voice.reference_audio,
        output_dir=OUTPUT_DIR,
    )
    voice_engine.load()

    return AppResources(
        profile=profile,
        llm_config=llm_config,
        voice_engine=voice_engine,
    )


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
    history: list[ConversationTurn],
) -> OpenAIConversationEngine:
    engine = OpenAIConversationEngine(
        profile=resources.profile,
        config=resources.llm_config,
    )
    engine._history = list(history)
    return engine


def _submit_message(
    user_text: str,
    history: list[HistoryItem] | None,
    resources: AppResources,
):
    current_history = _history_state_to_turns(history)
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
        conversation_engine = _create_conversation_engine(resources, current_history)
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
        voice_result = resources.voice_engine.generate(reply.text)

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


def _clear_conversation() -> tuple[list[ChatMessage], list[HistoryItem], object, str]:
    return (
        [],
        [],
        gr.update(value=None),
        "会話履歴をクリアしました。",
    )


def build_ui(resources: AppResources) -> gr.Blocks:
    initial_status = (
        "キャラクターとの会話を開始できます。\n"
        "送信するとOpenAIの返答を生成し、その返答をIrodori-TTSで音声化します。"
    )

    with gr.Blocks(title="OpenAI Character Chat - Irodori-TTS") as demo:
        gr.Markdown("# OpenAI Character Chat")

        chat_history = gr.State([])

        def submit_message(user_text: str, history: list[HistoryItem] | None):
            yield from _submit_message(user_text, history, resources)

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

        generated_audio = gr.Audio(
            label="生成音声",
            type="filepath",
            interactive=False,
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
            ],
            outputs=[
                user_input,
                chatbot,
                chat_history,
                generated_audio,
                status,
            ],
        )
        clear_button.click(
            _clear_conversation,
            outputs=[
                chatbot,
                chat_history,
                generated_audio,
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
