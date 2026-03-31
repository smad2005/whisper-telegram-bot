#!/usr/bin/env python3
"""
Whisper / Gemini Telegram Bot
Transcribes voice messages, audio files, and video notes using
either a local Whisper model or the Gemini API.

All configuration is done via environment variables — see .env.example
"""

import logging
import sys

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import load_config
from engines.gemini_engine import GeminiEngine
from engines.whisper_engine import WhisperEngine
from handlers.commands import cmd_start
from handlers.transcription import handle_voice
from services.idle_checker import post_init


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")


def build_engine(config):
    """Create the configured speech-to-text engine."""
    if config.engine == "gemini":
        return GeminiEngine(config.gemini)
    return WhisperEngine(config.whisper)


def main():
    """Initialize the STT engine and start the Telegram bot."""
    try:
        config = load_config()
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    app = (
        Application.builder()
        .token(config.telegram.token)
        .post_init(post_init)
        .read_timeout(config.telegram.polling_timeout)
        .write_timeout(30)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    app.bot_data["config"] = config
    app.bot_data["allowed_users"] = config.telegram.allowed_users
    app.bot_data["stt_engine"] = build_engine(config)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler(["t", "transcribe"], handle_voice))
    app.add_handler(
        MessageHandler(
            filters.VOICE
            | filters.AUDIO
            | filters.VIDEO_NOTE
            | filters.VIDEO
            | (filters.TEXT & (~filters.COMMAND)),
            handle_voice,
        )
    )

    log.info("Bot started (%s) with long polling (%ss timeout)", config.engine, config.telegram.polling_timeout)
    app.run_polling(
        drop_pending_updates=False,
        poll_interval=0.0,
        timeout=config.telegram.polling_timeout,
    )


if __name__ == "__main__":
    main()
