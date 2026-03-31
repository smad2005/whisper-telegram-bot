#!/usr/bin/env python3
"""
Whisper / Gemini Telegram Bot
Transcribes voice messages, audio files, and video notes using
either a local Whisper model or the Gemini API.

All configuration is done via environment variables — see .env.example
"""

import logging
import os
import sys
import tempfile
import time

from telegram import Update, ReplyParameters
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")

stt_engine = None
allowed_users = []


class WhisperEngine:
    """Offline speech-to-text engine powered by faster-whisper."""

    def __init__(self):
        from faster_whisper import WhisperModel

        self.model_path = os.getenv("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
        self.model2_path = os.getenv("WHISPER_MODEL2")
        self.special_lang = os.getenv("WHISPER_SPECIAL_LANG", "he")
        
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute = os.getenv("WHISPER_COMPUTE", "int8")
        
        log.info("Loading Whisper model: %s (%s/%s)", self.model_path, self.device, self.compute)
        t0 = time.time()
        self.model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute)
        self.model2 = None  # Lazy load
        
        lang = os.getenv("WHISPER_LANGUAGE", "auto")
        self.language = None if lang == "auto" else lang
        self.beam = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
        log.info("Whisper loaded in %.1fs", time.time() - t0)

    def transcribe(self, path: str) -> dict:
        """Transcribe an audio file and return text with metadata."""
        t0 = time.time()
        segs, info = self.model.transcribe(
            path,
            language=self.language,
            beam_size=self.beam,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )

        detected_lang = info.language
        engine_name = "whisper"

        # Check if we should switch to the special model
        if self.model2_path and detected_lang == self.special_lang:
            log.info("Switching to special model %s for language: %s", self.model2_path, detected_lang)
            if not self.model2:
                from faster_whisper import WhisperModel
                t_load = time.time()
                self.model2 = WhisperModel(self.model2_path, device=self.device, compute_type=self.compute)
                log.info("Special model loaded in %.1fs", time.time() - t_load)
            
            segs, info = self.model2.transcribe(
                path,
                language=detected_lang,
                beam_size=self.beam,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            )
            engine_name = f"whisper-special ({detected_lang})"

        texts = [s.text.strip() for s in segs]
        return {
            "text": " ".join(texts),
            "duration": info.duration,
            "elapsed": time.time() - t0,
            "engine": engine_name,
        }


class GeminiEngine:
    """Cloud-based speech-to-text engine powered by Google Gemini API."""

    def __init__(self):
        from google import genai

        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key == "YOUR_GEMINI_API_KEY":
            log.error("GEMINI_API_KEY is not set! Check your .env file.")
            sys.exit(1)
        self.client = genai.Client(api_key=key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        lang = os.getenv("GEMINI_LANGUAGE", "auto")
        self.language = "Detect language" if lang == "auto" else lang
        log.info("Gemini ready: %s (%s)", self.model, self.language)

    def transcribe(self, path: str) -> dict:
        """Send audio to Gemini API and return transcription with metadata."""
        from google.genai import types

        t0 = time.time()
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mimes = {
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                f"Transcribe this audio. Language: {self.language}. "
                f"Output ONLY the transcription, nothing else.",
                types.Part.from_bytes(
                    data=data, mime_type=mimes.get(ext, "audio/ogg")
                ),
            ],
        )
        return {
            "text": resp.text.strip(),
            "duration": 0,
            "elapsed": time.time() - t0,
            "engine": "gemini",
        }


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command — greet the user."""
    engine = os.getenv("STT_ENGINE", "whisper")
    await update.message.reply_text(
        f"Send me a voice message and I will transcribe it.\n"
        f"Engine: {engine}"
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages, audio files, and video notes."""
    user = update.effective_user
    if allowed_users and user.id not in allowed_users:
        await update.message.reply_text("Access denied.")
        return

    msg = update.message
    if msg.voice:
        file = await msg.voice.get_file()
    elif msg.audio:
        file = await msg.audio.get_file()
    elif msg.video_note:
        file = await msg.video_note.get_file()
    elif msg.video:
        file = await msg.video.get_file()
    else:
        return

    log.info("Voice from %s (%d)", user.first_name, user.id)
    status = await msg.reply_text(
        "Transcribing...",
        reply_parameters=ReplyParameters(message_id=msg.message_id),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()

    try:
        await file.download_to_drive(tmp.name)
        r = stt_engine.transcribe(tmp.name)
        text = r["text"]

        if not text.strip():
            await status.edit_text("(no speech detected)")
            return

        footer = f"Engine: {r['engine']} | {r['elapsed']:.1f}s"
        if r["duration"]:
            footer = f"Audio: {r['duration']:.1f}s | " + footer

        reply = f"{text}\n\n---\n{footer}"
        if len(reply) > 4096:
            await status.edit_text(reply[:4096])
            for i in range(4096, len(reply), 4096):
                await msg.reply_text(
                    reply[i : i + 4096],
                    reply_parameters=ReplyParameters(message_id=msg.message_id),
                )
        else:
            await status.edit_text(reply)

        log.info("Done: %s %.1fs: %s", r["engine"], r["elapsed"], text[:100])

    except Exception as e:
        log.error("Transcription error: %s", e, exc_info=True)
        await status.edit_text(f"Error: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def main():
    """Initialize the STT engine and start the Telegram bot."""
    global stt_engine, allowed_users

    token = os.getenv("BOT_TOKEN", "")
    if not token or token == "YOUR_BOT_TOKEN":
        log.error("BOT_TOKEN is not set! Check your .env file.")
        sys.exit(1)

    au = os.getenv("ALLOWED_USERS", "").split("#")[0].strip()
    if au:
        try:
            allowed_users = [int(x.strip()) for x in au.split(",") if x.strip()]
        except ValueError as e:
            log.error("Invalid user ID in ALLOWED_USERS: %s", e)
            sys.exit(1)

    engine = os.getenv("STT_ENGINE", "whisper")
    if engine == "gemini":
        stt_engine = GeminiEngine()
    else:
        stt_engine = WhisperEngine()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE | filters.VIDEO,
            handle_voice,
        )
    )

    log.info("Bot started (%s)", engine)
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
