import asyncio
import logging
import os
import tempfile
import time

from telegram import ReplyParameters, Update
from telegram.ext import ContextTypes

from services.progress import ProgressState, build_status_message


log = logging.getLogger("bot")


def _is_message_for_bot(msg, bot_username: str, bot_id: int) -> bool:
    """Return True if the bot should process this group message."""
    if msg.chat.type not in ["group", "supergroup"]:
        return True

    if not bot_username:
        return False

    bot_mention = f"@{bot_username}".lower()
    if msg.text and msg.text.startswith("/"):
        return True
    if msg.caption and bot_mention in msg.caption.lower():
        return True
    if msg.text and bot_mention in msg.text.lower():
        return True

    if msg.entities and msg.text:
        for entity in msg.entities:
            if entity.type == "mention":
                mention_text = msg.text[entity.offset:entity.offset + entity.length].lower()
                if mention_text == bot_mention:
                    return True

    return bool(
        msg.reply_to_message
        and msg.reply_to_message.from_user
        and msg.reply_to_message.from_user.id == bot_id
    )


def _resolve_target_message(msg):
    if msg.reply_to_message and (
        msg.reply_to_message.voice
        or msg.reply_to_message.audio
        or msg.reply_to_message.video_note
        or msg.reply_to_message.video
    ):
        return msg.reply_to_message
    return msg


async def _get_media_file(target_msg):
    if target_msg.voice:
        return await target_msg.voice.get_file()
    if target_msg.audio:
        return await target_msg.audio.get_file()
    if target_msg.video_note:
        return await target_msg.video_note.get_file()
    if target_msg.video:
        return await target_msg.video.get_file()
    return None


async def _update_status_periodically(status, engine, transcription_done: asyncio.Event, started_at: float):
    """Update Telegram status message while transcription is running."""
    await asyncio.sleep(5)
    tick = 0
    while not transcription_done.is_set():
        try:
            progress = getattr(engine, "progress", ProgressState())
            elapsed = int(time.time() - started_at)
            await status.edit_text(build_status_message(progress, elapsed, tick))
            tick += 1
            await asyncio.sleep(5)
        except Exception as exc:
            log.debug("Status update error: %s", exc)
            break


async def _stop_status_task(status_task, transcription_done: asyncio.Event):
    """Stop the background status updater before posting the final result."""
    transcription_done.set()
    if not status_task:
        return

    try:
        await asyncio.wait_for(status_task, timeout=1.0)
    except asyncio.TimeoutError:
        status_task.cancel()
        try:
            await status_task
        except asyncio.CancelledError:
            pass


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages, audio files, and video notes."""
    user = update.effective_user
    msg = update.message
    if not user or not msg:
        return

    allowed_users = context.application.bot_data.get("allowed_users", [])
    if allowed_users and user.id not in allowed_users:
        await msg.reply_text("Access denied.")
        return

    if msg.chat.type in ["group", "supergroup"]:
        bot_username = context.bot.username
        if not bot_username:
            log.warning("Bot username not available in context, fetching...")
            bot_me = await context.bot.get_me()
            bot_username = bot_me.username

        if not _is_message_for_bot(msg, bot_username or "", context.bot.id):
            log.info("Ignored message in %s: no mention of @%s found", msg.chat.type, bot_username)
            return

    target_msg = _resolve_target_message(msg)
    tg_file = await _get_media_file(target_msg)
    if not tg_file:
        if msg.chat.type in ["group", "supergroup"]:
            await msg.reply_text(
                "I was mentioned, but I can't see the voice message. 🧐\n\n"
                "To fix this, please **make me an Administrator** or disable **Privacy Mode** in @BotFather."
            )
        return

    engine = context.application.bot_data["stt_engine"]
    log.info("Voice from %s (%d)", user.first_name, user.id)
    status = await msg.reply_text(
        "Transcribing...",
        reply_parameters=ReplyParameters(message_id=msg.message_id),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()

    transcription_done = asyncio.Event()
    started_at = time.time()
    status_task = None

    try:
        await tg_file.download_to_drive(tmp.name)
        status_task = asyncio.create_task(
            _update_status_periodically(status, engine, transcription_done, started_at)
        )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.transcribe, tmp.name)
        text = result["text"]

        await _stop_status_task(status_task, transcription_done)
        status_task = None

        if not text.strip():
            await status.edit_text("(no speech detected)")
            return

        footer = f"Engine: {result['engine']} | {result['elapsed']:.1f}s"
        if result["duration"]:
            footer = f"Audio: {result['duration']:.1f}s | {footer}"

        reply = f"{text}\n\n---\n{footer}"
        if len(reply) > 4096:
            await status.edit_text(reply[:4096])
            for i in range(4096, len(reply), 4096):
                await msg.reply_text(
                    reply[i:i + 4096],
                    reply_parameters=ReplyParameters(message_id=msg.message_id),
                )
        else:
            await status.edit_text(reply)

        log.info("Done: %s %.1fs: %s", result["engine"], result["elapsed"], text[:100])
    except Exception as exc:
        log.error("Transcription error: %s", exc, exc_info=True)
        await _stop_status_task(status_task, transcription_done)
        status_task = None
        await status.edit_text(f"Error: {exc}")
    finally:
        await _stop_status_task(status_task, transcription_done)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

