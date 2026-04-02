import asyncio
import logging
import mimetypes
import os
import subprocess
import tempfile
import time

from telegram import ReplyParameters, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from services.progress import ProgressState, build_status_message
from services.subtitles import build_srt
from services.queue_manager import transcription_queue


log = logging.getLogger("bot")

BOT_API_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".ogg",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
}

SUPPORTED_VIDEO_DOCUMENT_EXTENSIONS = {
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
}


def _guess_suffix(file_name: str | None, mime_type: str | None, default: str = ".bin") -> str:
    """Pick a useful local file suffix for Telegram media downloads."""
    if file_name:
        _, ext = os.path.splitext(file_name)
        if ext:
            return ext.lower()

    if mime_type:
        guessed = mimetypes.guess_extension(mime_type, strict=False)
        if guessed:
            return guessed.lower()

    return default


def _is_supported_document(document) -> bool:
    """Return True for audio/video files sent as Telegram documents."""
    if not document:
        return False

    file_name = getattr(document, "file_name", "") or ""
    mime_type = (getattr(document, "mime_type", "") or "").lower()
    _, ext = os.path.splitext(file_name.lower())

    if mime_type.startswith("audio/") or mime_type.startswith("video/"):
        return True

    return ext in SUPPORTED_DOCUMENT_EXTENSIONS


def _is_video_document(document) -> bool:
    """Return True if a Telegram document should be treated as video."""
    if not document:
        return False

    file_name = getattr(document, "file_name", "") or ""
    mime_type = (getattr(document, "mime_type", "") or "").lower()
    _, ext = os.path.splitext(file_name.lower())

    if mime_type.startswith("video/"):
        return True

    return ext in SUPPORTED_VIDEO_DOCUMENT_EXTENSIONS


def _should_attach_srt(target_msg) -> bool:
    """Attach subtitles only for video inputs."""
    return bool(
        target_msg.video
        or target_msg.video_note
        or _is_video_document(getattr(target_msg, "document", None))
    )


def _needs_ffmpeg_audio_extraction(target_msg) -> bool:
    """Return True when the input is video and should be converted to audio first."""
    return _should_attach_srt(target_msg)


def _get_target_file_name(target_msg) -> str:
    """Get the best filename candidate for a Telegram media message."""
    if target_msg.voice:
        return "voice.ogg"
    if target_msg.audio:
        return target_msg.audio.file_name or "audio.mp3"
    if target_msg.video_note:
        return "video_note.mp4"
    if target_msg.video:
        return target_msg.video.file_name or "video.mp4"
    if target_msg.document and _is_supported_document(target_msg.document):
        return target_msg.document.file_name or "document.bin"
    return "media.bin"


def _get_target_media_size(target_msg) -> int | None:
    """Return Telegram-reported file size for the selected media, if available."""
    for media in (
        getattr(target_msg, "voice", None),
        getattr(target_msg, "audio", None),
        getattr(target_msg, "video_note", None),
        getattr(target_msg, "video", None),
        getattr(target_msg, "document", None),
    ):
        if media and getattr(media, "file_size", None):
            return int(media.file_size)
    return None


def _format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _build_file_too_big_message(size_bytes: int | None = None) -> str:
    """Build a user-facing explanation for Telegram Bot API download limits."""
    size_line = f"Your file size: {_format_size_mb(size_bytes)}\n" if size_bytes else ""
    return (
        "This file is too big for Telegram Bot API download.\n\n"
        f"{size_line}"
        f"Current Bot API getFile limit: {_format_size_mb(BOT_API_DOWNLOAD_LIMIT_BYTES)}\n\n"
        "Try one of these options:\n"
        "• send a smaller/compressed video\n"
        "• trim the video before sending\n"
        "• send audio extracted from the video\n"
        "• run a local telegram-bot-api server if you need bigger downloads"
    )


def _build_srt_name(target_msg) -> str:
    """Build an SRT filename based on the source media name."""
    source_name = _get_target_file_name(target_msg)
    stem, _ = os.path.splitext(source_name)
    stem = stem or "transcription"
    return f"{stem}.srt"


def _extract_audio_from_video(source_path: str) -> str:
    """Extract mono 16 kHz PCM WAV audio from a video file using ffmpeg."""
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output_file.name
    output_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as exc:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        error_output = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"ffmpeg failed to extract audio from video: {error_output}") from exc


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
        or _is_supported_document(msg.reply_to_message.document)
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
    if _is_supported_document(target_msg.document):
        return await target_msg.document.get_file()
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


def _set_engine_progress(engine, progress: ProgressState):
    """Update shared engine progress if the engine exposes a mutable progress field."""
    if hasattr(engine, "progress"):
        engine.progress = progress


async def _process_transcription_task(update: Update, context: ContextTypes.DEFAULT_TYPE, status):
    """Process a single transcription task (internal function called from queue processor)."""
    user = update.effective_user
    msg = update.message
    if not user or not msg:
        return

    target_msg = _resolve_target_message(msg)
    
    # Media file was already validated in handle_voice before adding to queue
    # Just get it again for processing
    tg_file = await _get_media_file(target_msg)
    if not tg_file:
        # This shouldn't happen since we validated before queuing, but handle it gracefully
        await status.edit_text("Error: Media file no longer accessible")
        return

    engine = context.application.bot_data["stt_engine"]
    log.info("Processing voice from %s (%d)", user.first_name, user.id)
    
    await status.edit_text("🔄 Processing...")

    source_file_name = _get_target_file_name(target_msg)
    source_suffix = _guess_suffix(
        source_file_name,
        getattr(target_msg.document, "mime_type", None)
        if getattr(target_msg, "document", None)
        else None,
        default=".ogg",
    )
    tmp = tempfile.NamedTemporaryFile(suffix=source_suffix, delete=False)
    tmp.close()
    prepared_path = tmp.name
    srt_path = None

    transcription_done = asyncio.Event()
    started_at = time.time()
    status_task = None

    try:
        await tg_file.download_to_drive(tmp.name)

        if _needs_ffmpeg_audio_extraction(target_msg):
            _set_engine_progress(engine, ProgressState(stage="extracting-audio"))
            status_task = asyncio.create_task(
                _update_status_periodically(status, engine, transcription_done, started_at)
            )
            await status.edit_text(build_status_message(engine.progress, int(time.time() - started_at), 0))

        if _needs_ffmpeg_audio_extraction(target_msg):
            loop = asyncio.get_running_loop()
            prepared_path = await loop.run_in_executor(None, _extract_audio_from_video, tmp.name)

        if not status_task:
            status_task = asyncio.create_task(
                _update_status_periodically(status, engine, transcription_done, started_at)
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.transcribe, prepared_path)
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

        segments = result.get("segments") or []
        if _should_attach_srt(target_msg) and segments:
            srt_content = build_srt(segments)
            if srt_content.strip():
                with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w", encoding="utf-8") as srt_file:
                    srt_file.write(srt_content)
                    srt_path = srt_file.name

                with open(srt_path, "rb") as srt_stream:
                    await msg.reply_document(
                        document=srt_stream,
                        filename=_build_srt_name(target_msg),
                        reply_parameters=ReplyParameters(message_id=msg.message_id),
                    )

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
        if prepared_path != tmp.name:
            try:
                os.unlink(prepared_path)
            except OSError:
                pass
        if srt_path:
            try:
                os.unlink(srt_path)
            except OSError:
                pass


async def _queue_processor():
    """Background task that processes transcription queue sequentially."""
    while True:
        try:
            task = await transcription_queue.get_next_task()
            if task:
                log.info("Processing task from queue (position was %d)", task.position)
                try:
                    await _process_transcription_task(task.update, task.context, task.status_message)
                except Exception as exc:
                    log.error("Error processing queued task: %s", exc, exc_info=True)
                    if task.status_message:
                        try:
                            await task.status_message.edit_text(f"Error: {exc}")
                        except Exception:
                            pass
                finally:
                    await transcription_queue.mark_task_complete()
                    # Update status messages for remaining items in queue
                    await transcription_queue.update_queue_status_messages()
            else:
                await asyncio.sleep(0.5)
        except Exception as exc:
            log.error("Queue processor error: %s", exc, exc_info=True)
            await asyncio.sleep(1)


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
    target_size = _get_target_media_size(target_msg)
    if target_size and target_size > BOT_API_DOWNLOAD_LIMIT_BYTES:
        await msg.reply_text(
            _build_file_too_big_message(target_size),
            reply_parameters=ReplyParameters(message_id=msg.message_id),
        )
        return

    # Check if there's actually a media file to process BEFORE adding to queue
    try:
        tg_file = await _get_media_file(target_msg)
    except BadRequest as exc:
        if "File is too big" in str(exc):
            await msg.reply_text(
                _build_file_too_big_message(target_size),
                reply_parameters=ReplyParameters(message_id=msg.message_id),
            )
            return
        raise

    if not tg_file:
        # No valid media file found - don't add to queue
        if msg.chat.type in ["group", "supergroup"]:
            await msg.reply_text(
                "I was mentioned, but I can't see the voice message. 🧐\n\n"
                "To fix this, please **make me an Administrator** or disable **Privacy Mode** in @BotFather."
            )
        return

    log.info("Voice message from %s (%d)", user.first_name, user.id)
    
    # Add to queue only if we have a valid media file
    queue_position = transcription_queue.get_queue_size()
    
    if queue_position == 0 and not transcription_queue.is_processing():
        status = await msg.reply_text(
            "🔄 Processing...",
            reply_parameters=ReplyParameters(message_id=msg.message_id),
        )
    else:
        status = await msg.reply_text(
            f"⏳ Queued for processing\nPosition: {queue_position + 1}",
            reply_parameters=ReplyParameters(message_id=msg.message_id),
        )
    
    await transcription_queue.add_task(update, context, status)
    await transcription_queue.update_queue_status_messages()


