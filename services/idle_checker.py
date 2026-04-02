import asyncio
import logging

from telegram.ext import Application


log = logging.getLogger("bot")

# Import will be set by bot.py to avoid circular import
_queue_processor_func = None


def set_queue_processor(func):
    """Set the queue processor function to be started in post_init."""
    global _queue_processor_func
    _queue_processor_func = func


async def idle_checker_task(app: Application):
    """Background task that periodically unloads idle Whisper models."""
    while True:
        await asyncio.sleep(60)
        engine = app.bot_data.get("stt_engine")
        if engine and hasattr(engine, "unload_if_idle"):
            engine.unload_if_idle()


async def post_init(app: Application):
    """Run after the application is initialized."""
    log.info("Starting background idle checker task...")
    asyncio.create_task(idle_checker_task(app))
    
    if _queue_processor_func:
        log.info("Starting transcription queue processor...")
        asyncio.create_task(_queue_processor_func())

