import asyncio
import logging

from telegram.ext import Application


log = logging.getLogger("bot")


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

