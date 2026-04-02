"""
Queue manager for sequential message processing.
Ensures only one transcription task runs at a time due to VRAM limitations.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from telegram import Message, Update
from telegram.ext import ContextTypes


log = logging.getLogger("bot")


@dataclass
class QueueItem:
    """Represents a queued transcription task."""
    update: Update
    context: ContextTypes.DEFAULT_TYPE
    position: int
    status_message: Message | None = None


class TranscriptionQueue:
    """Manages sequential processing of transcription tasks."""
    
    def __init__(self):
        self._queue: list[QueueItem] = []
        self._processing = False
        self._lock = asyncio.Lock()
        self._current_task: QueueItem | None = None
        self._task_counter = 0
        
    async def add_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE, status_message: Message) -> int:
        """
        Add a new transcription task to the queue.
        Returns the position in queue (0-based).
        """
        async with self._lock:
            self._task_counter += 1
            position = len(self._queue)
            item = QueueItem(
                update=update,
                context=context,
                position=position,
                status_message=status_message
            )
            self._queue.append(item)
            log.info("Task #%d added to queue at position %d", self._task_counter, position)
            return position
    
    async def get_queue_position(self, update: Update) -> int | None:
        """Get current position in queue for a specific update."""
        async with self._lock:
            for idx, item in enumerate(self._queue):
                if item.update.message and update.message and item.update.message.message_id == update.message.message_id:
                    return idx
            return None
    
    async def remove_task(self, update: Update) -> bool:
        """Remove a task from the queue."""
        async with self._lock:
            for idx, item in enumerate(self._queue):
                if item.update.message and update.message and item.update.message.message_id == update.message.message_id:
                    self._queue.pop(idx)
                    await self._update_queue_positions()
                    return True
            return False
    
    async def _update_queue_positions(self):
        """Update position numbers for all queued items."""
        for idx, item in enumerate(self._queue):
            item.position = idx
    
    async def get_next_task(self) -> QueueItem | None:
        """Get the next task from the queue."""
        async with self._lock:
            if self._queue:
                item = self._queue.pop(0)
                await self._update_queue_positions()
                self._current_task = item
                # Update status messages for remaining items immediately
                await self._update_status_messages_unlocked()
                return item
            return None
    
    async def _update_status_messages_unlocked(self):
        """Update status messages without acquiring lock (internal use only)."""
        for item in self._queue:
            if item.status_message:
                try:
                    queue_text = f"⏳ Queued for processing\nPosition: {item.position + 1}"
                    if item.position == 0:
                        queue_text = "⏳ Next in queue..."
                    await item.status_message.edit_text(queue_text)
                except Exception as exc:
                    log.debug("Failed to update queue status: %s", exc)
    
    async def mark_task_complete(self):
        """Mark the current task as complete."""
        async with self._lock:
            self._current_task = None
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return len(self._queue)
    
    def is_processing(self) -> bool:
        """Check if a task is currently being processed."""
        return self._current_task is not None
    
    async def update_queue_status_messages(self):
        """Update status messages for all items waiting in queue."""
        async with self._lock:
            await self._update_status_messages_unlocked()


# Global queue instance
transcription_queue = TranscriptionQueue()

