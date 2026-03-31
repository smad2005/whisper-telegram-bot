import logging
import sys
import time
from os import path as ospath

from config import GeminiConfig
from services.progress import ProgressState


log = logging.getLogger("bot")


class GeminiEngine:
    """Cloud-based speech-to-text engine powered by Google Gemini API."""

    def __init__(self, config: GeminiConfig):
        from google import genai

        if not config.api_key or config.api_key == "YOUR_GEMINI_API_KEY":
            log.error("GEMINI_API_KEY is not set! Check your .env file.")
            sys.exit(1)

        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.progress = ProgressState()
        log.info("Gemini ready: %s (%s)", self.config.model, self.config.language)

    def _set_progress(
        self,
        *,
        stage: str,
        current: float = 0.0,
        total: float = 0.0,
        audio_duration: float = 0.0,
        unit: str = "segments",
    ):
        self.progress = ProgressState(
            stage=stage,
            current=current,
            total=total,
            audio_duration=audio_duration,
            unit=unit,
        )

    def transcribe(self, file_path: str) -> dict:
        """Send audio to Gemini API and return transcription with metadata."""
        from google.genai import types

        self._set_progress(stage="uploading")
        t0 = time.time()
        with open(file_path, "rb") as file_obj:
            data = file_obj.read()

        ext = ospath.splitext(file_path)[1].lower()
        mimes = {
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }

        self._set_progress(stage="waiting-response")
        resp = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                f"Transcribe this audio. Language: {self.config.language}. Output ONLY the transcription, nothing else.",
                types.Part.from_bytes(data=data, mime_type=mimes.get(ext, "audio/ogg")),
            ],
        )
        self._set_progress(stage="finalizing", current=1, total=1)
        return {
            "text": resp.text.strip(),
            "duration": 0,
            "elapsed": time.time() - t0,
            "engine": "gemini",
        }

