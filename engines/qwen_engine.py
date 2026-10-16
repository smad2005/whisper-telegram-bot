import importlib.util
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
import traceback

from config import QwenConfig
from services.progress import ProgressState


log = logging.getLogger("bot")

SAMPLE_RATE = 16000
# When cutting a chunk, look for the quietest spot in this tail window (seconds).
SPLIT_SEARCH_SECONDS = 5.0
SPLIT_FRAME_SECONDS = 0.2


def _split_points(audio, chunk_seconds: int) -> list[tuple[int, int]]:
    """Split audio into <= chunk_seconds pieces, cutting at the quietest nearby frame."""
    import numpy as np

    total = len(audio)
    chunk = chunk_seconds * SAMPLE_RATE
    if total <= chunk:
        return [(0, total)]

    frame = int(SPLIT_FRAME_SECONDS * SAMPLE_RATE)
    search = int(SPLIT_SEARCH_SECONDS * SAMPLE_RATE)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = start + chunk
        if end >= total:
            bounds.append((start, total))
            break

        lo = max(start + frame, end - search)
        window = audio[lo:end]
        usable = (len(window) // frame) * frame
        if usable >= frame:
            energy = np.abs(window[:usable]).reshape(-1, frame).mean(axis=1)
            end = lo + int(np.argmin(energy)) * frame + frame // 2
        bounds.append((start, end))
        start = end
    return bounds


def _configure_worker_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class _WorkerQwenRuntime:
    """Actual Qwen3-ASR runtime living inside a dedicated child process."""

    def __init__(self, config: QwenConfig, progress_queue):
        self.config = config
        self.progress_queue = progress_queue
        self.model = None
        self.log = logging.getLogger("bot")

    def _set_progress(
        self,
        job_id: int,
        *,
        stage: str,
        current: float = 0.0,
        total: float = 0.0,
        audio_duration: float = 0.0,
        unit: str = "seconds",
    ):
        self.progress_queue.put(
            {
                "type": "progress",
                "job_id": job_id,
                "stage": stage,
                "current": current,
                "total": total,
                "audio_duration": audio_duration,
                "unit": unit,
            }
        )

    def _ensure_model_loaded(self):
        if self.model is not None:
            return self.model

        import torch
        from qwen_asr import Qwen3ASRModel

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if self.config.dtype not in dtypes:
            raise ValueError("QWEN_DTYPE must be one of: bfloat16, float16, float32.")

        self.log.info("Loading Qwen model: %s on %s (%s)", self.config.model_path, self.config.device, self.config.dtype)
        t0 = time.time()
        self.model = Qwen3ASRModel.from_pretrained(
            self.config.model_path,
            dtype=dtypes[self.config.dtype],
            device_map=self.config.device,
            max_inference_batch_size=self.config.batch_size,
            max_new_tokens=self.config.max_new_tokens,
        )
        self.log.info("Qwen model loaded in %.1fs", time.time() - t0)
        return self.model

    def transcribe(self, job_id: int, path: str) -> dict:
        from faster_whisper.audio import decode_audio

        t0 = time.time()
        self._set_progress(job_id, stage="loading-default")
        model = self._ensure_model_loaded()

        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        bounds = _split_points(audio, self.config.chunk_seconds)
        self._set_progress(job_id, stage="transcribing", total=duration, audio_duration=duration)

        language = self.config.language
        texts: list[str] = []
        segments: list[dict] = []
        detected: str | None = None
        batch = max(1, self.config.batch_size)

        for i in range(0, len(bounds), batch):
            group = bounds[i : i + batch]
            results = model.transcribe(
                audio=[(audio[s:e], SAMPLE_RATE) for s, e in group],
                language=[language] * len(group) if language else None,
            )
            for (s, e), res in zip(group, results):
                text = (res.text or "").strip()
                detected = detected or getattr(res, "language", None)
                if text:
                    texts.append(text)
                    segments.append({"start": s / SAMPLE_RATE, "end": e / SAMPLE_RATE, "text": text})
            self._set_progress(
                job_id,
                stage="transcribing",
                current=group[-1][1] / SAMPLE_RATE,
                total=duration,
                audio_duration=duration,
            )

        self._set_progress(job_id, stage="finalizing", current=duration, total=duration, audio_duration=duration)
        self.log.info("Qwen transcription complete: %d chunks, language=%s", len(bounds), detected)
        return {
            "text": " ".join(texts),
            "segments": segments,
            "duration": duration,
            "elapsed": time.time() - t0,
            "engine": f"qwen ({detected})" if detected else "qwen",
        }


def _worker_main(config: QwenConfig, command_queue, response_queue, progress_queue):
    _configure_worker_logging()
    runtime = _WorkerQwenRuntime(config, progress_queue)

    while True:
        command = command_queue.get()
        command_type = command.get("type")

        if command_type == "shutdown":
            response_queue.put({"type": "shutdown_ack"})
            break

        if command_type != "transcribe":
            response_queue.put(
                {
                    "type": "error",
                    "job_id": command.get("job_id"),
                    "error": f"Unknown command: {command_type}",
                }
            )
            continue

        job_id = command["job_id"]
        try:
            result = runtime.transcribe(job_id, command["path"])
            response_queue.put({"type": "result", "job_id": job_id, "result": result})
        except Exception as exc:
            response_queue.put(
                {
                    "type": "error",
                    "job_id": job_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


class QwenEngine:
    """Controller for a dedicated Qwen3-ASR worker process (same model as WhisperEngine)."""

    def __init__(self, config: QwenConfig):
        # find_spec: don't import torch into the bot process, only the worker needs it.
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("qwen_asr") is None:
            log.error(
                "STT_ENGINE=qwen, but Qwen dependencies (torch, qwen-asr) are not installed in this image. "
                "Set WITH_QWEN=1 in .env and rebuild: docker compose up --build"
            )
            sys.exit(1)

        self.config = config
        self.last_active = 0.0
        self.active_transcriptions = 0
        self.progress = ProgressState()
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._job_counter = 0
        self._worker = None
        self._command_queue = None
        self._response_queue = None
        self._progress_queue = None
        log.info(
            "Qwen engine initialized: %s on %s (%s, idle timeout: %ds, isolated worker mode)",
            config.model_path,
            config.device,
            config.dtype,
            config.idle_timeout,
        )

    def _worker_is_alive(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def _clear_ipc(self):
        self._worker = None
        self._command_queue = None
        self._response_queue = None
        self._progress_queue = None

    def _ensure_worker(self):
        if self._worker_is_alive():
            return

        self._command_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._progress_queue = self._ctx.Queue()
        self._worker = self._ctx.Process(
            target=_worker_main,
            args=(self.config, self._command_queue, self._response_queue, self._progress_queue),
            daemon=True,
        )
        self._worker.start()
        self.last_active = time.time()
        log.info("Started Qwen worker process PID %s", self._worker.pid)

    def _drain_progress(self, job_id: int):
        if not self._progress_queue:
            return

        while True:
            try:
                message = self._progress_queue.get_nowait()
            except queue.Empty:
                break

            if message.get("type") != "progress" or message.get("job_id") != job_id:
                continue

            self.progress = ProgressState(
                stage=message.get("stage", "idle"),
                current=float(message.get("current", 0.0) or 0.0),
                total=float(message.get("total", 0.0) or 0.0),
                audio_duration=float(message.get("audio_duration", 0.0) or 0.0),
                unit=message.get("unit", "seconds"),
            )

    def _stop_worker(self, reason: str) -> bool:
        if not self._worker:
            return False

        log.info("Stopping Qwen worker process (%s)", reason)
        try:
            if self._command_queue:
                self._command_queue.put({"type": "shutdown"})
        except Exception as exc:
            log.debug("Failed to send shutdown command to worker: %s", exc)

        try:
            if self._response_queue:
                message = self._response_queue.get(timeout=5)
                if message.get("type") != "shutdown_ack":
                    log.debug("Unexpected shutdown response from worker: %s", message)
        except Exception:
            pass

        self._worker.join(timeout=5)
        if self._worker.is_alive():
            log.warning("Qwen worker did not exit cleanly; terminating PID %s", self._worker.pid)
            self._worker.terminate()
            self._worker.join(timeout=5)

        exited = not self._worker.is_alive()
        if exited:
            log.info("Qwen worker process stopped; OS should reclaim RAM and VRAM now.")
        self._clear_ipc()
        self.progress = ProgressState()
        return exited

    def unload_if_idle(self) -> bool:
        if self.active_transcriptions > 0:
            return False
        if not self._worker_is_alive():
            return False
        if time.time() - self.last_active <= self.config.idle_timeout:
            return False
        return self._stop_worker("idle timeout")

    def transcribe(self, path: str) -> dict:
        with self._lock:
            self.active_transcriptions += 1
            try:
                self._ensure_worker()
                self._job_counter += 1
                job_id = self._job_counter
                self.progress = ProgressState(stage="loading-default", unit="seconds")
                self._command_queue.put({"type": "transcribe", "job_id": job_id, "path": path})

                while True:
                    self._drain_progress(job_id)
                    try:
                        message = self._response_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not self._worker_is_alive():
                            raise RuntimeError("Qwen worker process exited unexpectedly.")
                        continue

                    if message.get("type") == "shutdown_ack":
                        continue
                    if message.get("job_id") != job_id:
                        continue

                    self._drain_progress(job_id)
                    self.last_active = time.time()

                    if message.get("type") == "result":
                        return message["result"]

                    if message.get("type") == "error":
                        worker_traceback = message.get("traceback")
                        if worker_traceback:
                            log.error("Qwen worker traceback:\n%s", worker_traceback)
                        raise RuntimeError(message.get("error", "Qwen worker failed."))
            finally:
                self.active_transcriptions = max(0, self.active_transcriptions - 1)
