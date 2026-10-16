import importlib.util
import logging
import multiprocessing as mp
import queue
import re
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


# --- Speech-aware chunking and subtitle timing -------------------------------------------------
# Qwen3-ASR returns plain text without timestamps, so subtitle timing is derived from the audio:
# Silero VAD (bundled with faster-whisper) finds speech, RMS energy refines its boundaries, and the
# text of each chunk is split into phrases whose times are spread over the speech-only timeline.

VAD_THRESHOLD = 0.5
VAD_NEG_THRESHOLD = 0.35
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 200
CHUNK_PAD_SECONDS = 0.3  # extra audio context fed to Qwen around each speech chunk

ENERGY_FRAME = 512  # 32 ms at 16 kHz, same as the VAD frame
ENERGY_FRAME_SECONDS = ENERGY_FRAME / SAMPLE_RATE
NOISE_BLOCK_SECONDS = 10.0
NOISE_FLOOR_PERCENTILE = 15
GLOBAL_FLOOR_PERCENTILE = 10
NOISE_MULTIPLIER = 2.5
SPEECH_LEVEL_FRACTION = 0.35
ONSET_LOOKBACK_SECONDS = 0.6
OFFSET_LOOKAHEAD_SECONDS = 1.0
ENERGY_BRIDGE_SECONDS = 0.2

PAUSE_SNAP_SECONDS = 0.8  # max distance (in speech time) to pull a phrase border onto a pause
MIN_PIECE_SECONDS = 0.2
MAX_SUBTITLE_CHARS = 84
MAX_SUBTITLE_CHARS_CJK = 42
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])\s*")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+|(?<=[、，；：])\s*")
_CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


def _detect_speech(audio, max_seconds: float) -> list[tuple[float, float]]:
    """Speech intervals in seconds, each at most max_seconds long."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=VAD_THRESHOLD,
        neg_threshold=VAD_NEG_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        max_speech_duration_s=max_seconds,
        speech_pad_ms=0,
    )
    chunks = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in chunks]


def _refine_speech(audio, segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Extend VAD boundaries over frames whose RMS is above an adaptive noise floor.

    VAD tends to start a little late and end a little early. Onsets move back up to 0.6 s and
    ends forward up to 1 s, never into a neighbouring segment.
    """
    import numpy as np

    n = len(audio) // ENERGY_FRAME
    if n == 0 or not segments:
        return segments

    rms = np.sqrt((audio[: n * ENERGY_FRAME].reshape(n, ENERGY_FRAME) ** 2).mean(axis=1))
    block = max(1, round(NOISE_BLOCK_SECONDS / ENERGY_FRAME_SECONDS))
    global_floor = np.percentile(rms, GLOBAL_FLOOR_PERCENTILE)
    floors = [
        max(np.percentile(rms[i : i + block], NOISE_FLOOR_PERCENTILE), global_floor)
        for i in range(0, n, block)
    ]
    frame_floor = np.repeat(np.array(floors), block)[:n]

    spans = []
    for start, end in segments:
        a = min(n - 1, round(start / ENERGY_FRAME_SECONDS))
        b = min(n - 1, max(a, round(end / ENERGY_FRAME_SECONDS) - 1))
        spans.append([a, b])

    speech_level = np.median(np.concatenate([rms[a : b + 1] for a, b in spans]))
    active = rms >= np.maximum(frame_floor * NOISE_MULTIPLIER, speech_level * SPEECH_LEVEL_FRACTION)

    lookback = round(ONSET_LOOKBACK_SECONDS / ENERGY_FRAME_SECONDS)
    lookahead = round(OFFSET_LOOKAHEAD_SECONDS / ENERGY_FRAME_SECONDS)
    bridge = max(1, round(ENERGY_BRIDGE_SECONDS / ENERGY_FRAME_SECONDS))

    for i, span in enumerate(spans):
        limit = max(spans[i - 1][1] + 1 if i else 0, span[0] - lookback)
        earliest, gap = span[0], 0
        for f in range(span[0] - 1, limit - 1, -1):
            if active[f]:
                earliest, gap = f, 0
            else:
                gap += 1
                if gap > bridge:
                    break
        span[0] = earliest

    for i, span in enumerate(spans):
        limit = min(spans[i + 1][0] - 1 if i + 1 < len(spans) else n - 1, span[1] + lookahead)
        latest, gap = span[1], 0
        for f in range(span[1] + 1, limit + 1):
            if active[f]:
                latest, gap = f, 0
            else:
                gap += 1
                if gap > bridge:
                    break
        span[1] = latest

    return [(a * ENERGY_FRAME_SECONDS, (b + 1) * ENERGY_FRAME_SECONDS) for a, b in spans]


def _pack_chunks(segments: list[tuple[float, float]], max_seconds: float) -> list[list[tuple[float, float]]]:
    """Group consecutive speech segments into chunks spanning at most max_seconds."""
    chunks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for seg in segments:
        if current and seg[1] - current[0][0] > max_seconds:
            chunks.append(current)
            current = []
        current.append(seg)
    if current:
        chunks.append(current)
    return chunks


def _hard_wrap(text: str, limit: int, cjk: bool) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = limit if cjk else (text.rfind(" ", 0, limit + 1) or limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _split_text(text: str) -> list[str]:
    """Split text into subtitle-sized phrases: sentences first, long ones at clause breaks."""
    pieces: list[str] = []
    for sentence in filter(None, (s.strip() for s in _SENTENCE_END.split(text.strip()))):
        cjk = len(_CJK.findall(sentence)) > len(sentence) * 0.3
        limit = MAX_SUBTITLE_CHARS_CJK if cjk else MAX_SUBTITLE_CHARS
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue

        joiner = "" if cjk else " "
        current = ""
        for clause in filter(None, (c.strip() for c in _CLAUSE_BREAK.split(sentence))):
            candidate = f"{current}{joiner}{clause}" if current else clause
            if current and len(candidate) > limit:
                pieces.extend(_hard_wrap(current, limit, cjk))
                current = clause
            else:
                current = candidate
        if current:
            pieces.extend(_hard_wrap(current, limit, cjk))
    return pieces


def _allocate_times(pieces: list[str], spans: list[tuple[float, float]]) -> list[tuple[float, float, str]]:
    """Spread phrases over the speech-only timeline of a chunk, proportionally to text length."""
    total = sum(end - start for start, end in spans)
    weights = [max(1, len(re.sub(r"\s", "", piece))) for piece in pieces]
    weight_sum = sum(weights)

    def to_time(position: float, is_end: bool) -> float:
        passed = 0.0
        for start, end in spans:
            length = end - start
            if position < passed + length or (is_end and position <= passed + length):
                return start + (position - passed)
            passed += length
        return spans[-1][1]

    # Phrase borders by text length, in speech-time seconds (n pieces -> n - 1 inner borders).
    borders = []
    cumulative = 0
    for weight in weights[:-1]:
        cumulative += weight
        borders.append(cumulative / weight_sum * total)

    # Text length is only a rough guide: pull a border onto a real pause between speech spans when
    # it falls close to one, so a subtitle does not stretch over a pause or start before its speech.
    pauses = []
    passed = 0.0
    for start, end in spans[:-1]:
        passed += end - start
        pauses.append(passed)

    snapped: list[float] = []
    previous = 0.0
    for k, border in enumerate(borders):
        upper = borders[k + 1] if k + 1 < len(borders) else total
        nearest = min(pauses, key=lambda p: abs(p - border), default=None)
        if (
            nearest is not None
            and abs(nearest - border) <= PAUSE_SNAP_SECONDS
            and previous + MIN_PIECE_SECONDS <= nearest <= upper - MIN_PIECE_SECONDS
        ):
            border = nearest
        snapped.append(border)
        previous = border

    edges = [0.0, *snapped, total]
    return [
        (to_time(edges[i], False), to_time(edges[i + 1], True), piece) for i, piece in enumerate(pieces)
    ]


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
        self._set_progress(job_id, stage="transcribing", total=duration, audio_duration=duration)

        # chunks: list of speech-interval groups (seconds). Qwen gets the audio spanning each group.
        chunks: list[list[tuple[float, float]]] = []
        pad = CHUNK_PAD_SECONDS
        try:
            speech = _refine_speech(audio, _detect_speech(audio, self.config.chunk_seconds))
            chunks = _pack_chunks(speech, self.config.chunk_seconds)
        except Exception as exc:
            self.log.warning("VAD failed (%s), falling back to fixed-size chunks", exc)
        if not chunks:
            # No speech found (or VAD failed): don't drop audio silently, cut it at quiet points instead.
            pad = 0.0
            chunks = [[(s / SAMPLE_RATE, e / SAMPLE_RATE)] for s, e in _split_points(audio, self.config.chunk_seconds)]

        # Audio window per chunk: the chunk plus some context, but never past the midpoint of the gap
        # to the neighbouring chunk, otherwise a word on the border is transcribed twice.
        windows_all: list[tuple[int, int]] = []
        for idx, c in enumerate(chunks):
            left = (chunks[idx - 1][-1][1] + c[0][0]) / 2 if idx else 0.0
            right = (c[-1][1] + chunks[idx + 1][0][0]) / 2 if idx + 1 < len(chunks) else duration
            windows_all.append(
                (int(max(left, c[0][0] - pad) * SAMPLE_RATE), int(min(right, c[-1][1] + pad) * SAMPLE_RATE))
            )

        language = self.config.language
        texts: list[str] = []
        segments: list[dict] = []
        detected: str | None = None
        batch = max(1, self.config.batch_size)

        for i in range(0, len(chunks), batch):
            group = chunks[i : i + batch]
            windows = windows_all[i : i + batch]
            results = model.transcribe(
                audio=[(audio[s:e], SAMPLE_RATE) for s, e in windows],
                language=[language] * len(group) if language else None,
            )
            for spans, res in zip(group, results):
                text = (res.text or "").strip()
                detected = detected or getattr(res, "language", None)
                if not text:
                    continue
                texts.append(text)
                for start, end, piece in _allocate_times(_split_text(text) or [text], spans):
                    segments.append({"start": start, "end": end, "text": piece})
            self._set_progress(
                job_id,
                stage="transcribing",
                current=group[-1][-1][1],
                total=duration,
                audio_duration=duration,
            )

        self._set_progress(job_id, stage="finalizing", current=duration, total=duration, audio_duration=duration)
        self.log.info(
            "Qwen transcription complete: %d chunks, %d subtitle lines, language=%s", len(chunks), len(segments), detected
        )
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
