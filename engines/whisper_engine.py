import gc
import logging
import multiprocessing as mp
import queue
import subprocess
import threading
import time
import traceback
from ctypes import CDLL

from config import WhisperConfig
from services.progress import ProgressState


log = logging.getLogger("bot")

try:
    LIBC = CDLL("libc.so.6")
except OSError:
    LIBC = None


def _get_container_memory_mb() -> float | None:
    """Return current container memory usage in MiB from cgroups when available."""
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                return int(file_obj.read().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            continue
    return None


def _log_container_memory(logger: logging.Logger, prefix: str):
    """Log current container RAM usage when available."""
    memory_mb = _get_container_memory_mb()
    if memory_mb is not None:
        logger.info("%s container RAM: %.1f MiB", prefix, memory_mb)


def _configure_worker_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class _WorkerWhisperRuntime:
    """Actual Whisper runtime living inside a dedicated child process."""

    def __init__(self, config: WhisperConfig, progress_queue):
        self.config = config
        self.progress_queue = progress_queue
        self.model = None
        self.model2 = None
        self.log = logging.getLogger("bot")

    def _set_progress(
        self,
        job_id: int,
        *,
        stage: str,
        current: float = 0.0,
        total: float = 0.0,
        audio_duration: float = 0.0,
        unit: str = "segments",
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

    def _get_gpu_memory_snapshot(self) -> str | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            first_line = result.stdout.strip().splitlines()[0]
            used, total = [item.strip() for item in first_line.split(",", 1)]
            return f"{used}/{total} MiB"
        except Exception:
            return None

    def _log_gpu_memory(self, prefix: str):
        snapshot = self._get_gpu_memory_snapshot()
        if snapshot:
            self.log.info("%s GPU memory: %s", prefix, snapshot)

    def _get_process_rss_mb(self) -> float | None:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        return rss_kb / 1024.0
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _log_process_memory(self, prefix: str):
        rss_mb = self._get_process_rss_mb()
        if rss_mb is not None:
            self.log.info("%s process RSS: %.1f MiB", prefix, rss_mb)

    def _log_memory_snapshot(self, prefix: str):
        self._log_gpu_memory(prefix)
        self._log_process_memory(prefix)
        _log_container_memory(self.log, prefix)

    def _schedule_delayed_memory_snapshot(self, prefix: str, delay_seconds: int = 30):
        """Log memory again later to confirm allocator/cgroup state settles."""

        def _log_later():
            time.sleep(delay_seconds)
            self._log_memory_snapshot(prefix)

        threading.Thread(target=_log_later, daemon=True).start()

    def _trim_process_memory(self):
        if LIBC is None:
            return
        try:
            LIBC.malloc_trim(0)
        except Exception as exc:
            self.log.debug("malloc_trim failed: %s", exc)

    def _ensure_model_loaded(self, special: bool = False):
        from faster_whisper import WhisperModel

        if special:
            if not self.model2:
                self.log.info(
                    "Loading special Whisper model: %s on %s (%s)",
                    self.config.model2_path,
                    self.config.device2,
                    self.config.compute2,
                )
                t0 = time.time()
                self.model2 = WhisperModel(
                    self.config.model2_path,
                    device=self.config.device2,
                    compute_type=self.config.compute2,
                )
                self.log.info("Special model loaded in %.1fs", time.time() - t0)
            return self.model2

        if not self.model:
            self.log.info("Loading default Whisper model: %s", self.config.model_path)
            t0 = time.time()
            self.model = WhisperModel(
                self.config.model_path,
                device=self.config.device,
                compute_type=self.config.compute,
            )
            self.log.info("Default model loaded in %.1fs", time.time() - t0)
        return self.model

    def _release_model(self, attr_name: str, reason: str) -> bool:
        model_instance = getattr(self, attr_name, None)
        if not model_instance:
            return False

        self.log.info("Releasing %s model (%s)", attr_name, reason)
        self._log_memory_snapshot("Before release")

        try:
            unload_method = getattr(model_instance, "unload_model", None)
            if callable(unload_method):
                unload_method()
        except Exception as exc:
            self.log.warning("Could not call %s.unload_model(): %s", attr_name, exc)

        try:
            inner_model = getattr(model_instance, "model", None)
            inner_unload = getattr(inner_model, "unload_model", None)
            if callable(inner_unload):
                inner_unload()
        except Exception as exc:
            self.log.warning("Could not call inner unload_model() for %s: %s", attr_name, exc)

        setattr(self, attr_name, None)
        del model_instance
        gc.collect()
        gc.collect()
        self._trim_process_memory()
        self._log_memory_snapshot("After release")
        if reason != "worker shutdown":
            self._schedule_delayed_memory_snapshot("30s after release")
        return True

    def unload_all(self, reason: str):
        self._release_model("model", reason)
        self._release_model("model2", reason)

    def transcribe(self, job_id: int, path: str) -> dict:
        t0 = time.time()
        self._set_progress(job_id, stage="loading-default")
        model = self._ensure_model_loaded()

        self._set_progress(job_id, stage="detecting-language")
        segs, info = model.transcribe(
            path,
            language=self.config.language,
            beam_size=self.config.beam,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )

        detected_lang = info.language
        engine_name = "whisper"
        audio_duration = info.duration or 0.0
        self._set_progress(
            job_id,
            stage="language-detected",
            current=0.0,
            total=audio_duration,
            audio_duration=audio_duration,
            unit="seconds",
        )

        if self.config.model2_path and detected_lang == self.config.special_lang:
            self.log.info("Switching to special model for language: %s", detected_lang)

            if info.duration > 60 and self.model:
                self.log.info("Audio duration %.1fs > 60s, unloading first model to free memory", info.duration)
                segs = None
                self._release_model("model", "long special-language audio before loading model2")
                model = None
                self.log.info("First model unloaded before loading special model")

            self._set_progress(
                job_id,
                stage="loading-special",
                current=0.0,
                total=audio_duration,
                audio_duration=audio_duration,
                unit="seconds",
            )
            model2 = self._ensure_model_loaded(special=True)
            self._set_progress(
                job_id,
                stage="transcribing-special",
                current=0.0,
                total=audio_duration,
                audio_duration=audio_duration,
                unit="seconds",
            )
            segs, info = model2.transcribe(
                path,
                language=detected_lang,
                beam_size=self.config.beam,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            )
            engine_name = f"whisper-special ({detected_lang})"
        else:
            self._set_progress(
                job_id,
                stage="transcribing",
                current=0.0,
                total=audio_duration,
                audio_duration=audio_duration,
                unit="seconds",
            )

        self.log.info("Collecting transcription segments...")
        texts = []
        segment_count = 0

        for seg in segs:
            texts.append(seg.text.strip())
            segment_count += 1
            current_position = (
                min(float(getattr(seg, "end", 0.0) or 0.0), audio_duration)
                if audio_duration
                else float(segment_count)
            )
            total_value = audio_duration if audio_duration else float(max(segment_count, 1))
            self._set_progress(
                job_id,
                stage="transcribing-special" if engine_name.startswith("whisper-special") else "transcribing",
                current=current_position,
                total=total_value,
                audio_duration=audio_duration,
                unit="seconds" if audio_duration else "segments",
            )

            if segment_count % 10 == 0:
                if audio_duration:
                    self.log.info(
                        "Processed %d segments (%.1fs/%.1fs audio, %.1fs elapsed)",
                        segment_count,
                        current_position,
                        audio_duration,
                        time.time() - t0,
                    )
                else:
                    self.log.info("Processed %d segments (%.1fs elapsed)", segment_count, time.time() - t0)

        self._set_progress(
            job_id,
            stage="finalizing",
            current=audio_duration or float(segment_count),
            total=audio_duration or float(max(segment_count, 1)),
            audio_duration=audio_duration,
            unit="seconds" if audio_duration else "segments",
        )
        self.log.info("Transcription complete: %d segments processed", segment_count)
        return {
            "text": " ".join(texts),
            "duration": info.duration,
            "elapsed": time.time() - t0,
            "engine": engine_name,
        }


def _worker_main(config: WhisperConfig, command_queue, response_queue, progress_queue):
    _configure_worker_logging()
    runtime = _WorkerWhisperRuntime(config, progress_queue)

    while True:
        command = command_queue.get()
        command_type = command.get("type")

        if command_type == "shutdown":
            runtime.unload_all("worker shutdown")
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


class WhisperEngine:
    """Controller for a dedicated Whisper worker process."""

    def __init__(self, config: WhisperConfig):
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
        log.info("Whisper engine initialized (idle timeout: %ds, isolated worker mode)", self.config.idle_timeout)

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

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

    def _worker_is_alive(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def _clear_ipc(self):
        self._worker = None
        self._command_queue = None
        self._response_queue = None
        self._progress_queue = None

    def _log_parent_memory_snapshot(self, prefix: str):
        _log_container_memory(log, prefix)

    def _schedule_parent_memory_snapshot(self, prefix: str, delay_seconds: int = 30):
        """Log container RAM later from the controller process."""

        def _log_later():
            time.sleep(delay_seconds)
            self._log_parent_memory_snapshot(prefix)

        threading.Thread(target=_log_later, daemon=True).start()

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
        log.info("Started Whisper worker process PID %s", self._worker.pid)

    def _drain_progress(self, job_id: int):
        if not self._progress_queue:
            return

        while True:
            try:
                message = self._progress_queue.get_nowait()
            except queue.Empty:
                break

            if message.get("type") != "progress":
                continue
            if message.get("job_id") != job_id:
                continue

            self.progress = ProgressState(
                stage=message.get("stage", "idle"),
                current=float(message.get("current", 0.0) or 0.0),
                total=float(message.get("total", 0.0) or 0.0),
                audio_duration=float(message.get("audio_duration", 0.0) or 0.0),
                unit=message.get("unit", "segments"),
            )

    def _stop_worker(self, reason: str) -> bool:
        if not self._worker:
            return False

        log.info("Stopping Whisper worker process (%s)", reason)
        self._log_parent_memory_snapshot("Before worker stop")
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
            log.warning("Whisper worker did not exit cleanly; terminating PID %s", self._worker.pid)
            self._worker.terminate()
            self._worker.join(timeout=5)

        exited = not self._worker.is_alive()
        if exited:
            log.info("Whisper worker process stopped; OS should reclaim RAM and VRAM now.")
            self._log_parent_memory_snapshot("After worker stop")
            self._schedule_parent_memory_snapshot("30s after worker stop")
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
                job_id = self._next_job_id()
                self._set_progress(stage="loading-default")
                self._command_queue.put({"type": "transcribe", "job_id": job_id, "path": path})

                while True:
                    self._drain_progress(job_id)
                    try:
                        message = self._response_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not self._worker_is_alive():
                            raise RuntimeError("Whisper worker process exited unexpectedly.")
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
                            log.error("Whisper worker traceback:\n%s", worker_traceback)
                        raise RuntimeError(message.get("error", "Whisper worker failed."))
            finally:
                self.active_transcriptions = max(0, self.active_transcriptions - 1)

