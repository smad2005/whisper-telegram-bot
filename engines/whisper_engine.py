import gc
import logging
import subprocess
import time

from config import WhisperConfig
from services.progress import ProgressState


log = logging.getLogger("bot")


class WhisperEngine:
    """Offline speech-to-text engine powered by faster-whisper."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = None
        self.model2 = None
        self.last_active = 0.0
        self.active_transcriptions = 0
        self.progress = ProgressState()
        log.info("Whisper engine initialized (idle timeout: %ds)", self.config.idle_timeout)

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

    def _ensure_model_loaded(self, special: bool = False):
        """Lazy load models when needed."""
        from faster_whisper import WhisperModel

        self.last_active = time.time()

        if special:
            if not self.model2:
                log.info(
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
                log.info("Special model loaded in %.1fs", time.time() - t0)
            return self.model2

        if not self.model:
            log.info("Loading default Whisper model: %s", self.config.model_path)
            t0 = time.time()
            self.model = WhisperModel(
                self.config.model_path,
                device=self.config.device,
                compute_type=self.config.compute,
            )
            log.info("Default model loaded in %.1fs", time.time() - t0)
        return self.model

    def _get_gpu_memory_snapshot(self) -> str | None:
        """Return a short VRAM usage string when nvidia-smi is available."""
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
        """Log current VRAM usage when running on CUDA."""
        snapshot = self._get_gpu_memory_snapshot()
        if snapshot:
            log.info("%s GPU memory: %s", prefix, snapshot)

    def _release_model(self, attr_name: str, reason: str) -> bool:
        """Force release a loaded Whisper model and try to free VRAM immediately."""
        model_instance = getattr(self, attr_name, None)
        if not model_instance:
            return False

        log.info("Releasing %s model (%s)", attr_name, reason)
        self._log_gpu_memory("Before release")

        try:
            unload_method = getattr(model_instance, "unload_model", None)
            if callable(unload_method):
                unload_method()
        except Exception as exc:
            log.warning("Could not call %s.unload_model(): %s", attr_name, exc)

        try:
            inner_model = getattr(model_instance, "model", None)
            inner_unload = getattr(inner_model, "unload_model", None)
            if callable(inner_unload):
                inner_unload()
        except Exception as exc:
            log.warning("Could not call inner unload_model() for %s: %s", attr_name, exc)

        setattr(self, attr_name, None)
        del model_instance
        gc.collect()
        gc.collect()
        self._log_gpu_memory("After release")
        return True

    def unload_if_idle(self) -> bool:
        """Unload models from memory if they have been idle too long."""
        if self.active_transcriptions > 0:
            return False

        if (self.model or self.model2) and (time.time() - self.last_active > self.config.idle_timeout):
            log.info("Inactivity timeout reached (%ds), unloading Whisper models...", self.config.idle_timeout)
            self._release_model("model", "idle timeout")
            self._release_model("model2", "idle timeout")
            log.info("Models unloaded. Current RAM usage may drop.")
            return True
        return False

    def transcribe(self, path: str) -> dict:
        """Transcribe an audio file and return text with metadata."""
        self.active_transcriptions += 1
        try:
            t0 = time.time()
            self._set_progress(stage="loading-default")
            model = self._ensure_model_loaded()

            self._set_progress(stage="detecting-language")
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
                stage="language-detected",
                current=0.0,
                total=audio_duration,
                audio_duration=audio_duration,
                unit="seconds",
            )

            if self.config.model2_path and detected_lang == self.config.special_lang:
                log.info("Switching to special model for language: %s", detected_lang)

                if info.duration > 60 and self.model:
                    log.info("Audio duration %.1fs > 60s, unloading first model to free memory", info.duration)
                    # Important: drop local references too, otherwise VRAM may stay occupied
                    segs = None
                    self._release_model("model", "long special-language audio before loading model2")
                    model = None
                    log.info("First model unloaded before loading special model")

                self._set_progress(
                    stage="loading-special",
                    current=0.0,
                    total=audio_duration,
                    audio_duration=audio_duration,
                    unit="seconds",
                )
                model2 = self._ensure_model_loaded(special=True)
                self._set_progress(
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
                    stage="transcribing",
                    current=0.0,
                    total=audio_duration,
                    audio_duration=audio_duration,
                    unit="seconds",
                )

            log.info("Collecting transcription segments...")
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
                    stage="transcribing-special" if engine_name.startswith("whisper-special") else "transcribing",
                    current=current_position,
                    total=total_value,
                    audio_duration=audio_duration,
                    unit="seconds" if audio_duration else "segments",
                )

                if segment_count % 10 == 0:
                    if audio_duration:
                        log.info(
                            "Processed %d segments (%.1fs/%.1fs audio, %.1fs elapsed)",
                            segment_count,
                            current_position,
                            audio_duration,
                            time.time() - t0,
                        )
                    else:
                        log.info("Processed %d segments (%.1fs elapsed)", segment_count, time.time() - t0)

            self._set_progress(
                stage="finalizing",
                current=audio_duration or float(segment_count),
                total=audio_duration or float(max(segment_count, 1)),
                audio_duration=audio_duration,
                unit="seconds" if audio_duration else "segments",
            )
            log.info("Transcription complete: %d segments processed", segment_count)
            self.last_active = time.time()
            return {
                "text": " ".join(texts),
                "duration": info.duration,
                "elapsed": time.time() - t0,
                "engine": engine_name,
            }
        finally:
            self.active_transcriptions = max(0, self.active_transcriptions - 1)

