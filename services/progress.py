from dataclasses import dataclass


@dataclass(slots=True)
class ProgressState:
    stage: str = "idle"
    current: float = 0.0
    total: float = 0.0
    audio_duration: float = 0.0
    unit: str = "segments"


def draw_progress_bar(current: float, total: float, length: int = 10) -> str:
    """Draw a visual progress bar using Unicode blocks."""
    if total <= 0:
        percent = 0.0
        filled = 0
    else:
        clamped_current = max(0.0, min(current, total))
        percent = (clamped_current / total) * 100
        filled = int(length * clamped_current / total)
    bar = "█" * filled + "░" * (length - filled)
    return f"{bar} {percent:.1f}%"


def draw_indeterminate_bar(step: int, length: int = 10) -> str:
    """Draw an animated bar for stages where total progress is unknown."""
    pos = step % length
    cells = ["░"] * length
    cells[pos] = "█"
    if pos > 0:
        cells[pos - 1] = "▓"
    return "".join(cells)


def format_stage(stage: str) -> str:
    labels = {
        "idle": "Waiting",
        "loading-default": "Loading default model",
        "detecting-language": "Detecting language",
        "language-detected": "Preparing transcription",
        "loading-special": "Loading Hebrew model",
        "transcribing": "Transcribing",
        "transcribing-special": "Transcribing with Hebrew model",
        "finalizing": "Finalizing",
        "uploading": "Uploading audio",
        "waiting-response": "Waiting for response",
    }
    return labels.get(stage, "Processing")


def format_seconds_compact(value: float) -> str:
    """Format seconds into a compact human-readable duration."""
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def calculate_eta(current: float, total: float, elapsed_seconds: int) -> str:
    """Estimate remaining time from current progress."""
    if current <= 0 or total <= 0 or elapsed_seconds <= 0:
        return "calculating..."
    if current >= total:
        return "0s"
    rate = current / float(elapsed_seconds)
    if rate <= 0:
        return "calculating..."
    remaining = (total - current) / rate
    return format_seconds_compact(remaining)


def build_status_message(progress: ProgressState, elapsed_seconds: int, tick: int) -> str:
    current = float(progress.current or 0.0)
    total = float(progress.total or 0.0)
    stage = format_stage(progress.stage)
    eta = calculate_eta(current, total, elapsed_seconds)

    has_real_progress = current > 0 or progress.stage == "finalizing"
    if total > 0 and has_real_progress:
        bar = draw_progress_bar(current, total)
        if progress.unit == "seconds":
            return (
                f"🎤 {stage}\n"
                f"📈 Progress: {bar}\n"
                f"🎧 Audio: {current:.1f}/{total:.1f}s\n"
                f"⏱️ Elapsed: {format_seconds_compact(elapsed_seconds)}\n"
                f"⌛ ETA: {eta}"
            )
        return (
            f"🎤 {stage}\n"
            f"📈 Progress: {bar}\n"
            f"📊 Segments: {int(current)}/{int(total)}\n"
            f"⏱️ Elapsed: {format_seconds_compact(elapsed_seconds)}\n"
            f"⌛ ETA: {eta}"
        )

    bar = draw_indeterminate_bar(tick)
    if progress.audio_duration > 0:
        return (
            f"🔄 {stage}\n"
            f"📈 Progress: {bar}\n"
            f"🎧 Audio: {progress.audio_duration:.1f}s\n"
            f"⏱️ Elapsed: {format_seconds_compact(elapsed_seconds)}\n"
            f"⌛ ETA: calculating..."
        )
    return (
        f"🔄 {stage}\n"
        f"📈 Progress: {bar}\n"
        f"⏱️ Elapsed: {format_seconds_compact(elapsed_seconds)}\n"
        f"⌛ ETA: calculating..."
    )

