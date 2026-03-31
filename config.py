from dataclasses import dataclass, field
import os


@dataclass(slots=True)
class TelegramConfig:
    token: str
    allowed_users: list[int] = field(default_factory=list)
    polling_timeout: int = 120


@dataclass(slots=True)
class WhisperConfig:
    model_path: str
    model2_path: str | None
    special_lang: str
    device: str
    compute: str
    device2: str
    compute2: str
    language: str | None
    beam: int
    idle_timeout: int


@dataclass(slots=True)
class GeminiConfig:
    api_key: str
    model: str
    language: str


@dataclass(slots=True)
class BotConfig:
    telegram: TelegramConfig
    engine: str
    whisper: WhisperConfig
    gemini: GeminiConfig


def _parse_allowed_users(raw_value: str) -> list[int]:
    raw_value = raw_value.split("#")[0].strip()
    if not raw_value:
        return []

    try:
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid user ID in ALLOWED_USERS: {exc}") from exc


def load_config() -> BotConfig:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token or token == "YOUR_BOT_TOKEN":
        raise ValueError("BOT_TOKEN is not set! Check your .env file.")

    engine = os.getenv("STT_ENGINE", "whisper").strip().lower() or "whisper"
    if engine not in {"whisper", "gemini"}:
        raise ValueError("STT_ENGINE must be either 'whisper' or 'gemini'.")

    whisper_language = os.getenv("WHISPER_LANGUAGE", "auto").strip()
    gemini_language = os.getenv("GEMINI_LANGUAGE", "auto").strip()

    return BotConfig(
        telegram=TelegramConfig(
            token=token,
            allowed_users=_parse_allowed_users(os.getenv("ALLOWED_USERS", "")),
            polling_timeout=int(os.getenv("TELEGRAM_POLL_TIMEOUT", "120")),
        ),
        engine=engine,
        whisper=WhisperConfig(
            model_path=os.getenv("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2"),
            model2_path=os.getenv("WHISPER_MODEL2") or None,
            special_lang=os.getenv("WHISPER_SPECIAL_LANG", "he"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute=os.getenv("WHISPER_COMPUTE", "int8"),
            device2=os.getenv("WHISPER_DEVICE2", os.getenv("WHISPER_DEVICE", "cpu")),
            compute2=os.getenv("WHISPER_COMPUTE2", os.getenv("WHISPER_COMPUTE", "int8")),
            language=None if whisper_language == "auto" else whisper_language,
            beam=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
            idle_timeout=int(os.getenv("WHISPER_IDLE_TIMEOUT", "600")),
        ),
        gemini=GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            language="Detect language" if gemini_language == "auto" else gemini_language,
        ),
    )

