<div align="center">

<!-- Light mode logo -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.icons8.com/fluency/96/microphone.png">
  <source media="(prefers-color-scheme: light)" srcset="https://img.icons8.com/fluency/96/microphone.png">
  <img alt="Whisper Telegram Bot" src="https://img.icons8.com/fluency/96/microphone.png" width="96">
</picture>

# Whisper Telegram Bot

**A Telegram bot that transcribes voice messages, audio files, and video notes into text.**<br>
Choose between **local Whisper** (offline, private) or **Google Gemini** (fast, free tier).

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Telegram Bot API](https://img.shields.io/badge/Telegram-Bot_API-26A5E4?logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Created by [Dmytro](https://links.demitrich.od.ua/)**

<br>

[Quick Start](#-quick-start) · [Configuration](#%EF%B8%8F-configuration) · [Engines](#-engines) · [Docker Commands](#-docker-commands)

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

**Dual Engine Support**
- 🔇 **Whisper** — fully offline, privacy-first
- ☁️ **Gemini** — cloud-based, blazing fast

</td>
<td width="50%">

**Wide Format Support**
- 🎤 Voice messages
- 🎵 Audio files (MP3, WAV, FLAC, AAC, M4A)
- 📹 Video notes & videos

</td>
</tr>
<tr>
<td>

**Easy Deployment**
- 🐳 One-command Docker setup
- 🔧 All config via `.env` file
- 🔄 Auto-restart on failure

</td>
<td>

**Access Control**
- 🔒 Optional user whitelist
- 📊 Per-message transcription stats
- 📝 Structured logging

</td>
</tr>
</table>

---

## 🚀 Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/whisper-telegram-bot.git
cd whisper-telegram-bot
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:

```env
BOT_TOKEN=123456:ABC-DEF...    # from @BotFather
STT_ENGINE=gemini               # or "whisper" or "qwen"
GEMINI_API_KEY=AIza...          # from https://aistudio.google.com/apikey
```

**3. Launch**

```bash
docker compose up -d
```

That's it! Send a voice message to your bot and get the transcription back.

---

## ⚙️ Configuration

All settings are defined in the `.env` file:

| Variable | Default | Description |
|:---|:---|:---|
| `BOT_TOKEN` | — | **Required.** Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `ALLOWED_USERS` | *(empty)* | Comma-separated Telegram user IDs. Empty = allow everyone |
| `STT_ENGINE` | `whisper` | Speech-to-text engine: `gemini`, `whisper` or `qwen` |

### Gemini Settings

| Variable | Default | Description |
|:---|:---|:---|
| `GEMINI_API_KEY` | — | **Required for Gemini.** [Get your key](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GEMINI_LANGUAGE` | `English` | Target transcription language |

### Whisper Settings

| Variable | Default | Description |
|:---|:---|:---|
| `WHISPER_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Any [faster-whisper](https://github.com/SYSTRAN/faster-whisper) compatible model |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` (GPU) |
| `WHISPER_COMPUTE` | `int8` | Compute type: `int8`, `float16`, `float32` |
| `WHISPER_LANGUAGE` | `auto` | Language code or `auto` for auto-detection |
| `WHISPER_BEAM_SIZE` | `5` | Beam search size (higher = more accurate, slower) |

### Qwen3-ASR Settings

Only used with `STT_ENGINE=qwen`. The image must be built with `WITH_QWEN=1` (see [Qwen3-ASR](#qwen3-asr-local-gpu)).

| Variable | Default | Description |
|:---|:---|:---|
| `WITH_QWEN` | `0` | **Build-time.** `1` installs `torch` + `qwen-asr` into the image (large). Rebuild after changing |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-1.7B` or the lighter `Qwen/Qwen3-ASR-0.6B` |
| `QWEN_DEVICE` | `cuda:0` | `cuda:0` or `cpu` |
| `QWEN_DTYPE` | `bfloat16` | `bfloat16`, `float16`, `float32` |
| `QWEN_LANGUAGE` | `auto` | `auto` or a language name, e.g. `Japanese`, `English`, `Russian` |
| `QWEN_MAX_NEW_TOKENS` | `512` | Max generated tokens per chunk |
| `QWEN_CHUNK_SECONDS` | `30` | Max chunk length. Long audio is split at speech pauses (Silero VAD), chunks are also the unit for subtitles |
| `QWEN_BATCH_SIZE` | `4` | Chunks per inference batch. Lower it if you run out of VRAM |
| `QWEN_IDLE_TIMEOUT` | `600` | Seconds of inactivity before the model is unloaded |

---

## 🔊 Engines

### Gemini (Recommended for most users)

Uses Google's Gemini API for transcription. Fast, accurate, and free within limits.

| | |
|:---|:---|
| **Speed** | ⚡ ~2-5 seconds |
| **Privacy** | Audio sent to Google servers |
| **Free Tier** | 15 RPM / 1M tokens per day |
| **Languages** | 100+ languages |
| **Setup** | Just an API key |

### Whisper (Privacy-first)

Runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) locally. No data leaves your server.

| | |
|:---|:---|
| **Speed** | 🐢 ~10-30 seconds (CPU) |
| **Privacy** | ✅ Fully offline |
| **Cost** | Free forever |
| **Languages** | 99 languages |
| **Setup** | Model downloads on first run (~1 GB) |

### Qwen3-ASR (local, GPU)

Runs [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) locally via `transformers`. Like Whisper, it runs in an isolated worker process that exits when idle, so RAM and VRAM are returned to the OS.

| | |
|:---|:---|
| **Privacy** | ✅ Fully offline |
| **Hardware** | NVIDIA GPU recommended. The 1.7B model is ~4.4 GB of weights in bf16, so on 6 GB cards lower `QWEN_BATCH_SIZE` or use `Qwen3-ASR-0.6B` |
| **Languages** | 30 (see below) |
| **Setup** | Build with `WITH_QWEN=1` (image grows to ~10 GB), model downloads on first request (~4.4 GB) |

**Supported languages:** Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian, Romanian.
**Not supported:** Ukrainian, Hebrew (use `STT_ENGINE=whisper` for those). The `WHISPER_MODEL2` Hebrew model is not used in `qwen` mode. Full list: [model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

**Enable it:**

```env
STT_ENGINE=qwen
WITH_QWEN=1
```

```bash
docker compose up -d --build
```

**Subtitles (SRT for videos).** Qwen3-ASR returns plain text without timestamps, so the bot derives timing from the audio:

1. Silero VAD (bundled with `faster-whisper`) finds speech; the audio is cut at pauses, never mid-phrase, into chunks of up to `QWEN_CHUNK_SECONDS`.
2. Speech boundaries are refined by RMS energy (adaptive noise floor): starts may move back up to 0.6 s, ends forward up to 1 s, never into the neighbouring phrase.
3. The text of each chunk is split into subtitle lines (sentences, long ones at commas, max ~84 characters or ~42 for CJK).
4. Lines share the chunk's speech time proportionally to their text length, and a line border close to a real pause (within 0.8 s) is snapped to that pause.

Timing is therefore an approximation, not true word alignment: it is accurate at pauses and less so inside a long uninterrupted phrase. If VAD finds no speech, the audio is cut into fixed chunks at quiet points instead, so nothing is dropped. This applies only to the `qwen` engine; Whisper uses its own segment timestamps.

If `STT_ENGINE=qwen` but the image was built without `WITH_QWEN=1`, the bot exits at startup and tells you to rebuild.

**Model cache.** Models are cached in the shared `hf-cache` Docker volume, see [Where Models Are Stored](#-where-models-are-stored). Loading Qwen from a Windows bind mount took ~218 s in our test vs ~5.5 s from the volume.

---

## 💾 Where Models Are Stored

All downloaded models (Whisper and Qwen) are stored in one Docker volume, `hf-cache`, mounted at `/root/.cache/huggingface` (the standard Hugging Face cache, `hub/models--<org>--<name>`). Gemini is a cloud API and stores nothing.

- **Name on the host:** `<project>_hf-cache`. `<project>` is the compose project name, by default the folder name, so `whisper-telegram-bot_hf-cache`.
- **Location:** inside Docker's virtual disk (WSL2), **not** in the project folder, so it is not visible in Explorer.
- **Why a volume and not `./models`:** `transformers` (Qwen) memory-maps the weights, which is very slow on a Windows bind mount.
- Models are downloaded automatically on first use and survive `docker compose down` and rebuilds. Only `docker compose down -v` or `docker volume rm` deletes them.

**Inspect and clean up:**

```bash
docker system df -v                       # volume size
docker run --rm -v whisper-telegram-bot_hf-cache:/d alpine du -sh /d/hub/*     # what is inside
docker run --rm -v whisper-telegram-bot_hf-cache:/d alpine rm -rf /d/hub/models--Systran--faster-whisper-large-v2   # delete one model
docker compose down && docker volume rm whisper-telegram-bot_hf-cache          # delete all models
```

Deleting a Docker volume does not shrink `docker_data.vhdx` on Windows; compact the disk in Docker Desktop (Settings → Resources) to get the space back.

**Migrating models from an older `./models` folder** (earlier versions bind-mounted it): copy them into the volume once, `./models` itself is not modified.

```bash
docker compose up -d --no-start
docker run --rm -v "$(pwd)/models:/src:ro" -v whisper-telegram-bot_hf-cache:/dst --entrypoint sh whisper-telegram-bot-bot \
  -c 'mkdir -p /dst/hub && cp -a /src/hub/models--* /dst/hub/'
```

After checking the bot works you can delete `./models`.

---

## 🐳 Docker Commands

```bash
docker compose up -d          # Start in background
docker compose logs -f        # Follow live logs
docker compose watch         # Auto-sync *.py changes and restart the bot
docker compose restart        # Restart after .env changes
docker compose down           # Stop the bot
docker compose up -d --build  # Rebuild after code changes
```

For development, `docker compose watch` syncs the project into the container and restarts the bot automatically while ignoring non-code files such as `models/`, `README.md`, and Docker metadata. Changes to `requirements.txt` or `requirements-qwen.txt` trigger a rebuild.

---

## 📁 Project Structure

```
whisper-telegram-bot/
├── bot.py               # Bot entry point, picks the STT engine
├── engines/             # whisper_engine.py, gemini_engine.py, qwen_engine.py
├── Dockerfile           # Multi-stage container image definition
├── docker-compose.yml   # Service orchestration
├── requirements.txt     # Python dependencies
├── requirements-qwen.txt # Optional Qwen3-ASR dependencies (WITH_QWEN=1)
├── .env.example         # Environment template
├── .dockerignore        # Docker build exclusions
├── .gitignore           # Git exclusions
└── LICENSE              # MIT License
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using [python-telegram-bot](https://python-telegram-bot.org/) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [Google Gemini](https://ai.google.dev/)**

</div>
