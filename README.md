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
STT_ENGINE=gemini               # or "whisper"
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
| `STT_ENGINE` | `whisper` | Speech-to-text engine: `gemini` or `whisper` |

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

For development, `docker compose watch` syncs the project into the container and restarts the bot automatically while ignoring non-code files such as `models/`, `README.md`, and Docker metadata. Changes to `requirements.txt` trigger a rebuild.

---

## 📁 Project Structure

```
whisper-telegram-bot/
├── bot.py               # Main bot logic with Whisper & Gemini engines
├── Dockerfile           # Container image definition
├── docker-compose.yml   # Service orchestration
├── requirements.txt     # Python dependencies
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
