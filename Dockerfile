# ---- build stage: install python deps into a venv ----
FROM ubuntu:24.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- runtime stage: CUDA runtime + python + ffmpeg only ----
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY bot.py config.py ./
COPY engines ./engines
COPY handlers ./handlers
COPY services ./services

CMD ["python3", "bot.py"]
