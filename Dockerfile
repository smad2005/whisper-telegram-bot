FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Install python and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py config.py ./
COPY engines ./engines
COPY handlers ./handlers
COPY services ./services

CMD ["python3", "bot.py"]
