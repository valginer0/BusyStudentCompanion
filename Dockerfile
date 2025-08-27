# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Set environment variables to force CPU
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/cache/huggingface
ENV FORCE_CUDA=0
ENV CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt before installing dependencies
COPY requirements.txt requirements.txt

# Install CPU version of torch and dependencies FIRST
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Replace host config with Docker-specific defaults
COPY src/book_to_essay/config.docker.py /app/src/book_to_essay/config.py

EXPOSE 8501

CMD ["streamlit", "run", "src/book_to_essay/streamlit_app.py"]
