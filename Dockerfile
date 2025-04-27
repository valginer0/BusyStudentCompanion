# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Set work directory
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

# Copy requirements
COPY requirements.txt requirements.txt
COPY requirements-cpu.txt requirements-cpu.txt
COPY requirements-gpu.txt requirements-gpu.txt

ARG TARGET=cpu

# Install Python dependencies (CPU by default; can be overridden at build time)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    if [ "$TARGET" = "gpu" ]; then pip install -r requirements-gpu.txt; else pip install -r requirements-cpu.txt; fi

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to run Streamlit app
CMD ["streamlit", "run", "src/book_to_essay/streamlit_app.py"]
