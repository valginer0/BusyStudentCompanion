# BusyStudentCompanion Docker Setup Guide

This guide explains how to run BusyStudentCompanion using Docker and Docker Compose for both CPU and GPU environments.

## Prerequisites

1. Docker and Docker Compose installed
2. (For GPU) NVIDIA drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. Optional: HuggingFace cache directory for model weights

## Quick Start

1. Copy or update your `.env` file if needed (not required for basic Streamlit usage).
2. Build and run with Docker Compose:

```bash
# For CPU version:
docker-compose up app-cpu

# For GPU version (requires NVIDIA Docker):
docker-compose up app-gpu

# For GPU code testing on CPU:
docker-compose up app-gpu-test
```

The Streamlit app will be available at:
- [http://localhost:8501](http://localhost:8501) (CPU)
- [http://localhost:8502](http://localhost:8502) (GPU)
- [http://localhost:8503](http://localhost:8503) (GPU test)

## Volumes and Caching

Model and HuggingFace caches are persisted using Docker volumes:
- `model-cache`: `/cache/torch`
- `huggingface-cache`: `/cache/huggingface`

This avoids repeated downloads of large model files.

## Customization

- Modify `docker-compose.yml` for custom ports, cache locations, or commands.
- Edit `requirements.txt`, `requirements-cpu.txt`, or `requirements-gpu.txt` as needed.

## Troubleshooting

- Ensure Docker and (for GPU) NVIDIA Docker are installed and running.
- If you see CUDA errors on CPU, use the CPU service or test service.
- For persistent cache, do not remove the Docker volumes.

---

For more details, see the main README or contact the maintainer.
