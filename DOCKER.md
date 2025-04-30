# BusyStudentCompanion Docker Setup Guide

This guide explains how to run BusyStudentCompanion using Docker and Docker Compose for both CPU and GPU environments.

## Prerequisites

1. Docker and Docker Compose installed
2. (For GPU) NVIDIA drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. Optional: HuggingFace cache directory for model weights

## Quick Start

1. Copy or update your `.env` file if needed (not required for basic Streamlit usage).
2. **Pull** the pre-built image(s) (optional but faster):
    ```bash
    # Authenticate once per machine
    echo <GITHUB_PAT> | docker login ghcr.io -u <github_username> --password-stdin
    
    # CPU image
    docker-compose pull app-cpu
    # GPU image (optional, requires NVIDIA Docker)
    docker-compose pull app-gpu
    ```
3. **Run with Docker Compose** (handles volumes & ports automatically):
    ```bash
    # CPU version
    docker-compose up -d app-cpu
    
    # GPU version (requires NVIDIA Docker)
    docker-compose up -d app-gpu
    ```

Or, to build locally instead of pulling:
```bash
# Build CPU
docker-compose build app-cpu
# Build GPU
docker-compose build app-gpu
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
