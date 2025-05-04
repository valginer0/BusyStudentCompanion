# BusyStudentCompanion Docker Setup Guide

This guide explains how to run BusyStudentCompanion using Docker and Docker Compose for both CPU and GPU environments.

## Available Docker Images

The project includes three Docker configurations:

1. **CPU Image** (`app-cpu`): Standard CPU-only version for most users
2. **GPU Image - BETA** (`app-gpu`): GPU-accelerated version requiring NVIDIA hardware
3. **GPU Test Image** (`app-gpu-test`): For testing GPU code on CPU systems

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
    
    # Test GPU code on CPU
    docker-compose up -d app-gpu-test
    ```

Or, to build locally instead of pulling:
```bash
# Build CPU
docker-compose build app-cpu
# Build GPU (beta)
docker-compose build app-gpu
# Build GPU test
docker-compose build app-gpu-test
```

The Streamlit app will be available at:
- [http://localhost:8501](http://localhost:8501) (CPU)
- [http://localhost:8502](http://localhost:8502) (GPU - beta)
- [http://localhost:8503](http://localhost:8503) (GPU test on CPU)

## GPU Support - Important Notes

The GPU version (`app-gpu`) includes:

1. **Automatic GPU verification**: The container will check if your GPU is properly configured and provide helpful diagnostics if issues are detected
2. **Beta status**: This image is currently in beta as it may not have been fully tested on all GPU configurations
3. **Graceful fallback**: If GPU is not available, the application will still run but will warn about reduced performance

Common issues when using the GPU version:
- Missing NVIDIA Container Toolkit
- Incompatible CUDA drivers
- Using `--gpus all` flag when running with `docker run` directly

## Testing GPU Code Without GPU Hardware

If you want to verify that your code structure works correctly but don't have a GPU:

1. Use the `app-gpu-test` service: `docker-compose up app-gpu-test`
2. This runs on CPU but has the environments configured to simulate GPU code paths
3. Useful for testing before deploying to GPU environments

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
- Check GPU logs with: `docker-compose logs app-gpu` to see if GPU was detected

---

For more details, see the main README or contact the maintainer.
