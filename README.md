# BusyStudentCompanion

A powerful AI-powered tool that helps students analyze books and generate well-structured essays. Using state-of-the-art AI technology (**Mistral** or **DeepSeek** model, configurable), this companion tool streamlines the process of understanding literature and creating thoughtful essay content.

## Features

- Analyze books and generate essays with either **Mistral** or **DeepSeek** language models (selectable in configuration)
- MLA-compliant citations and Works Cited generation
- Supports TXT, PDF, and EPUB formats
- Thesis-driven, evidence-based essay generation
- Streamlit-based graphical user interface for easy use

## Installation

1. Clone the repository:
```bash
git clone https://github.com/valginer0/BusyStudentCompanion.git
cd BusyStudentCompanion
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# First install base requirements
pip install -r requirements.txt

# Then, based on your system:
# For systems with NVIDIA GPU:
pip install -r requirements-gpu.txt

# OR for CPU-only systems:
pip install -r requirements-cpu.txt
```

4. Set up environment variables and select model:
```bash
cp .env.example .env
# Edit .env file with your API keys and configuration
# Set the MODEL_NAME variable to either 'mistralai/Mistral-7B-Instruct-v0.1' or 'deepseek-ai/deepseek-llm-7b-base'
```

## Model Selection

You can choose between **Mistral** and **DeepSeek** models for essay generation. To change the model, edit the `MODEL_NAME` variable in your `.env` file or in `src/book_to_essay/config.py`:

```
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # For Mistral
# or
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"   # For DeepSeek
```

## Model Availability

Both the **Mistral** and **DeepSeek** models are free to use for research and non-commercial purposes:

- [Mistral-7B-Instruct-v0.1 on Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) ([License](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/LICENSE))
- [DeepSeek LLM-7B on Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base) ([License](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/blob/main/LICENSE))

## Model Download

You do **not** need to manually download the models.  
**On first run, the application will automatically download the selected model** (as specified by `MODEL_NAME` in your `.env` or `config.py`). The download is ~4GB and will be cached for future runs.

## First Run Notice

On the first run, the application will:
1. Download the selected language model (optimized size: ~4GB)
2. The download may take 15-30 minutes depending on your internet speed
3. The model will be cached locally and subsequent runs will start immediately
4. Memory usage is optimized using 4-bit quantization while maintaining high quality

## Caching System

The application uses a two-level caching system:

1. **Model Caching**: Downloaded model files are stored in subdirectories such as:
   - `/home/user/.cache/busy_student_companion/models/models--deepseek-ai--*`
   - `/home/user/.cache/busy_student_companion/models/models--mistralai--*`
2. **Generation Caching**: Results of chunk analysis are stored in `/home/user/.cache/busy_student_companion/models/chunk_cache/*.pkl` files

When developing or testing changes to the essay generation or filtering logic:

```bash
# Clear ONLY the chunk cache to test changes to generation logic
rm -f ~/.cache/busy_student_companion/models/chunk_cache/*.pkl

# DO NOT delete model files as they are large and time-consuming to re-download
```

**Note**: If changes to the essay generation or filtering logic do not appear to take effect, it is likely because the system is using cached results. Always clear the chunk cache before testing.

## Book/Text File Metadata Requirements

For best results and robust MLA citation extraction, each input text file should include the following metadata at the top of the file:

```
Title: <Book Title>
Author: <Author Name>
```

Example:
```
Title: Romeo and Juliet
Author: William Shakespeare
```

- The tool will extract `Title` and `Author` from the first 40 lines of the file.
- If not found, it will fallback to parsing the filename (format: `Author - Title[ - Extra].ext`).

**Supported formats:** `.txt`, `.pdf`, `.epub`

## Usage

1. Start the web interface:
```bash
# Add project root to PYTHONPATH and run the app
PYTHONPATH=$PYTHONPATH:$(pwd) streamlit run src/book_to_essay/streamlit_app.py
```

2. Upload your book file (PDF, EPUB, or DOCX format)
3. The AI will analyze the content and generate an essay
4. Review and download the generated essay

## Container Usage

You can run BusyStudentCompanion in a Docker container for easy sharing and deployment.

### Build the Docker image (CPU, default)

```bash
docker build -t busystudentcompanion:cpu .
```

### Build the Docker image (GPU, NVIDIA CUDA)

```bash
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04 --build-arg TARGET=gpu -t busystudentcompanion:gpu .
```

### GPU Support (NVIDIA)

To use GPU acceleration inside Docker containers, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) **once on your computer**. This allows Docker to access your NVIDIA GPU.

- You only need to install the toolkit once per machine.
- Follow the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your operating system.
- After installation, you can run GPU-enabled containers with `--gpus all`.

If you do not install the toolkit, GPU support will not be available, but you can still run the app on CPU.

### Run the container (GPU)

- Requires NVIDIA Container Toolkit (see above)

```bash
docker run --gpus all -p 8501:8501 --rm busystudentcompanion:gpu
```

- The app will be available at [http://localhost:8501](http://localhost:8501)

### Run the container (CPU)

```bash
docker run -p 8501:8501 --rm busystudentcompanion:cpu
```

### Recommended: Persist Model Cache

To avoid re-downloading large model files every time you run the container, **mount a cache directory** from your host:

```bash
# CPU
docker run -p 8501:8501 -v /your/cache/dir:/cache --rm busystudentcompanion:cpu

# GPU
docker run --gpus all -p 8501:8501 -v /your/cache/dir:/cache --rm busystudentcompanion:gpu
```

Replace `/your/cache/dir` with a directory path on your host machine (e.g., `/home/youruser/bsc_cache`).

### Running from a Container Registry

If the image is already published to a registry (e.g., Docker Hub):

```bash
# Pull the image (CPU example)
docker pull yourdockerhubusername/busystudentcompanion:cpu

# Run it (with cache recommended)
docker run -p 8501:8501 -v /your/cache/dir:/cache --rm yourdockerhubusername/busystudentcompanion:cpu
```

For GPU:

```bash
docker pull yourdockerhubusername/busystudentcompanion:gpu
docker run --gpus all -p 8501:8501 -v /your/cache/dir:/cache --rm yourdockerhubusername/busystudentcompanion:gpu
```

Replace `yourdockerhubusername` with your Docker Hub username or your target registry.

### Push to a registry (example: Docker Hub)

```bash
docker tag busystudentcompanion:cpu yourdockerhubusername/busystudentcompanion:cpu
# or for GPU
# docker tag busystudentcompanion:gpu yourdockerhubusername/busystudentcompanion:gpu
docker push yourdockerhubusername/busystudentcompanion:cpu
# docker push yourdockerhubusername/busystudentcompanion:gpu
```

Replace `yourdockerhubusername` with your Docker Hub username or your target registry.

## Running Scripts (WSL Ubuntu on Windows 11)

Use the following command pattern to run scripts (from Windows):

```
wsl bash -c 'cd /home/val/projects/BusyStudentCompanion; source .venv/bin/activate; PYTHONPATH=$PYTHONPATH:/home/val/projects/BusyStudentCompanion pytest tests'
```

Example:
```
wsl bash -c 'cd /home/val/projects/BusyStudentCompanion; source .venv/bin/activate; PYTHONPATH=$PYTHONPATH:/home/val/projects/BusyStudentCompanion python3 test_essay_generation.py'
```

- Use UNC paths (e.g., `\\wsl$\\Ubuntu\\home\\val\\projects\\BusyStudentCompanion\\...`) for file/codebase operations.
- Use WSL bash commands for running scripts.

## Project Structure

```
BusyStudentCompanion/
├── src/
│   └── book_to_essay/
│       ├── ai_book_to_essay_generator.py  # Main AI processing logic
│       ├── cache_manager.py              # Content caching system
│       ├── config.py                     # Configuration management
│       ├── model_handler.py              # DeepSeek model integration
│       └── streamlit_app.py             # Web interface
├── .env.example                         # Example environment variables
├── requirements.txt                     # Python dependencies
└── README.md                           # Project documentation
```

## Configuration

The following environment variables can be configured in `.env`:

- `CACHE_DIR`: Directory for cached content (default: `.cache`)
- `MAX_CACHE_SIZE`: Maximum cache size in MB (default: 1000)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek for their powerful language model
- Streamlit for the web interface framework
- PyMuPDF and ebooklib for document processing capabilities