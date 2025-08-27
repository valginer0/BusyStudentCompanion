"""Docker-specific configuration for BusyStudentCompanion.
This file is identical to the host `config.py` except that its default
cache path points inside the container volume (/app/cache/models).
The Dockerfile copies this file over the host version during image build so
that application code continues to import `src.book_to_essay.config`.
"""
import os
import logging
import torch
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if one is baked into the image or supplied via bind mount.
load_dotenv()

# -----------------------------------------------------------------------------
# Model Settings (Docker default)
# -----------------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
MAX_LENGTH = 2048
TEMPERATURE = 0.7

# Chunking settings
MAX_CHUNK_SIZE = 1500
MAX_CHUNKS_PER_ANALYSIS = 5

# -----------------------------------------------------------------------------
# Cache Settings (Docker volume mounted at /cache)
# -----------------------------------------------------------------------------
MODEL_CACHE_ROOT = os.getenv('MODEL_CACHE_ROOT', '/app/cache/models')
MODEL_CACHE_DIR = MODEL_CACHE_ROOT
CONTENT_CACHE_DIR = os.path.join(MODEL_CACHE_ROOT, 'content')

os.makedirs(MODEL_CACHE_ROOT, exist_ok=True)
os.makedirs(CONTENT_CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Environment detection (GPU, bitsandbytes)
# -----------------------------------------------------------------------------
try:
    HAS_GPU = torch.cuda.is_available()
    HAS_BITSANDBYTES = False
    try:
        import bitsandbytes as bnb  # noqa: F401
        HAS_BITSANDBYTES = True
    except ImportError:
        pass
except ImportError:
    HAS_GPU = False
    HAS_BITSANDBYTES = False
    logger.warning("PyTorch not available, defaulting to CPU mode")

logger.info(f"Environment detected (Docker): GPU={HAS_GPU}, BitsAndBytes={HAS_BITSANDBYTES}")

def safe_model_cache_dir(model_name: str) -> str:
    """Sanitise model name so it can be used as a directory."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model_name)

# -----------------------------------------------------------------------------
# App & file settings (same as host)
# -----------------------------------------------------------------------------
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 50))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4000))

SUPPORTED_FORMATS = ['.pdf', '.txt', '.epub']
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'output'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuantizationConfig:
    """Configuration for model quantisation inside Docker."""
    @staticmethod
    def get_config():
        if HAS_GPU and HAS_BITSANDBYTES:
            logger.info("Using 4-bit quantisation with bitsandbytes (Docker)")
            return {
                "method": "4bit",
                "load_config": {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    ),
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "low_cpu_mem_usage": True,
                },
                "post_load_quantize": None,
            }
        elif not HAS_GPU:
            logger.info("Using standard CPU configuration (Docker)")
            return {
                "method": "cpu",
                "load_config": {"device_map": "cpu", "low_cpu_mem_usage": True},
                "post_load_quantize": None,
            }
        else:
            logger.info("Using FP16 (Docker GPU)")
            return {
                "method": "fp16",
                "load_config": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                },
                "post_load_quantize": None,
            }

# Export
QUANT_CONFIG = QuantizationConfig.get_config()
