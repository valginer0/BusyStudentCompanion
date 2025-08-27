"""Configuration settings for the Book to Essay Generator."""
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

# Load environment variables from a .env file if present (host development convenience)
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# -----------------------------------------------------------------------------
# Model Settings (static defaults for host usage)
# -----------------------------------------------------------------------------
# MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"  # Larger DeepSeek model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Default to Mistral for testing
# MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"  # Smaller DeepSeek model

# Default model generation parameters
MAX_LENGTH = 2048
TEMPERATURE = 0.7

# Chunking settings for large texts
MAX_CHUNK_SIZE = 1500  # Max size of text chunks when processing big docs
MAX_CHUNKS_PER_ANALYSIS = 5  # Max number of chunks to analyze per essay

# -----------------------------------------------------------------------------
# Cache Settings (host default: XDG-like path under ~/.cache)
# -----------------------------------------------------------------------------
MODEL_CACHE_ROOT = os.getenv('MODEL_CACHE_ROOT', os.path.expanduser('~/.cache/busy_student_companion/models'))
# Do not append model-specific subfolders; let HuggingFace manage them
MODEL_CACHE_DIR = MODEL_CACHE_ROOT
CONTENT_CACHE_DIR = os.path.join(MODEL_CACHE_ROOT, 'content')
# Ensure directories exist
os.makedirs(MODEL_CACHE_ROOT, exist_ok=True)
os.makedirs(CONTENT_CACHE_DIR, exist_ok=True)

# Check for GPU and bitsandbytes
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    
    # More robust check for bitsandbytes
    HAS_BITSANDBYTES = False
    try:
        import bitsandbytes as bnb
        HAS_BITSANDBYTES = True
    except ImportError:
        pass
except ImportError:
    HAS_GPU = False
    HAS_BITSANDBYTES = False
    logger.warning("PyTorch not available, defaulting to CPU mode")

logger.info(f"Environment detected: GPU={HAS_GPU}, BitsAndBytes={HAS_BITSANDBYTES}")

def safe_model_cache_dir(model_name):
    # Convert model name to a safe directory string
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model_name)

# -----------------------------------------------------------------------------
# App Settings
# -----------------------------------------------------------------------------
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 50))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4000))

# -----------------------------------------------------------------------------
# File Settings
# -----------------------------------------------------------------------------
SUPPORTED_FORMATS = ['.pdf', '.txt', '.epub']
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'output'

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuantizationConfig:
    """Configuration for model quantization."""
    
    @staticmethod
    def get_config():
        """Get the appropriate quantization configuration based on environment."""
        if HAS_GPU and HAS_BITSANDBYTES:
            logger.info("Using 4-bit quantization with bitsandbytes")
            return {
                "method": "4bit",
                "load_config": {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    ),
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "low_cpu_mem_usage": True
                },
                "post_load_quantize": None
            }
        elif not HAS_GPU:
            logger.info("Using standard CPU configuration (no quantization)")
            return {
                "method": "cpu",
                "load_config": {
                    "device_map": "cpu",
                    "low_cpu_mem_usage": True
                },
                "post_load_quantize": None
            }
        else:
            logger.info("Using 16-bit floating point (GPU)")
            return {
                "method": "fp16",
                "load_config": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                },
                "post_load_quantize": None
            }

# Get quantization configuration
QUANT_CONFIG = QuantizationConfig.get_config()
