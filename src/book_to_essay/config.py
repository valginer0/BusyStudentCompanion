"""Configuration settings for the Book to Essay Generator."""
import os
import logging
import torch
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for GPU and bitsandbytes
HAS_GPU = torch.cuda.is_available()
HAS_BITSANDBYTES = False
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    pass

# Log environment detection
logger.info(f"Environment detected: GPU={HAS_GPU}, BitsAndBytes={HAS_BITSANDBYTES}")

# Model Settings
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
MAX_LENGTH = 2048
TEMPERATURE = 0.7

class QuantizationConfig:
    """Configuration for model quantization."""
    
    @staticmethod
    def get_config():
        """Get the appropriate quantization configuration based on environment."""
        if HAS_GPU and HAS_BITSANDBYTES:
            logger.info("Using 4-bit quantization with bitsandbytes")
            return {
                "method": "4bit",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }
        elif not HAS_GPU:
            logger.info("Using 8-bit dynamic quantization for CPU")
            return {
                "method": "8bit_cpu",
                "device_map": "cpu",
                "low_cpu_mem_usage": True
            }
        else:
            logger.info("Using FP16 quantization")
            return {
                "method": "fp16",
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }

# Get quantization configuration
QUANT_CONFIG = QuantizationConfig.get_config()

# Cache Settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')  # Specific directory for model files
CONTENT_CACHE_DIR = os.path.join(CACHE_DIR, 'content')  # Directory for processed content
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(CONTENT_CACHE_DIR, exist_ok=True)

# App Settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 50))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4000))

# File Settings
SUPPORTED_FORMATS = ['.pdf', '.txt', '.epub']
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'output'

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
