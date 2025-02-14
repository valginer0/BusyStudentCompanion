"""Configuration settings for the Book to Essay Generator."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Settings
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
MAX_LENGTH = 2048
TEMPERATURE = 0.7

# Quantization Settings
USE_INT4_QUANTIZATION = True     # Enable 4-bit quantization for ~70% size reduction
COMPUTE_DTYPE = "float16"        # Use float16 for initial loading
QUANT_TYPE = "nf4"              # "nf4" (normal float) has better accuracy than "fp4"
USE_NESTED_QUANT = True         # Enable nested quantization for maximum size reduction
LOAD_QUANTIZED = True           # Try to load pre-quantized weights if available

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
