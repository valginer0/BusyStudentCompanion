"""Configuration settings for the Book to Essay Generator."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Settings
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct"
MAX_LENGTH = 2048
TEMPERATURE = 0.7

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

# Cache Settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
