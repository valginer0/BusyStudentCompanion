"""
Model loader utility for BusyStudentCompanion.
Handles model and tokenizer loading, including quantization.
"""
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from .config import MODEL_NAME, MODEL_CACHE_DIR, QUANT_CONFIG

logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR):
    logger.info(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

def load_model(model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR, quant_config: dict = QUANT_CONFIG):
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        **quant_config.get("load_config", {})
    )
    # Apply post-load quantization if specified
    if quant_config.get("post_load_quantize", False):
        logger.info("Applying post-load quantization...")
        config = quant_config.get("post_load_quantize", {})
        if config:
            # Apply dynamic quantization for CPU
            if quant_config.get("method") == "8bit_cpu":
                logger.info("Applying dynamic quantization for CPU...")
                model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
    return model
