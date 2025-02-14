"""DeepSeek model handler for essay generation."""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.book_to_essay.config import (
    MODEL_NAME, MAX_LENGTH, TEMPERATURE,
    USE_INT4_QUANTIZATION, COMPUTE_DTYPE,
    QUANT_TYPE, USE_NESTED_QUANT, MODEL_CACHE_DIR
)

class DeepSeekHandler:
    def __init__(self):
        try:
            # First try to load with quantization
            if USE_INT4_QUANTIZATION:
                try:
                    import bitsandbytes as bnb
                    # Configure 4-bit quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,  # Use 4-bit instead of 8-bit
                        bnb_4bit_compute_dtype=getattr(torch, COMPUTE_DTYPE),
                        bnb_4bit_quant_type=QUANT_TYPE,
                        bnb_4bit_use_double_quant=USE_NESTED_QUANT
                    )
                    model_kwargs = {
                        "quantization_config": quantization_config,
                        "load_in_4bit": True,  # Add this at top level
                        "low_cpu_mem_usage": True
                    }
                except ImportError:
                    print("Warning: bitsandbytes not properly installed, falling back to FP16")
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True
                    }
            else:
                # Use regular FP16 if quantization is disabled
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }

            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Then load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",  # Will automatically use GPU if available
                cache_dir=MODEL_CACHE_DIR,
                **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {str(e)}")
        
    def generate_essay(self, context: str, prompt: str, max_length: int = MAX_LENGTH) -> str:
        """Generate an essay using the DeepSeek model."""
        instruction = f"""You are a skilled academic writer. {prompt}

Please ensure that:
1. All quotes are properly integrated into the text
2. Each quote has an MLA in-text citation (Author Page)
3. The essay follows proper MLA formatting
4. The Works Cited section at the end follows MLA format exactly

Context:
{context}"""
        
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
