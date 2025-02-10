"""DeepSeek model handler for essay generation."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .config import MODEL_NAME, MAX_LENGTH, TEMPERATURE

class DeepSeekHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
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
