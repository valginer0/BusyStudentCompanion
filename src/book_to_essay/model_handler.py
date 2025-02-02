"""DeepSeek model handler for essay generation."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_NAME, MAX_LENGTH, TEMPERATURE

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
        instruction = f"""Based on the following context, write an essay addressing this prompt: {prompt}
        
        Context:
        {context}
        
        Write a well-structured essay with proper citations and references."""
        
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
