"""Handler for loading and managing the DeepSeek model."""
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.quantization
from src.book_to_essay.config import (
    MODEL_NAME, MAX_LENGTH, TEMPERATURE,
    MODEL_CACHE_DIR, QUANT_CONFIG
)

logger = logging.getLogger(__name__)

class DeepSeekHandler:
    """Handler for the DeepSeek model."""
    
    def __init__(self):
        """Initialize the model and tokenizer with appropriate quantization."""
        try:
            logger.info("Loading model and tokenizer...")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Load model with quantization config
            logger.info("Loading model with quantization config...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                **QUANT_CONFIG["load_config"]
            )
            
            # Initialize text_chunks
            self.text_chunks = []
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
    def process_text(self, text: str) -> None:
        """Process input text and store it for later use."""
        logger.info("Processing text input...")
        
        # Get the tokenizer's maximum length
        max_length = self.tokenizer.model_max_length
        if max_length > 4096:  # Some models report very large max lengths
            max_length = 4096
            
        # Calculate chunk size with overlap
        chunk_size = max_length - 200  # Leave room for prompts and overlap
        overlap_size = 100  # Number of tokens to overlap between chunks
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        logger.info(f"Text tokenized into {len(tokens)} tokens")
        
        # Split into chunks with overlap
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            if start > 0:  # Not the first chunk
                start = start - overlap_size
            chunk = tokens[start:end]
            chunks.append(chunk)
            start = end
            
        # Convert chunks back to text
        text_chunks = [self.tokenizer.decode(chunk) for chunk in chunks]
        
        # Store chunks for later use
        self.text_chunks = text_chunks
        logger.info(f"Text split into {len(text_chunks)} chunks and stored in self.text_chunks")
        logger.info(f"Text chunks assignment: {self.text_chunks}")

    def generate_essay(self, topic: str, word_limit: int = 500) -> str:
        """Generate an essay on the given topic using the loaded text."""
        if not hasattr(self, 'text_chunks'):
            raise ValueError("No text has been loaded. Please load text first.")

        # Ensure word_limit is an integer
        word_limit = int(word_limit)
        logger.info(f"Generating essay on topic: {topic} with word limit: {word_limit}")
        
        # Process each chunk
        summaries = []
        for i, chunk in enumerate(self.text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(self.text_chunks)}")
            
            prompt = f"""Based on the following text, help me write part of an essay about {topic}.
            Extract only the relevant information from this part of the text.
            Text: {chunk}"""
            
            try:
                response = self.model.generate(
                    **self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096),
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True
                )
                summary = self.tokenizer.decode(response[0], skip_special_tokens=True)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue

        # Combine summaries into final essay
        combined_prompt = f"""Using the following extracted information, write a coherent essay about {topic}.
        Keep it under {word_limit} words and make it flow naturally.
        Information: {' '.join(summaries)}"""
        
        try:
            response = self.model.generate(
                **self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=4096),
                max_new_tokens=min(1500, word_limit * 2),  # Reasonable limit based on word_limit
                temperature=0.7,
                do_sample=True
            )
            essay = self.tokenizer.decode(response[0], skip_special_tokens=True)
            return essay
        except Exception as e:
            logger.error(f"Error generating final essay: {str(e)}")
            raise

    def generate_essay_original(self, context: str, prompt: str, max_length: int = MAX_LENGTH) -> str:
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
