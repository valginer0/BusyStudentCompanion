"""Handler for loading and managing the DeepSeek model."""
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.quantization
from src.book_to_essay.config import (
    MODEL_NAME, MAX_LENGTH, TEMPERATURE,
    MODEL_CACHE_DIR, QUANT_CONFIG
)
from typing import List, Dict

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
        if not text:
            logger.warning("Empty text provided to process_text")
            return
            
        logger.info(f"Processing text with {len(text)} characters")
        
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

    def generate_essay(self, topic: str, word_limit: int = 500, style: str = "academic", sources: List[Dict] = None) -> str:
        """Generate an essay on the given topic using the loaded text with MLA formatting.
        
        Args:
            topic: The essay topic or prompt
            word_limit: Maximum word count for the essay
            style: Writing style (academic, analytical, argumentative, expository)
            sources: List of source information dictionaries for citations
            
        Returns:
            A formatted essay with MLA citations
        """
        if not hasattr(self, 'text_chunks') or not self.text_chunks:
            raise ValueError("No text has been loaded. Please load text first.")

        # Ensure word_limit is an integer
        word_limit = int(word_limit)
        logger.info(f"Generating essay on topic: {topic} with word limit: {word_limit}, style: {style}")
        
        # Prepare MLA citations if sources are provided
        mla_citations = []
        if sources:
            for source in sources:
                # Extract author from filename (assuming format: Author - Title.ext)
                author = source['name'].split(' - ')[0] if ' - ' in source['name'] else "Unknown Author"
                title = source['name'].split(' - ')[1].rsplit('.', 1)[0] if ' - ' in source['name'] else source['name']
                
                # Format MLA citation based on source type
                if source['type'] == 'pdf':
                    citation = f"{author}. \"{title}.\" PDF file."
                elif source['type'] == 'epub':
                    citation = f"{author}. \"{title}.\" E-book."
                else:  # txt and others
                    citation = f"{author}. \"{title}.\""
                
                mla_citations.append(citation)
        
        # Process each chunk with source information
        chunk_analyses = []
        for i, chunk in enumerate(self.text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(self.text_chunks)}")
            
            # Include source information and MLA requirements in the chunk prompt
            source_info = ""
            if sources:
                source_info = "Source Materials:\n"
                for source in sources:
                    source_info += f"- {source['name']} ({source['type']})\n"
            
            # Improved chunk analysis prompt
            prompt = f"""Analyze this excerpt from the text regarding the theme of "{topic}".
            Extract relevant information, quotes, and evidence that could support an {style} essay.
            Focus only on content relevant to "{topic}".
            
            {source_info}
            Text excerpt: {chunk}"""
            
            try:
                logger.info(f"Processing chunk {i+1} with {len(chunk)} characters")
                response = self.model.generate(
                    **self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096),
                    max_new_tokens=min(500, word_limit),  
                    temperature=0.7,
                    do_sample=True
                )
                chunk_analysis = self.tokenizer.decode(response[0], skip_special_tokens=True)
                chunk_analyses.append(chunk_analysis)
                logger.info(f"Successfully processed chunk {i+1}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Combine chunk analyses into final essay with MLA formatting
        citations_text = ""
        if mla_citations:
            citations_text = "Works Cited:\n" + "\n".join(mla_citations)
        
        # Improved final essay generation prompt
        combined_prompt = f"""Write a {style} essay analyzing the theme of {topic} in the text.

Requirements:
1. The essay should be approximately {word_limit} words
2. Use {style} writing style
3. Include relevant quotes from the source materials
4. Use MLA in-text citations (Author Page) for each quote
5. The essay should be thesis-driven, not a plot summary
6. End with a Works Cited section

Analysis from text excerpts:
{' '.join(chunk_analyses)}

Write ONLY the essay content below. Do not include any instructions, explanations, or meta-commentary:
"""
        
        try:
            logger.info("Generating final essay with model...")
            # Generate the essay
            response = self.model.generate(
                **self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=4096),
                max_new_tokens=min(1000, word_limit * 2),
                temperature=0.7,
                do_sample=True
            )
            
            # Get the generated text
            essay = self.tokenizer.decode(response[0], skip_special_tokens=True)
            logger.info(f"Raw essay generated with {len(essay)} characters")
            
            # Extract only the essay part (after the prompt)
            if "Write ONLY the essay content below. Do not include any instructions, explanations, or meta-commentary:" in essay:
                essay = essay.split("Write ONLY the essay content below. Do not include any instructions, explanations, or meta-commentary:")[1].strip()
            
            # Additional extraction logic to handle other potential formats
            # Remove any lines that look like instructions
            lines = essay.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines that look like instructions
                if line.strip().startswith("Write a paper on") or "should be" in line or "must use" in line:
                    continue
                filtered_lines.append(line)
            
            essay = '\n'.join(filtered_lines)
            
            # Add Works Cited if not already included
            if "Works Cited" not in essay and mla_citations:
                essay += f"\n\n{citations_text}"
                
            return essay
        except Exception as e:
            logger.error(f"Error generating final essay: {str(e)}")
            raise

    def generate_essay_original(self, context: str, prompt: str, max_length: int = MAX_LENGTH) -> str:
        """Generate an essay using the DeepSeek model.
        
        Warning: This method is deprecated and will be removed in a future version.
        Use generate_essay() instead, which supports chunking for large texts.
        """
        import warnings
        warnings.warn(
            "generate_essay_original() is deprecated and will be removed in a future version. "
            "Use generate_essay() instead, which supports chunking for large texts.",
            DeprecationWarning, 
            stacklevel=2
        )
        
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
