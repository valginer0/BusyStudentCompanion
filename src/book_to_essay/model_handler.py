"""Handler for loading and managing the language model."""
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.quantization
import hashlib
import os
import pickle
from pathlib import Path
from src.book_to_essay.config import (
    MODEL_NAME, MAX_LENGTH, TEMPERATURE, MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS,
    MODEL_CACHE_DIR, QUANT_CONFIG
)
from typing import List, Dict

logger = logging.getLogger(__name__)

class DeepSeekHandler:
    """Handler for the language model."""
    
    def __init__(self):
        """Initialize the model and tokenizer with appropriate quantization."""
        try:
            logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
            
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
                **QUANT_CONFIG.get("load_config", {})
            )
            
            # Apply post-load quantization if specified
            if QUANT_CONFIG.get("post_load_quantize", False):
                logger.info("Applying post-load quantization...")
                config = QUANT_CONFIG.get("post_load_quantize", {})
                if config:
                    # Apply dynamic quantization for CPU
                    if QUANT_CONFIG.get("method") == "8bit_cpu":
                        logger.info("Applying dynamic quantization for CPU...")
                        self.model = torch.quantization.quantize_dynamic(
                            self.model, 
                            {torch.nn.Linear}, 
                            dtype=torch.qint8
                        )
            
            # Initialize text_chunks
            self.text_chunks = []
            
            # Initialize chunk cache
            self.chunk_cache_dir = Path(os.path.join(MODEL_CACHE_DIR, "chunk_cache"))
            self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Model loaded successfully")
            
            # Set model to evaluation mode for inference
            self.model.eval()
            # Disable gradient calculation for inference
            torch.set_grad_enabled(False)
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
    def process_text(self, text: str) -> None:
        """Process input text and store it for later use."""
        if not text:
            logger.warning("Empty text provided to process_text")
            return
            
        logger.info(f"Processing text with {len(text)} characters")
        
        # Clear existing chunks
        self.text_chunks = []
        
        # Split text into manageable chunks for processing
        if len(text) > MAX_CHUNK_SIZE * 2:  # Only chunk if text is significantly large
            logger.info(f"Text is large, splitting into chunks of max {MAX_CHUNK_SIZE} characters")
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed chunk size, store current chunk and start new one
                if len(current_chunk) + len(paragraph) > MAX_CHUNK_SIZE and current_chunk:
                    self.text_chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Add paragraph separator if not the first paragraph in chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if not empty
            if current_chunk:
                self.text_chunks.append(current_chunk)
                
            # If we have too many chunks, select representative ones
            if len(self.text_chunks) > MAX_CHUNKS_PER_ANALYSIS:
                logger.info(f"Too many chunks ({len(self.text_chunks)}), selecting {MAX_CHUNKS_PER_ANALYSIS} representative chunks")
                # Select chunks evenly distributed throughout the text
                step = len(self.text_chunks) / MAX_CHUNKS_PER_ANALYSIS
                selected_chunks = []
                for i in range(MAX_CHUNKS_PER_ANALYSIS):
                    idx = min(int(i * step), len(self.text_chunks) - 1)
                    selected_chunks.append(self.text_chunks[idx])
                self.text_chunks = selected_chunks
        else:
            # For smaller texts, just use the whole text as a single chunk
            self.text_chunks = [text]
            
        logger.info(f"Text processed into {len(self.text_chunks)} chunks")

    def _get_chunk_cache_key(self, chunk: str, topic: str, style: str, word_limit: int) -> str:
        """Generate a cache key for a chunk based on its content and generation parameters."""
        # Create a unique hash based on chunk content and generation parameters
        cache_key_data = f"{chunk}_{topic}_{style}_{word_limit}_{MODEL_NAME}"
        return hashlib.md5(cache_key_data.encode()).hexdigest()
    
    def _get_chunk_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a chunk analysis."""
        return self.chunk_cache_dir / f"{cache_key}.pkl"
    
    def _get_cached_chunk_analysis(self, chunk: str, topic: str, style: str, word_limit: int) -> str:
        """Get cached chunk analysis if it exists."""
        try:
            cache_key = self._get_chunk_cache_key(chunk, topic, style, word_limit)
            cache_path = self._get_chunk_cache_path(cache_key)
            
            if cache_path.exists():
                logger.info(f"Using cached chunk analysis for key: {cache_key[:8]}...")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error accessing chunk cache: {str(e)}")
        return None
    
    def _cache_chunk_analysis(self, chunk: str, topic: str, style: str, word_limit: int, analysis: str):
        """Cache chunk analysis for future use."""
        try:
            cache_key = self._get_chunk_cache_key(chunk, topic, style, word_limit)
            cache_path = self._get_chunk_cache_path(cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis, f)
            logger.info(f"Cached chunk analysis with key: {cache_key[:8]}...")
        except Exception as e:
            logger.error(f"Error caching chunk analysis: {str(e)}")

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
            
            # Check if we have a cached analysis for this chunk
            cached_analysis = self._get_cached_chunk_analysis(chunk, topic, style, word_limit)
            if cached_analysis:
                chunk_analyses.append(cached_analysis)
                logger.info(f"Using cached analysis for chunk {i+1}")
                continue
            
            # Include source information and MLA requirements in the chunk prompt
            source_info = ""
            if sources:
                source_info = "Source Materials:\n"
                for source in sources:
                    source_info += f"- {source['name']} ({source['type']})\n"
            
            # Improved chunk analysis prompt for Mistral model
            prompt = f"""<s>[INST] You are a literary analysis expert. Analyze the following excerpt from a text regarding {topic}.

INSTRUCTIONS:
1. Extract key quotes, themes, and evidence related to {topic}
2. Identify character actions and dialogue that illustrate {topic}
3. Note literary devices used to convey {topic}
4. Focus ONLY on content relevant to {topic}
5. Format your response as a concise analysis

{source_info}

TEXT EXCERPT:
{chunk}

YOUR ANALYSIS: [/INST]"""
            
            try:
                logger.info(f"Processing chunk {i+1} with {len(chunk)} characters")
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                
                # Move inputs to the same device as the model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                response = self.model.generate(
                    **inputs,
                    max_new_tokens=min(500, word_limit),  
                    temperature=0.7,
                    do_sample=True
                )
                chunk_analysis = self.tokenizer.decode(response[0], skip_special_tokens=True)
                
                # Extract only the model's response
                if "[/INST]" in chunk_analysis:
                    chunk_analysis = chunk_analysis.split("[/INST]")[1].strip()
                
                # Cache the chunk analysis for future use
                self._cache_chunk_analysis(chunk, topic, style, word_limit, chunk_analysis)
                
                chunk_analyses.append(chunk_analysis)
                logger.info(f"Successfully processed chunk {i+1}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Combine chunk analyses into final essay with MLA formatting
        citations_text = ""
        if mla_citations:
            citations_text = "Works Cited:\n" + "\n".join(mla_citations)
        
        # Improved final essay generation prompt for Mistral model
        combined_prompt = f"""<s>[INST] You are a skilled academic writer creating a {style} essay about {topic} in the text.

ESSAY REQUIREMENTS:
1. Write approximately {word_limit} words
2. Use {style} writing style
3. Begin with a clear thesis statement about {topic}
4. Include relevant quotes with MLA in-text citations (Author Page)
5. Make the essay thesis-driven, NOT a plot summary
6. End with a Works Cited section

ANALYSIS FROM TEXT:
{' '.join(chunk_analyses)}

WRITE THE COMPLETE ESSAY: [/INST]"""
        
        try:
            logger.info("Generating final essay with model...")
            # Generate the essay
            inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = self.model.generate(
                **inputs,
                max_new_tokens=min(1500, word_limit * 3),  # Allow for longer essays
                temperature=0.7,
                do_sample=True
            )
            
            # Get the generated text
            essay = self.tokenizer.decode(response[0], skip_special_tokens=True)
            logger.info(f"Raw essay generated with {len(essay)} characters")
            
            # Extract only the essay part (after the prompt)
            if "[/INST]" in essay:
                essay = essay.split("[/INST]")[1].strip()
            
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
