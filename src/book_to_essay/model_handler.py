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
import re

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
            prompt = f"""<s>[INST] You are a literary analysis expert. Analyze the following excerpt from a text regarding '{topic}' (NOT about social media or data analysis).

INSTRUCTIONS:
1. Extract key quotes, themes, and evidence related to '{topic}'
2. Identify character actions and dialogue that illustrate '{topic}'
3. Note literary devices used to convey '{topic}'
4. Focus ONLY on content relevant to '{topic}'
5. Format your response as a concise analysis

{source_info}

TEXT EXCERPT:
{chunk}

YOUR ANALYSIS: [/INST]"""

            # Print the prompt for debugging
            print(f"\nDEBUG - CHUNK ANALYSIS PROMPT:\n{prompt[:500]}...\n")
            
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
                
                # Debug logging for chunk analysis
                logger.info(f"Raw chunk analysis starts with: {chunk_analysis[:100]}...")
                logger.info(f"Contains [/INST]? {'[/INST]' in chunk_analysis}")
                
                # Extract only the model's response
                if "[/INST]" in chunk_analysis:
                    chunk_analysis = chunk_analysis.split("[/INST]")[1].strip()
                    logger.info(f"After [/INST] extraction, chunk analysis starts with: {chunk_analysis[:100]}...")
                else:
                    logger.warning("Could not find [/INST] marker in the chunk analysis")
                
                # Filter out instruction text and social media references from chunk analysis
                instruction_keywords = [
                    "INSTRUCTIONS:", "Extract key", "Identify character", "Note literary", 
                    "Focus ONLY", "Format your", "Source Materials:", "TEXT EXCERPT:",
                    "social media", "data points", "trends, statistics", "percentages"
                ]
                
                # Split into lines and filter out instruction lines
                chunk_lines = chunk_analysis.split('\n')
                filtered_chunk_lines = []
                
                for line in chunk_lines:
                    should_skip = False
                    for keyword in instruction_keywords:
                        if keyword.lower() in line.lower():
                            should_skip = True
                            break
                    
                    # Skip numbered list items that look like instructions
                    if re.match(r'^\d+\.\s+(Extract|Identify|Note|Focus|Format)', line.strip()):
                        should_skip = True
                    
                    if not should_skip:
                        filtered_chunk_lines.append(line)
                
                chunk_analysis = '\n'.join(filtered_chunk_lines)
                logger.info(f"After filtering, chunk analysis starts with: {chunk_analysis[:100]}...")
                
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
        
        # Create source info for the final essay prompt
        source_info = ""
        if sources:
            source_info = "Source Materials:\n"
            for source in sources:
                source_info += f"- {source['name']} ({source['type']})\n"
        
        # Improved final essay generation prompt for Mistral model
        analysis_text = ' '.join(chunk_analyses)
        prompt = f"""<s>[INST] You are a professional essay writer. Write a {style} essay on {topic} using the following analysis of a text.

REQUIREMENTS:
1. Write a {word_limit}-word {style} essay with a clear thesis
2. Use MLA format with proper citations
3. Include textual evidence and quotes from the source
4. Analyze themes and literary devices, avoid plot summary
5. Maintain academic tone and proper structure

{source_info}

ANALYSIS NOTES:
{analysis_text}

Write a complete, well-structured essay: [/INST]"""

        # Print the prompt for debugging
        print(f"\nDEBUG - FINAL ESSAY PROMPT:\n{prompt[:500]}...\n")
        
        try:
            logger.info("Generating final essay with model...")
            # Generate the essay
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
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
            
            # Debug logging
            logger.info(f"Raw essay starts with: {essay[:100]}...")
            logger.info(f"Contains [/INST]? {'[/INST]' in essay}")
            
            # Extract only the essay part (after the prompt)
            if "[/INST]" in essay:
                essay = essay.split("[/INST]")[1].strip()
                logger.info(f"After [/INST] extraction, essay starts with: {essay[:100]}...")
            else:
                logger.warning("Could not find [/INST] marker in the generated text")
            
            # Comprehensive filtering to remove all instruction text and prompt repetition
            essay_lines = essay.split('\n')
            filtered_essay_lines = []
            
            # Patterns to identify and remove
            instruction_patterns = [
                r'^\[\d+\]', # Numbered requirements like [1], [2]
                r'^REQUIREMENTS:', 
                r'^INSTRUCTIONS:',
                r'^Source Materials:',
                r'^ANALYSIS NOTES:',
                r'^TEXT EXCERPT:',
                r'^Write a \d+-word',
                r'^Use MLA format',
                r'^Include textual evidence',
                r'^Analyze themes',
                r'^Maintain academic tone',
                r'<s>\[INST\]',
                r'</s>',
                r'^The Project Gutenberg eBook',
                r'^This ebook is for the use',
                r'^\s*-\s*Shakespeare',
                r'^\s*-\s*[^-]+\.txt \(txt\)',
                r'^ANALYSIS FROM TEXT:',
                r'^WRITE THE COMPLETE ESSAY:',
                r'^HOMEWORK',
                r'social media',
                r'data points',
                r'trends, statistics',
                r'percentages'
            ]
            
            # Skip sections between certain markers
            skip_until_empty_line = False
            in_instruction_block = False
            
            for line in essay_lines:
                # Skip empty lines at the beginning
                if not filtered_essay_lines and not line.strip():
                    continue
                
                # Check if we should skip until an empty line
                if skip_until_empty_line:
                    if not line.strip():
                        skip_until_empty_line = False
                    continue
                
                # Check for section markers that indicate we should skip until empty line
                if "Source Materials:" in line or "TEXT EXCERPT:" in line:
                    skip_until_empty_line = True
                    continue
                
                # Check for instruction block start
                if line.strip().startswith("[1]") or line.strip().startswith("REQUIREMENTS:") or line.strip().startswith("INSTRUCTIONS:"):
                    in_instruction_block = True
                    continue
                
                # End of instruction block
                if in_instruction_block and not line.strip():
                    in_instruction_block = False
                    continue
                
                # Skip lines in instruction block
                if in_instruction_block:
                    continue
                
                # Check against all patterns
                should_skip = False
                for pattern in instruction_patterns:
                    if re.search(pattern, line):
                        should_skip = True
                        break
                
                if not should_skip:
                    filtered_essay_lines.append(line)
            
            # Join the filtered lines
            essay = '\n'.join(filtered_essay_lines)
            
            # Remove any remaining instruction text at the beginning
            essay = re.sub(r'^.*?(?=\w{3,}.*?[.!?])', '', essay, flags=re.DOTALL)
            
            # Log the filtered essay
            logger.info(f"After comprehensive filtering, essay starts with: {essay[:100]}...")
            
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
