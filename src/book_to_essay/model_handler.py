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
            
            # Improved chunk analysis prompt for Mistral model with clearer separation
            prompt = f"""<s>[INST] You are a literary scholar analyzing literature. Your task is to extract and analyze material from this text excerpt that relates to '{topic}'. 

TEXT TO ANALYZE:
{chunk}

YOUR TASK:
- Identify key quotes that illustrate '{topic}'
- Note significant themes, motifs, and literary devices related to '{topic}'
- Analyze character development and dialogue that relates to '{topic}'
- Extract evidence for literary analysis about '{topic}'

IMPORTANT: Your response must ONLY contain analysis content. DO NOT:
- Repeat these instructions
- Include phrases like "here is my analysis" or "as requested"
- Include section headers like "Analysis:" or "Key Quotes:"
- Refer to yourself, the reader, or the task itself
- Mention social media, data analysis, AI, or homework

Start directly with substantive analysis. [/INST]"""

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
                    "social media", "data analysis",
                    "YOUR ANALYSIS", "do not repeat these instructions", "do not include these instructions"
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
        
        # Improved final essay generation prompt for Mistral model with clearer separation
        analysis_text = ' '.join(chunk_analyses)
        prompt = f"""<s>[INST] You are writing an essay as a literary scholar. Write a well-structured, {style} essay analyzing '{topic}' based on the provided literary analysis.

CONTENT TO USE:
{analysis_text}

ESSAY SPECIFICATIONS:
- Length: Approximately {word_limit} words
- Format: MLA style with proper citations
- Style: {style.capitalize()}
- Focus: Analysis of '{topic}' with textual evidence
- Structure: Introduction with thesis, body paragraphs with evidence, and conclusion

ESSAY STRUCTURE:
- Introduction: Begin with context about the work and present a clear thesis statement about '{topic}'
- Body: Develop 2-3 main points with textual evidence and analysis
- Conclusion: Synthesize your analysis and explain the significance of '{topic}'

IMPORTANT GUIDELINES:
- AVOID plot summary - focus on analysis
- INCLUDE specific textual evidence and quotes
- USE MLA in-text citations when quoting (Author Page)
- DO NOT discuss social media, data analysis, AI, or homework
- DO NOT include any meta-text about writing an essay
- DO NOT repeat these instructions or include explanatory text

START YOUR ESSAY DIRECTLY with the introduction paragraph. [/INST]"""

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
            
            # Completely revised filtering approach
            try:
                # First, try to find a proper essay beginning
                essay_start_patterns = [
                    r'(?:In|The|This|Shakespeare|Romeo|Juliet|Love|Tragedy|Throughout|When|Many|One|[A-Z][a-z]+\'s).*?\.',  # Sentences starting with common words
                    r'[A-Z][a-z]+\'s.*?\.',  # Possessive proper noun followed by text
                    r'[A-Z][a-z]+ [a-z]+ [a-z]+ [a-z]+.*?\.',  # Capitalized word followed by lowercase words
                    r'(?:[A-Z][a-z]+ ){2,}.*?\.',  # Multiple capitalized words followed by text
                    r'"[^"]+".*?\.',  # Quote followed by text
                ]
                
                essay_start = None
                for pattern in essay_start_patterns:
                    match = re.search(pattern, essay, re.IGNORECASE)
                    if match:
                        essay_start = match.start()
                        break
                
                if essay_start is not None:
                    essay = essay[essay_start:]
                    logger.info(f"Found essay start at position {essay_start}")
                else:
                    # If we can't find a proper start, use aggressive filtering
                    logger.warning("Could not find proper essay start, using aggressive filtering")
                    
                    # Split into lines and aggressively filter
                    lines = essay.split('\n')
                    filtered_lines = []
                    
                    # Skip lines that match these patterns
                    skip_patterns = [
                        r'^\s*\d+\.',  # Numbered items
                        r'^\s*-\s+',  # Bullet points
                        r'^\s*REQUIREMENTS',
                        r'^\s*INSTRUCTIONS',
                        r'^\s*ANALYSIS',
                        r'^\s*Source Materials:',
                        r'^\s*TEXT EXCERPT:',
                        r'^\s*<s>',
                        r'^\s*</s>',
                        r'^\s*\[INST\]',
                        r'^\s*\[/INST\]',
                        r'^\s*Write a',
                        r'^\s*Use MLA',
                        r'^\s*Include',
                        r'^\s*Analyze',
                        r'^\s*Maintain',
                        r'^\s*DO NOT',
                        r'^\s*The Project Gutenberg',
                        r'^\s*This ebook',
                        r'social media',
                        r'data analysis',
                        r'artificial intelligence',
                        r'HOMEWORK',
                        r'most other parts of the world',
                        r'^\s*ESSAY',
                        r'start directly with',
                        r'do not repeat these instructions',
                        r'do not include these instructions',
                        r'^\s*CONTENT TO USE',
                        r'^\s*ESSAY SPECIFICATIONS',
                        r'^\s*ESSAY STRUCTURE',
                        r'^\s*IMPORTANT GUIDELINES',
                        r'^\s*START YOUR ESSAY',
                        r'my analysis',
                        r'as requested',
                        r'as you requested',
                        r'in this essay',
                        r'in this analysis',
                        r'First I will',
                        r'I have analyzed',
                        r'I have written',
                        r'my response',
                        r'the instructions',
                        r'guidelines',
                        r'Here is',
                        r'based on your request',
                        r'following your',
                        r'as per your',
                        r'as instructed',
                        r'following the',
                        
                        # Enhanced patterns for better filtering
                        r'.*\d+[- ]word.*',
                        r'.*analytical essay.*',
                        r'.*academic tone.*',
                        r'.*proper structure.*',
                        r'.*proper citations.*',
                        r'.*textual evidence.*',
                        r'.*literary devices.*',
                        r'.*plot summary.*',
                        r'.*clear thesis.*',
                        r'.*complete, well-structured.*',
                        r'.*Works Cited:.*',
                        r'.*\[HOMEWORK\].*',
                        r'^\s*Write an essay.*',
                        r'^\s*Create an essay.*',
                        r'^\s*Craft an essay.*',
                        r'^\s*Develop an essay.*',
                        r'^\s*Compose an essay.*'
                    ]
                    
                    in_skip_section = False
                    for line in lines:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Check if we're entering a section to skip
                        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                            in_skip_section = True
                            continue
                            
                        # End skip section on blank line
                        if in_skip_section and not line.strip():
                            in_skip_section = False
                            continue
                            
                        # Skip lines in skip sections
                        if in_skip_section:
                            continue
                            
                        # Keep lines that start with capital letters and have reasonable length
                        if re.match(r'^[A-Z"\']', line.strip(), re.IGNORECASE) and len(line.strip()) > 20:
                            filtered_lines.append(line)
                    
                    essay = '\n'.join(filtered_lines)
                    
                    # Additional post-processing to remove any remaining instruction text patterns
                    if essay:
                        # Remove specific patterns that commonly appear at the beginning of essays
                        for pattern in [
                            r'^Write a.*essay.*\n',
                            r'^.*\d+[- ]word.*\n',
                            r'^.*MLA format.*\n',
                            r'^.*textual evidence.*\n',
                            r'^.*avoid plot summary.*\n',
                            r'^.*academic tone.*\n',
                            r'^\d+\. .*\n',  # Numbered instructions
                            r'^\[.*\].*\n',  # Content in brackets
                            r'^most other parts.*\n',
                            r'^Works Cited:.*\n'
                        ]:
                            essay = re.sub(pattern, '', essay, flags=re.MULTILINE | re.IGNORECASE)
                        
                        # Remove multiple sequential blank lines that might result from filtering
                        essay = re.sub(r'\n{3,}', '\n\n', essay)
                
                # Try a different approach if the essay is still problematic
                if not essay or len(essay.strip()) < 100:
                    logger.warning("Essay too short after filtering, trying sentence extraction")
                    # Extract all proper sentences from the original text
                    sentences = re.findall(r'[A-Z][^.!?]*[.!?]', essay, re.IGNORECASE)
                    if sentences and len(sentences) >= 3:
                        essay = ' '.join(sentences)
                    else:
                        # Try to find a paragraph that looks like an essay
                        paragraphs = re.split(r'\n\s*\n', essay)
                        valid_paragraphs = []
                        for para in paragraphs:
                            # Check if paragraph looks like proper essay content
                            if len(para.strip()) > 100 and not any(re.search(pattern, para, re.IGNORECASE) for pattern in skip_patterns):
                                valid_paragraphs.append(para.strip())
                        
                        if valid_paragraphs:
                            essay = '\n\n'.join(valid_paragraphs)
                
                # Verify essay has proper structure
                if essay and len(essay.strip()) >= 100:
                    # Check if it has at least 3 paragraphs
                    paragraphs = re.split(r'\n\s*\n', essay)
                    if len(paragraphs) < 3:
                        # Find paragraph boundaries using sentences
                        sentences = re.findall(r'[A-Z][^.!?]*[.!?]', essay, re.IGNORECASE)
                        if len(sentences) >= 9:  # Minimum 9 sentences for 3 paragraphs
                            # Group into paragraphs of 3-5 sentences each
                            reconstructed_paragraphs = []
                            current_para = []
                            
                            for i, sentence in enumerate(sentences):
                                current_para.append(sentence)
                                
                                # Start a new paragraph every 3-5 sentences
                                if len(current_para) >= 3 and (len(current_para) >= 5 or i % 4 == 3):
                                    reconstructed_paragraphs.append(' '.join(current_para))
                                    current_para = []
                            
                            # Add any remaining sentences as the last paragraph
                            if current_para:
                                reconstructed_paragraphs.append(' '.join(current_para))
                            
                            if len(reconstructed_paragraphs) >= 3:
                                essay = '\n\n'.join(reconstructed_paragraphs)
                                logger.info("Reconstructed essay into structured paragraphs")
                
                logger.info(f"After comprehensive filtering, essay starts with: {essay[:100] if essay else 'EMPTY'}...")
                
                # If essay is still empty or too short, create a fallback essay based on topic
                if not essay or len(essay.strip()) < 100:
                    logger.warning("Essay too short or empty after all filtering, using fallback")
                    essay = self._generate_fallback_essay(topic, style, word_limit)
            
            except Exception as e:
                logger.error(f"Error during essay filtering: {str(e)}")
                # Provide a fallback essay if filtering fails
                essay = self._generate_fallback_essay(topic, style, word_limit)
            
            # Add Works Cited if not already included
            if "Works Cited" not in essay and mla_citations:
                essay += f"\n\n{citations_text}"
                
            return essay
        except Exception as e:
            logger.error(f"Error generating final essay: {str(e)}")
            raise

    def _generate_fallback_essay(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a fallback essay when the main generation process fails.
        
        Args:
            topic: The essay topic
            style: The writing style
            word_limit: Target word count
            
        Returns:
            A basic essay on the topic
        """
        logger.info(f"Generating fallback essay on topic: {topic}")
        
        # Multi-stage fallback approach
        # Try several prompts with increasing specificity
        
        # Stage 1: Simple, direct prompt with role
        try:
            logger.info("Trying fallback stage 1: Simple prompt with role")
            prompt = f"""<s>[INST] As an English literature professor, write a {word_limit}-word {style} essay analyzing {topic}. Follow MLA format with in-text citations. Begin directly with your essay. [/INST]"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = self.model.generate(
                **inputs,
                max_new_tokens=min(1000, word_limit * 2),
                temperature=0.6,  # Lower temperature for more predictable output
                do_sample=True
            )
            
            fallback_essay = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Extract only the essay part (after the prompt)
            if "[/INST]" in fallback_essay:
                fallback_essay = fallback_essay.split("[/INST]")[1].strip()
            
            # Basic filtering
            lines = fallback_essay.split('\n')
            filtered_lines = []
            for line in lines:
                if line.strip() and not line.startswith('[') and not line.startswith('<') and not line.startswith('As an'):
                    filtered_lines.append(line)
            
            fallback_essay = '\n'.join(filtered_lines)
            
            # Check if it's a proper essay (at least 100 chars and starts with a capital letter)
            if fallback_essay and len(fallback_essay.strip()) >= 100 and re.match(r'^[A-Z"]', fallback_essay.strip()):
                return fallback_essay
                
        except Exception as e:
            logger.error(f"Error in fallback stage 1: {str(e)}")
        
        # Stage 2: More structured prompt with essay structure guidance
        try:
            logger.info("Trying fallback stage 2: Structured prompt with essay guidance")
            prompt = f"""<s>[INST] Write a clear, focused {style} essay on {topic}. 

Your essay must include:
1. Introduction with thesis statement
2. Body paragraphs with evidence
3. Conclusion

Begin your essay immediately. [/INST]"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = self.model.generate(
                **inputs,
                max_new_tokens=min(1000, word_limit * 2),
                temperature=0.5,  # Even lower temperature
                do_sample=True
            )
            
            fallback_essay = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Extract after instruction
            if "[/INST]" in fallback_essay:
                fallback_essay = fallback_essay.split("[/INST]")[1].strip()
            
            # More aggressive filtering
            lines = fallback_essay.split('\n')
            filtered_lines = []
            skip_patterns = [r'Your essay', r'Introduction', r'Body', r'Conclusion', r'must include', r'Begin your']
            
            for line in lines:
                if line.strip() and not any(re.search(pattern, line) for pattern in skip_patterns):
                    filtered_lines.append(line)
            
            fallback_essay = '\n'.join(filtered_lines)
            
            # Check if it's a proper essay
            if fallback_essay and len(fallback_essay.strip()) >= 100:
                return fallback_essay
                
        except Exception as e:
            logger.error(f"Error in fallback stage 2: {str(e)}")
        
        # Stage 3: Template-based essay with topic-specific content
        logger.info("Using template-based fallback essay for topic: " + topic)
        
        # Check topic category to determine template style
        character_related = any(word in topic.lower() for word in ["character", "protagonist", "antagonist", "hero", "villain"])
        theme_related = any(word in topic.lower() for word in ["theme", "motif", "symbolism", "imagery", "metaphor", "allegory"])
        literary_device = any(word in topic.lower() for word in ["irony", "foreshadowing", "allusion", "personification", "tone", "mood"])
        
        # Select appropriate template based on topic type
        if character_related:
            return self._generate_character_essay_template(topic, style, word_limit)
        elif theme_related:
            return self._generate_theme_essay_template(topic, style, word_limit)
        elif literary_device:
            return self._generate_literary_device_essay_template(topic, style, word_limit)
        else:
            # General template
            return f"""The analysis of {topic} reveals significant insights into the literary work. In examining this subject, several patterns emerge that warrant careful consideration and analysis. The author's treatment of {topic} serves multiple purposes within the narrative structure.

First, {topic} functions as a central element that shapes character development throughout the work. The ways in which characters interact with and respond to {topic} reveals their motivations, values, and internal conflicts. This character-based analysis provides readers with a deeper understanding of the psychological dimensions at play.

Second, {topic} operates as a thematic device that connects to broader ideas within the text. By examining how {topic} relates to the work's major themes, we can see the author's commentary on larger social, philosophical, or ethical questions. The textual evidence supports an interpretation that {topic} serves as both a literal element and a symbolic representation of these deeper concerns.

Finally, the stylistic and structural choices surrounding {topic} demonstrate sophisticated literary craftsmanship. The author's use of language, imagery, and narrative structure in relation to {topic} enhances its significance and impact on the reader's experience. This technical analysis reveals the deliberate artistic choices that elevate the work beyond mere storytelling.

Through careful examination of textual evidence, it becomes clear that {topic} functions as an essential component of the work's literary merit and meaning. This analysis demonstrates how a focused study of specific elements can illuminate our understanding of literature as both art and cultural expression."""

    def _generate_character_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a character-focused essay template."""
        return f"""The character development related to {topic} represents a masterful example of literary craftsmanship. Through careful examination of the text, we can identify how the author constructs this character study to convey deeper thematic meaning and psychological insight.

In the early portions of the narrative, {topic} is established through specific character actions and dialogue that reveal fundamental traits and motivations. The author's initial characterization creates a foundation upon which more complex development can build. These early scenes are crucial for understanding the character's journey and the narrative's overall trajectory.

As the work progresses, complications arise that challenge and deepen our understanding of {topic}. The character's responses to conflict reveal layers of complexity that move beyond simplistic interpretations. Notable scenes demonstrating this include moments of internal conflict and decisive action that show character growth or revealing contradictions.

The relationship between {topic} and other characters provides additional insight into the author's thematic concerns. These interpersonal dynamics highlight questions of identity, morality, and human connection that extend beyond the individual character study. Through these relationships, the author explores broader questions about human nature and social structures.

By the work's conclusion, the development of {topic} has reached a resolution that reflects the author's overall literary vision. Whether through transformation, tragic realization, or confirmed identity, the character's journey illustrates core themes of the work. This resolution demonstrates how character development serves as a vehicle for the author's broader artistic and philosophical aims.

This analysis of {topic} demonstrates how character study provides a lens through which to understand the work's literary merit and thematic depth. By examining specific textual elements related to characterization, we gain insight into both the technical craftsmanship and meaningful content of the literature."""

    def _generate_theme_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a theme-focused essay template."""
        return f"""The thematic exploration of {topic} throughout the literary work reveals the author's artistic vision and philosophical concerns. By analyzing how this theme develops and functions, we gain insight into both the work's literary craftsmanship and its broader cultural significance.

The introduction of {topic} as a thematic element begins subtly in the early portions of the work. Through carefully selected imagery, dialogue, and narrative focus, the author establishes this theme in ways that prepare readers for its fuller development. These initial instances may seem minor but provide essential foundation for the theme's evolution.

As the narrative progresses, {topic} emerges more prominently through key scenes that directly engage with this thematic concern. The author employs literary techniques such as symbolism, contrast, and parallel structures to emphasize the theme's importance. Textual evidence demonstrates how characters' experiences and choices illuminate different facets of {topic}, creating a multidimensional thematic exploration.

The author's treatment of {topic} connects to broader literary traditions and cultural contexts. By placing this thematic exploration within its historical and literary framework, we can better understand its significance and innovation. The work both draws upon established thematic patterns and contributes new perspectives to ongoing artistic and intellectual dialogues.

The resolution of {topic} as a thematic element provides insight into the author's ultimate vision. Whether through reconciliation, tragic realization, or ambiguous conclusion, the final treatment of this theme reflects the work's philosophical stance. This thematic resolution demonstrates how literature can engage with complex ideas while maintaining artistic integrity.

This analysis of {topic} as a central theme illustrates how literary works construct meaning through patterns of imagery, characterization, and narrative structure. By examining the specific textual elements that develop this theme, we appreciate both the technical sophistication and intellectual depth of the work."""

    def _generate_literary_device_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a literary device-focused essay template."""
        return f"""The author's use of {topic} as a literary device demonstrates sophisticated artistic technique and contributes significantly to the work's meaning. Through careful analysis of how this device functions within the text, we can better understand both the formal craftsmanship and thematic purpose of the literature.

The implementation of {topic} appears strategically throughout the narrative, creating patterns that reward close reading and analysis. The author deploys this literary technique with varying intensity and purpose, demonstrating masterful control of the narrative craft. Early instances establish the device's presence, while later occurrences build upon this foundation with increasing complexity.

Specific examples of {topic} within the text reveal its multifaceted purpose. In several key passages, this literary device serves to illuminate character psychology, advance plot development, and reinforce thematic concerns simultaneously. The author's technical skill is evident in how seamlessly {topic} integrates into the narrative structure rather than appearing as a forced or artificial element.

The relationship between {topic} and other literary elements creates a cohesive artistic vision. This device does not function in isolation but works in concert with characterization, setting, dialogue, and other narrative components. This integration demonstrates the author's holistic approach to literary creation, where technical devices serve the work's broader artistic aims.

The significance of {topic} extends beyond technical achievement to influence the reader's experience and interpretation. This literary device shapes how readers engage with the text, directing attention, creating emotional responses, and guiding intellectual understanding. The effect on readers reveals how formal elements actively construct meaning rather than merely decorating content.

This analysis of {topic} as a literary device demonstrates the inseparability of form and content in literature. By examining how technical aspects of writing contribute to meaning, we develop a deeper appreciation for literature as a carefully constructed art form that communicates through both what it says and how it is said."""

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
