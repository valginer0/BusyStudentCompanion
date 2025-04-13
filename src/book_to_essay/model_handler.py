"""Handler for loading and managing the language model."""
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.quantization
import re
import os
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .config import (
    MODEL_NAME, MODEL_CACHE_DIR, MAX_LENGTH, TEMPERATURE, 
    MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS, QUANT_CONFIG
)
from .prompts.factory import PromptTemplateFactory
from .fallback import FallbackReason

logger = logging.getLogger(__name__)

class DeepSeekHandler:
    """Handler for the language model."""
    
    def __init__(self, model=None, tokenizer=None, prompt_template=None):
        """Initialize the model and tokenizer with appropriate quantization.
        
        Args:
            model: Optional pre-loaded model to reuse
            tokenizer: Optional pre-loaded tokenizer to reuse
            prompt_template: Optional pre-loaded prompt template to reuse
        """
        try:
            # Initialize chunk cache directory regardless of model reuse
            self.chunk_cache_dir = Path(os.path.join(MODEL_CACHE_DIR, "chunk_cache"))
            self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize text_chunks
            self.text_chunks = []
            
            # If model and tokenizer are provided, reuse them
            if model is not None and tokenizer is not None:
                logger.info("Reusing existing model and tokenizer")
                self.model = model
                self.tokenizer = tokenizer
                self.prompt_template = prompt_template or PromptTemplateFactory.create(MODEL_NAME)
                return
                
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
            
            # Initialize prompt template based on model
            if prompt_template is None:
                self.prompt_template = PromptTemplateFactory.create(MODEL_NAME)
            else:
                self.prompt_template = prompt_template
            logger.info(f"Using prompt template for model type: {self.prompt_template.__class__.__name__}")
            
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
        """Generate an essay using chunk-based analysis.
        
        Args:
            topic: Essay topic to write about
            word_limit: Target word count
            style: Writing style (academic, analytical, etc.)
            sources: Optional list of citation sources
            
        Returns:
            A generated essay
        """
        logger.info(f"Generating essay on topic: {topic} with {word_limit} word limit in {style} style")
        
        try:
            # Validate text chunks exist
            if not hasattr(self, 'text_chunks') or not self.text_chunks:
                logger.error("No text has been loaded. Please load text first.")
                raise RuntimeError("No text has been loaded. Please load text first.")
            
            # If source not provided, try to extract from book data
            if sources is None and hasattr(self, 'book_data'):
                sources = [self.book_data]
            
            # Prepare MLA citations
            mla_citations = None
            citations_text = ""
            if sources:
                mla_citations = []
                for source in sources:
                    if 'author' in source and 'title' in source:
                        mla_citation = f"{source['author']}. {source['title']}."
                        if 'publisher' in source:
                            mla_citation += f" {source['publisher']},"
                        if 'year' in source:
                            mla_citation += f" {source['year']}."
                        mla_citations.append(mla_citation)
                
                if mla_citations:
                    citations_text = "Works Cited\n\n" + "\n".join(mla_citations)
            
            # Process the chunks
            try:
                chunk_analyses = []
                
                # Log chunk processing start
                logger.info(f"Processing {len(self.text_chunks)} text chunks for analysis")
                
                if not self.text_chunks or all(not chunk.strip() for chunk in self.text_chunks):
                    logger.error("All text chunks are empty or whitespace")
                    raise RuntimeError("All text chunks are empty or whitespace")
                
                # Analyze each chunk
                for i, chunk in enumerate(self.text_chunks):
                    if not chunk.strip():
                        logger.warning(f"Skipping empty chunk {i}")
                        continue
                    
                    try:
                        logger.info(f"Analyzing chunk {i+1}/{len(self.text_chunks)} - length: {len(chunk)}")
                        # Print sample of chunk for debugging
                        logger.debug(f"Chunk {i+1} starts with: {chunk[:50]}...")
                        
                        # Analyze the chunk using the prompt template
                        analysis = self._analyze_chunk(chunk, topic, style, word_limit)
                        
                        # Add the analysis to our list if non-empty
                        if analysis and analysis.strip():
                            chunk_analyses.append(analysis)
                        else:
                            logger.warning(f"Chunk {i+1} analysis was empty")
                    except Exception as e:
                        logger.error(f"Error analyzing chunk {i+1}: {str(e)}")
                        # Continue with other chunks
                
                # If no analyses were produced, raise an error
                if not chunk_analyses:
                    logger.error("No chunk analyses were produced")
                    raise RuntimeError("No chunk analyses were produced")
                
                logger.info(f"Successfully analyzed {len(chunk_analyses)} chunks")
                
                # Generate the essay using the analyses
                try:
                    # Combine all analyses into one
                    combined_analysis = "\n\n".join(chunk_analyses)
                    
                    # Check if analysis exceeds token limit
                    approx_tokens = len(combined_analysis.split())
                    if approx_tokens > 4000:  # Conservative estimate
                        logger.warning(f"Analysis may exceed token limit ({approx_tokens} words)")
                        
                        # Truncate to avoid token limit issues
                        combined_analysis = self._truncate_text(combined_analysis, 3500)
                        logger.info(f"Truncated analysis to {len(combined_analysis.split())} words")
                    
                    # Format the prompt for essay generation
                    prompt = self.prompt_template.format_essay_prompt(
                        topic=topic,
                        style=style,
                        word_limit=word_limit,
                        analysis=combined_analysis,
                        citations=mla_citations
                    )
                    
                    # Do the generation
                    device = next(self.model.parameters()).device
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                    
                    logger.info("Generating full essay from analysis")
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=min(2048, word_limit * 3),  # Limit based on word count
                            temperature=TEMPERATURE,
                            do_sample=True
                        )
                    
                    # Decode the model output
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Try to extract just the essay part
                    essay = self.prompt_template.extract_response(response)
                    
                    logger.info(f"Generated essay with length: {len(essay)}")
                except Exception as e:
                    logger.error(f"Error generating essay from analyses: {str(e)}")
                    raise RuntimeError(f"Essay generation failed due to an internal error: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error during chunk analysis: {str(e)}")
                raise RuntimeError(f"Chunk analysis failed due to an internal error: {str(e)}")
            
            # Post-process the essay
            try:
                logger.info("Filtering and cleaning essay")
                
                # Define patterns to skip (instructions, etc.)
                skip_patterns = [
                    r"(?i)essay:", r"(?i)essay on", 
                    r"(?i)here'?s? (?:an|my|the) essay", r"(?i)I'?ll write an essay",
                    r"(?i)this essay will", r"(?i)in this essay",
                    r"(?i)instructions", r"(?i)prompt", r"(?i)guidelines",
                    r"(?i)understand(?:ing)? the", r"(?i)analyze the text",
                    r"(?i)let'?s analyze", r"(?i)let'?s begin",
                    r"(?i)I will analyze", r"(?i)I'?ll analyze",
                    r"(?i)start(?:ing)? (?:with|by)", r"(?i)start directly",
                    r"(?i)do not repeat", r"(?i)don'?t repeat",
                    r"(?i)Title:", r"(?i)MLA Format:",
                    r"(?i)ESSAY"
                ]
                
                # Try to find where the actual essay content begins
                essay_start_patterns = [
                    r'(?<=\n\n)[A-Z][^.!?]{10,}[.!?]',  # Paragraph starting after 2 newlines
                    r'^[A-Z][^.!?]{10,}[.!?]',  # Essay starting with a proper sentence at the beginning
                    r'(?<=\n)[A-Z][^.!?]{10,}[.!?]',  # Paragraph starting after 1 newline
                    r'[A-Z][^.!?]{20,}[.!?]'   # Any proper longer sentence
                ]
                
                for pattern in essay_start_patterns:
                    match = re.search(pattern, essay)
                    if match:
                        start_index = match.start()
                        essay = essay[start_index:]
                        logger.info(f"Found essay start using pattern: {pattern}")
                        break
                
                # Skip any header lines that don't look like essay content
                lines = essay.split('\n')
                start_line = 0
                for i, line in enumerate(lines):
                    if any(re.search(pattern, line) for pattern in skip_patterns):
                        logger.info(f"Skipping line {i}: {line}")
                        start_line = i + 1
                    elif len(line.strip()) > 30 and re.match(r'[A-Z]', line.strip()):
                        # Found a substantive line that starts with uppercase
                        break
                
                # Reconstruct the essay from the starting line
                if start_line > 0 and start_line < len(lines):
                    essay = '\n'.join(lines[start_line:])
                
                if not essay or len(essay.strip()) < 100:
                    logger.warning("Essay too short after filtering, raising error.")
                    raise RuntimeError(f"Essay generation failed due to insufficient content: {len(essay)} characters")
                
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
                
            except Exception as e:
                logger.error(f"Error during essay filtering: {str(e)}")
                raise RuntimeError(f"Essay filtering failed due to an internal error: {str(e)}")
            
            # Print the filtered essay for debugging
            print("\n" + "="*50 + " FILTERED ESSAY " + "="*50)
            print(essay)
            print("="*120)
            
            # Add Works Cited if not already included
            if "Works Cited" not in essay and mla_citations:
                essay += f"\n\n{citations_text}"
                
            return essay
        except Exception as e:
            logger.error(f"Error generating final essay: {str(e)}")
            raise

    def _generate_fallback_essay(self, topic: str, style: str, word_limit: int, reason: FallbackReason = FallbackReason.UNKNOWN) -> str:
        """Generate a fallback essay when the main generation process fails."""
        # THIS METHOD IS NOW EFFECTIVELY OBSOLETE DUE TO CHANGES IN generate_essay
        logger.warning(f"Fallback essay generation triggered for topic '{topic}' due to {reason.name}. NOTE: This function should no longer be called.")
        
        # Simple template generation based on topic keywords
        if "character" in topic.lower():
            return f"""The character development related to {topic} represents a masterful example of literary craftsmanship. Through careful examination of the text, we can identify how the author constructs this character study to convey deeper thematic meaning and psychological insight.

In the early portions of the narrative, {topic} is established through specific character actions and dialogue that reveal fundamental traits and motivations. The author's initial characterization creates a foundation upon which more complex development can build. These early scenes are crucial for understanding the character's journey and the narrative's overall trajectory.

As the work progresses, complications arise that challenge and deepen our understanding of {topic}. The character's responses to conflict reveal layers of complexity that move beyond simplistic interpretations. Notable scenes demonstrating this include moments of internal conflict and decisive action that show character growth or revealing contradictions.

The relationship between {topic} and other characters provides additional insight into the author's thematic concerns. These interpersonal dynamics highlight questions of identity, morality, and human connection that extend beyond the individual character study. Through these relationships, the author explores broader questions about human nature and social structures.

By the work's conclusion, the development of {topic} has reached a resolution that reflects the author's overall literary vision. Whether through transformation, tragic realization, or confirmed identity, the character's journey illustrates core themes of the work. This resolution demonstrates how character development serves as a vehicle for the author's broader artistic and philosophical aims.

This analysis of {topic} demonstrates how character study provides a lens through which to understand the work's literary merit and thematic depth. By examining specific textual elements related to characterization, we gain insight into both the technical craftsmanship and meaningful content of the literature."""

    def _generate_character_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a character-focused essay template."""
        logger.info(f"Creating character-focused template essay for topic: {topic}")
        return f"""The character development related to {topic} represents a masterful example of literary craftsmanship. Through careful examination of the text, we can identify how the author constructs this character study to convey deeper thematic meaning and psychological insight.

In the early portions of the narrative, {topic} is established through specific character actions and dialogue that reveal fundamental traits and motivations. The author's initial characterization creates a foundation upon which more complex development can build. These early scenes are crucial for understanding the character's journey and the narrative's overall trajectory.

As the work progresses, complications arise that challenge and deepen our understanding of {topic}. The character's responses to conflict reveal layers of complexity that move beyond simplistic interpretations. Notable scenes demonstrating this include moments of internal conflict and decisive action that show character growth or revealing contradictions.

The relationship between {topic} and other characters provides additional insight into the author's thematic concerns. These interpersonal dynamics highlight questions of identity, morality, and human connection that extend beyond the individual character study. Through these relationships, the author explores broader questions about human nature and social structures.

By the work's conclusion, the development of {topic} has reached a resolution that reflects the author's overall literary vision. Whether through transformation, tragic realization, or confirmed identity, the character's journey illustrates core themes of the work. This resolution demonstrates how character development serves as a vehicle for the author's broader artistic and philosophical aims.

This analysis of {topic} demonstrates how character study provides a lens through which to understand the work's literary merit and thematic depth. By examining specific textual elements related to characterization, we gain insight into both the technical craftsmanship and meaningful content of the literature."""

    def _generate_theme_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a theme-focused essay template."""
        logger.info(f"Creating theme-focused template essay for topic: {topic}")
        return f"""The thematic exploration of {topic} throughout the literary work reveals the author's artistic vision and philosophical concerns. By analyzing how this theme develops and functions, we gain insight into both the work's literary craftsmanship and its broader cultural significance.

The introduction of {topic} as a thematic element begins subtly in the early portions of the work. Through carefully selected imagery, dialogue, and narrative focus, the author establishes this theme in ways that prepare readers for its fuller development. These initial instances may seem minor but provide essential foundation for the theme's evolution.

As the narrative progresses, {topic} emerges more prominently through key scenes that directly engage with this thematic concern. The author employs literary techniques such as symbolism, contrast, and parallel structures to emphasize the theme's importance. Textual evidence demonstrates how characters' experiences and choices illuminate different facets of {topic}, creating a multidimensional thematic exploration.

The author's treatment of {topic} connects to broader literary traditions and cultural contexts. By placing this thematic exploration within its historical and literary framework, we can better understand its significance and innovation. The work both draws upon established thematic patterns and contributes new perspectives to ongoing artistic and intellectual dialogues.

The resolution of {topic} as a thematic element provides insight into the author's ultimate vision. Whether through reconciliation, tragic realization, or ambiguous conclusion, the final treatment of this theme reflects the work's philosophical stance. This thematic resolution demonstrates how literature can engage with complex ideas while maintaining artistic integrity.

This analysis of {topic} as a central theme illustrates how literary works construct meaning through patterns of imagery, characterization, and narrative structure. By examining the specific textual elements that develop this theme, we appreciate both the technical sophistication and intellectual depth of the work."""

    def _generate_literary_device_essay_template(self, topic: str, style: str, word_limit: int) -> str:
        """Generate a literary device-focused essay template."""
        logger.info(f"Creating literary device-focused template essay for topic: {topic}")
        return f"""The author's use of {topic} as a literary device demonstrates sophisticated artistic technique and contributes significantly to the work's meaning. Through careful analysis of how this device functions within the text, we can better understand both the formal craftsmanship and thematic purpose of the literature.

The implementation of {topic} appears strategically throughout the narrative, creating patterns that reward close reading and analysis. The author deploys this literary technique with varying intensity and purpose, demonstrating masterful control of the narrative craft. Early instances establish the device's presence, while later occurrences build upon this foundation with increasing complexity.

Specific examples of {topic} within the text reveal its multifaceted purpose. In several key passages, this literary device serves to illuminate character psychology, advance plot development, and reinforce thematic concerns simultaneously. The author's technical skill is evident in how seamlessly {topic} integrates into the narrative structure rather than appearing as a forced or artificial element.

The relationship between {topic} and other literary elements creates a cohesive artistic vision. This device does not function in isolation but works in concert with characterization, setting, dialogue, and other narrative components. This integration demonstrates the author's holistic approach to literary creation, where technical devices serve the work's broader artistic aims.

The significance of {topic} extends beyond technical achievement to influence the reader's experience and interpretation. This literary device shapes how readers engage with the text, directing attention, creating emotional responses, and guiding intellectual understanding. The effect on readers reveals how formal elements actively construct meaning rather than merely decorating content.

This analysis of {topic} as a literary device demonstrates the inseparability of form and content in literature. By examining how technical aspects of writing contribute to meaning, we develop a deeper appreciation for literature as a carefully constructed art form that communicates through both what it says and how it is said."""

    def _analyze_chunk(self, chunk: str, topic: str, style: str, word_limit: int) -> str:
        """Analyze a text chunk for relevance to the topic."""
        logger.info(f"Analyzing chunk of {len(chunk)} characters for topic: {topic}")
        
        try:
            # Check if we have a cached analysis for this chunk
            cached_analysis = self._get_cached_chunk_analysis(chunk, topic, style, word_limit)
            if cached_analysis:
                logger.info("Using cached chunk analysis")
                return cached_analysis
            
            # Format the prompt using the template
            prompt = self.prompt_template.format_chunk_analysis_prompt(
                chunk=chunk,
                topic=topic
            )
            
            # Tokenize and prepare inputs
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            
            # Generate the analysis
            with torch.no_grad():
                response = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=TEMPERATURE,
                    do_sample=True
                )
            
            # Decode and extract the response
            analysis = self.tokenizer.decode(response[0], skip_special_tokens=True)
            analysis = self.prompt_template.extract_response(analysis)
            
            # Filter out instruction text
            instruction_keywords = [
                "INSTRUCTIONS:", "Extract key", "Identify character", "Note literary", 
                "Focus ONLY", "Format your", "Source Materials:", "TEXT EXCERPT:",
                "social media", "data analysis",
                "YOUR ANALYSIS", "do not repeat these instructions", "do not include these instructions",
                "start directly with", "ESSAY", "do not"
            ]
            
            # Split into lines and filter out instruction lines
            analysis_lines = analysis.split('\n')
            filtered_lines = []
            
            for line in analysis_lines:
                should_skip = False
                for keyword in instruction_keywords:
                    if keyword.lower() in line.lower():
                        should_skip = True
                        break
                
                # Skip numbered list items that look like instructions
                if re.match(r'^\d+\.\s+(Extract|Identify|Note|Focus|Format)', line.strip()):
                    should_skip = True
                
                if not should_skip:
                    filtered_lines.append(line)
            
            analysis = '\n'.join(filtered_lines)
            
            # Cache the analysis
            self._cache_chunk_analysis(chunk, topic, style, word_limit, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing chunk: {str(e)}")
            return ""

    def _truncate_text(self, text: str, target_words: int) -> str:
        """Truncate text to a target word count while preserving paragraph structure.
        
        Args:
            text: The text to truncate
            target_words: Target number of words
            
        Returns:
            Truncated text
        """
        words = text.split()
        
        if len(words) <= target_words:
            return text
        
        # If we need significant truncation, preserve first 60% and last 40%
        if len(words) > target_words * 1.5:
            first_chunk_size = int(target_words * 0.6)
            last_chunk_size = target_words - first_chunk_size
            
            first_chunk = ' '.join(words[:first_chunk_size])
            last_chunk = ' '.join(words[-last_chunk_size:])
            
            return f"{first_chunk}\n\n[...]\n\n{last_chunk}"
        
        # For minor truncation, just take the first N words
        return ' '.join(words[:target_words])
