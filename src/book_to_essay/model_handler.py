"""Handler for loading and managing the language model."""
import logging
from .model_loader import load_tokenizer, load_model
import torch
from .config import (
    MODEL_NAME, MODEL_CACHE_DIR, MAX_LENGTH, TEMPERATURE, 
    MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS, QUANT_CONFIG
)
from .prompts.factory import PromptTemplateFactory
from .fallback import FallbackReason
import nltk # Added for sentence tokenization
import re
from .chunk_analysis_manager import ChunkAnalysisManager
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class DeepSeekHandler:
    """Handler for the language model."""
    
    def __init__(
        self,
        model: object,
        tokenizer: object,
        prompt_template: Optional[object] = None,
        max_token_threshold: int = 4000,
        truncate_token_target: int = 3500,
        min_essay_length: int = 100,
        chunk_manager: Optional[ChunkAnalysisManager] = None
    ) -> None:
        """
        Initialize the model handler with model, tokenizer, and prompt template.
        Args:
            model: Pre-loaded model to reuse.
            tokenizer: Pre-loaded tokenizer to reuse.
            prompt_template: Optional pre-loaded prompt template to reuse.
            max_token_threshold: Max tokens before truncation is triggered.
            truncate_token_target: Target tokens after truncation.
            min_essay_length: Minimum length in characters for a valid essay.
            chunk_manager: Optional ChunkAnalysisManager instance to reuse.
        Raises:
            RuntimeError: If model or tokenizer fails to initialize.
        """
        self.max_token_threshold = max_token_threshold
        self.truncate_token_target = truncate_token_target
        self.min_essay_length = min_essay_length
        
        # Initialize chunk analysis manager
        self.chunk_manager = chunk_manager or ChunkAnalysisManager()
        # Initialize text_chunks
        self.text_chunks = []

        if model is None or tokenizer is None:
            raise ValueError("ModelHandler requires model and tokenizer to be provided. Use model_loader.py to load them.")

        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template or PromptTemplateFactory.create(MODEL_NAME)
        
        logger.info("Model loaded successfully")
        
        # Set model to evaluation mode for inference
        self.model.eval()
        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)
        
    def process_text(self, text: str) -> None:
        """
        Process input text, tokenize into sentences, and store as chunks using ChunkAnalysisManager.
        Args:
            text: The input text to process.
        Returns:
            None. Updates self.text_chunks in place.
        """
        self.text_chunks = self.chunk_manager.split_text_into_chunks(text)
        logger.info(f"Text processed into {len(self.text_chunks)} chunks")

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
                    mla_citation = f"{source['author']}. {source['title']} ."
                    if 'publisher' in source:
                        mla_citation += f" {source['publisher']},"
                    if 'year' in source:
                        mla_citation += f" {source['year']}."
                    mla_citations.append(mla_citation)
            
            if mla_citations:
                citations_text = "Works Cited\n\n" + "\n".join(mla_citations)
        
        # Process the chunks
        chunk_analyses = []
        logger.info(f"Processing {len(self.text_chunks)} text chunks for analysis")
        
        if not self.text_chunks or all(not chunk.strip() for chunk in self.text_chunks):
            logger.error("All text chunks are empty or whitespace")
            raise RuntimeError("All text chunks are empty or whitespace")
        
        # Analyze each chunk
        for chunk in self.text_chunks:
            analysis = self._analyze_chunk(chunk, topic, style, word_limit)
            chunk_analyses.append(analysis)
        
        # If no analyses were produced, raise an error
        if not chunk_analyses:
            logger.error("No chunk analyses were produced")
            raise RuntimeError("No chunk analyses were produced")
        
        logger.info(f"Successfully analyzed {len(chunk_analyses)} chunks")
        
        # Generate the essay using the analyses
        try:
            # Combine all analyses into one
            combined_analysis = "\n\n".join(chunk_analyses)
            # Always truncate combined_analysis to match test expectation and ensure token safety
            combined_analysis = self._truncate_text(combined_analysis, self.truncate_token_target)
            
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
                    elif len(line.strip()) > 30 and re.match(r'^[A-Z]', line.strip()):
                        # Found a substantive line that starts with uppercase
                        break
                
                # Reconstruct the essay from the starting line
                if start_line > 0 and start_line < len(lines):
                    essay = '\n'.join(lines[start_line:])
                
                if not essay or len(essay.strip()) < self.min_essay_length:
                    logger.warning("Essay too short after filtering, raising error.")
                    raise RuntimeError(f"Essay generation failed due to insufficient content: {len(essay)} characters")
                
                # Verify essay has proper structure
                if essay and len(essay.strip()) >= self.min_essay_length:
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
        """Analyze a text chunk for relevance to the topic, using chunk manager for caching."""
        cached = self.chunk_manager.get_cached_chunk_analysis(chunk, topic, style, word_limit)
        if cached is not None:
            return cached
        analysis = self._perform_chunk_analysis(chunk, topic, style, word_limit)
        self.chunk_manager.cache_chunk_analysis(chunk, topic, style, word_limit, analysis)
        return analysis

    # Helpers for chunk analysis, extracted to simplify _analyze_chunk
    def _perform_chunk_analysis(self, chunk: str, topic: str, style: str, word_limit: int) -> str:
        """Generate analysis for a chunk (without cache)."""
        prompt = self.prompt_template.format_chunk_analysis_prompt(chunk=chunk, topic=topic)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=TEMPERATURE,
                do_sample=True
            )
        analysis = self.tokenizer.decode(response[0], skip_special_tokens=True)
        analysis = self.prompt_template.extract_response(analysis)
        return self._filter_analysis(analysis)

    def _filter_analysis(self, analysis: str) -> str:
        """
        Remove instruction lines from chunk analysis.
        Args:
            analysis: The chunk analysis string.
        Returns:
            The filtered analysis string with instructions removed.
        """
        instruction_keywords = [
            "INSTRUCTIONS:", "Extract key", "Identify character", "Note literary",
            "Focus ONLY", "Format your", "Source Materials:", "TEXT EXCERPT:",
            "social media", "data analysis", "YOUR ANALYSIS",
            "do not repeat these instructions", "do not include these instructions",
            "start directly with", "ESSAY", "do not"
        ]
        lines = analysis.split('\n')
        filtered = []
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in instruction_keywords):
                continue
            if re.match(r'^\d+\.\s+(Extract|Identify|Note|Focus|Format)', line.strip()):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

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
