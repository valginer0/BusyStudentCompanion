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
from .utils import truncate_text, filter_analysis, prepare_citations, format_essay_from_analyses, postprocess_essay
from .utils import truncate_text, filter_analysis, prepare_citations, format_essay_from_analyses

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
        """Generate an essay using chunk-based analysis."""
        logger.info(f"Generating essay on topic: {topic} with {word_limit} word limit in {style} style")

        if not hasattr(self, 'text_chunks') or not self.text_chunks:
            logger.error("No text has been loaded. Please load text first.")
            raise RuntimeError("No text has been loaded. Please load text first.")

        if sources is None and hasattr(self, 'book_data'):
            sources = [self.book_data]

        mla_citations, citations_text = self._prepare_citations(sources)

        try:
            analyses = self._collect_chunk_analyses(topic, style, word_limit)
            combined_analysis = "\n\n".join(analyses)
            combined_analysis = self._truncate_combined_analysis(combined_analysis)
            prompt = self.prompt_template.format_essay_prompt(
                topic=topic,
                style=style,
                word_limit=word_limit,
                analysis=combined_analysis,
                citations=mla_citations
            )
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(2048, word_limit * 3),
                    temperature=TEMPERATURE,
                    do_sample=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            essay = self.prompt_template.extract_response(response)
            if citations_text and "Works Cited" not in essay:
                essay += f"\n\nWorks Cited:\n{citations_text}"
            essay = self._postprocess_essay(essay, word_limit)
        except Exception as e:
            logger.error(f"Error during essay generation: {str(e)}")
            essay = self._generate_fallback_essay(topic, style, word_limit, reason=FallbackReason.GENERATION_ERROR)
        return essay

    def _prepare_citations(self, sources):
        return prepare_citations(sources)

    def _collect_chunk_analyses(self, topic, style, word_limit):
        analyses = []
        for chunk in self.text_chunks:
            analysis = self._analyze_chunk(chunk, topic, style, word_limit)
            analysis = self._filter_analysis(analysis)
            analyses.append(analysis)
        return analyses

    def _format_essay_from_analyses(self, analyses, citations_text, word_limit, style):
        return format_essay_from_analyses(analyses, citations_text, word_limit, style)

    def _postprocess_essay(self, essay, word_limit):
        return postprocess_essay(essay, word_limit)

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
        return filter_analysis(analysis)

    def _truncate_text(self, text: str, target_words: int) -> str:
        """Truncate text to a target word count while preserving paragraph structure.
        Args:
            text: The text to truncate
            target_words: Target number of words
        Returns:
            Truncated text
        """
        return truncate_text(text, target_words)

    def _truncate_combined_analysis(self, combined_analysis: str) -> str:
        return self._truncate_text(combined_analysis, self.truncate_token_target)
