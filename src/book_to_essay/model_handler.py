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
from src.book_to_essay.error_utils import log_and_raise
from src.book_to_essay.prompts.config import PromptConfig

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
            log_and_raise("No text has been loaded. Please load text first.", None, RuntimeError)

        if sources is None and hasattr(self, 'book_data'):
            sources = [self.book_data]

        try:
            analyses = self._collect_chunk_analyses(topic, style, word_limit)
            citations, citations_text = self._prepare_citations(sources)
            essay = self.prompt_template.format_essay_from_analyses(analyses, citations_text, word_limit, style)
            essay = self._postprocess_essay(essay, word_limit)
        except Exception as e:
            log_and_raise("Error during essay generation", e, RuntimeError)

        if not essay or not essay.strip():
            log_and_raise("Essay generation failed: empty result returned.", None, RuntimeError)

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
        config = PromptConfig(
            analysis_text=chunk,
            topic=topic,
            style=style,
            word_limit=word_limit
        )
        prompt = self.prompt_template.format_chunk_analysis_prompt(config)
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
