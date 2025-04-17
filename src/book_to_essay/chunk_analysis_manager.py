"""Handles chunk splitting, analysis, and caching for essay generation."""
import os
import pickle
from pathlib import Path
from typing import List, Optional
import logging
from .config import MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS, MODEL_CACHE_DIR
from .chunk_utilities import get_chunk_cache_key, get_chunk_cache_path
import nltk

logger = logging.getLogger(__name__)

class ChunkAnalysisManager:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the chunk analysis manager, including cache directory.
        Args:
            cache_dir: Directory for chunk analysis cache. Defaults to MODEL_CACHE_DIR/chunk_cache
        """
        self.chunk_cache_dir = Path(cache_dir or os.path.join(MODEL_CACHE_DIR, "chunk_cache"))
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split input text into manageable chunks for analysis.
        Args:
            text: Input text to split.
        Returns:
            List of text chunks.
        """
        stripped_text = text.strip()
        if not stripped_text:
            return []
        if len(stripped_text) <= MAX_CHUNK_SIZE:
            return [stripped_text]
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(stripped_text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue
            if len(sentence_stripped) > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                chunks.append(sentence_stripped)
                current_chunk = ""
                continue
            if current_chunk and len(current_chunk) + len(sentence_stripped) + 1 > MAX_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_stripped
            else:
                current_chunk = (current_chunk + " " + sentence_stripped).strip() if current_chunk else sentence_stripped
        if current_chunk:
            chunks.append(current_chunk.strip())
        if len(chunks) > MAX_CHUNKS_PER_ANALYSIS:
            step = len(chunks) / MAX_CHUNKS_PER_ANALYSIS
            selected_chunks = [chunks[min(int(i * step), len(chunks) - 1)] for i in range(MAX_CHUNKS_PER_ANALYSIS)]
            return selected_chunks
        return chunks

    def get_chunk_cache_key(self, chunk: str, topic: str, style: str, word_limit: int) -> str:
        return get_chunk_cache_key(chunk, topic, style, word_limit)

    def get_chunk_cache_path(self, cache_key: str) -> Path:
        return get_chunk_cache_path(cache_key, self.chunk_cache_dir)

    def get_cached_chunk_analysis(self, chunk: str, topic: str, style: str, word_limit: int) -> Optional[str]:
        """
        Retrieve cached chunk analysis if available.
        """
        cache_key = self.get_chunk_cache_key(chunk, topic, style, word_limit)
        cache_path = self.get_chunk_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Using cached chunk analysis for key: {cache_key[:8]}...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def cache_chunk_analysis(self, chunk: str, topic: str, style: str, word_limit: int, analysis: str) -> None:
        """
        Cache the chunk analysis for future reuse.
        """
        cache_key = self.get_chunk_cache_key(chunk, topic, style, word_limit)
        cache_path = self.get_chunk_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(analysis, f)
        logger.info(f"Cached chunk analysis with key: {cache_key[:8]}...")
