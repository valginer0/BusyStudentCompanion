import os
import shutil
import tempfile
import pytest
from src.book_to_essay.chunk_analysis_manager import ChunkAnalysisManager

class TestChunkAnalysisManager:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ChunkAnalysisManager(cache_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_split_text_into_chunks_basic(self):
        text = "Sentence one. Sentence two. Sentence three."
        chunks = self.manager.split_text_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0].startswith("Sentence one")

    def test_split_text_long(self):
        # Create a long text that should be split
        sentence = "A long sentence. " * 100
        text = sentence.strip()
        chunks = self.manager.split_text_into_chunks(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 1500

    def test_split_text_empty(self):
        assert self.manager.split_text_into_chunks("") == []
        assert self.manager.split_text_into_chunks("   ") == []

    def test_chunk_cache_roundtrip(self):
        chunk = "test chunk"
        topic = "topic"
        style = "academic"
        word_limit = 500
        analysis = "analysis result"
        # Should not exist yet
        assert self.manager.get_cached_chunk_analysis(chunk, topic, style, word_limit) is None
        # Cache it
        self.manager.cache_chunk_analysis(chunk, topic, style, word_limit, analysis)
        # Should now retrieve
        cached = self.manager.get_cached_chunk_analysis(chunk, topic, style, word_limit)
        assert cached == analysis

    def test_cache_key_uniqueness(self):
        c1 = self.manager.get_chunk_cache_key("a", "t", "s", 1)
        c2 = self.manager.get_chunk_cache_key("a", "t", "s", 2)
        assert c1 != c2
        c3 = self.manager.get_chunk_cache_key("a", "other", "s", 1)
        assert c1 != c3
