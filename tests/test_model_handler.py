import pytest
from unittest.mock import patch
from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.config import MAX_CHUNK_SIZE


@pytest.fixture
def handler():
    """Fixture to create a DeepSeekHandler instance with mocked __init__."""
    # Mock __init__ to avoid actual model loading during testing simple methods
    with patch('src.book_to_essay.model_handler.DeepSeekHandler.__init__', return_value=None) as mock_init:
        instance = DeepSeekHandler() # This call uses the mocked __init__
        # Manually initialize attributes normally set by __init__ that process_text might rely on
        instance.text_chunks = []
        # process_text doesn't seem to rely on tokenizer or model directly
        yield instance


class TestDeepSeekHandler:

    def test_process_text_short(self, handler):
        """Test processing text shorter than MAX_CHUNK_SIZE."""
        short_text = "This is a sentence. This is another sentence. " * 5
        # Ensure the test text is actually shorter than the limit
        assert len(short_text) < MAX_CHUNK_SIZE, \
            f"Test text length ({len(short_text)}) is not less than MAX_CHUNK_SIZE ({MAX_CHUNK_SIZE})"

        handler.process_text(short_text)

        assert len(handler.text_chunks) == 1, "Should produce exactly one chunk for short text"
        # process_text seems to strip the final chunk, let's verify
        assert handler.text_chunks[0] == short_text.strip(), "The single chunk should match the stripped input text"

    def test_process_text_empty(self, handler):
        """Test processing empty text."""
        handler.process_text("")
        assert handler.text_chunks == [], "Processing empty text should result in empty chunks"
        handler.process_text("   \n  \t ")
        assert handler.text_chunks == [], "Processing whitespace-only text should result in empty chunks"

    def test_process_text_splits_correctly(self, handler):
        """Test processing text slightly longer than MAX_CHUNK_SIZE."""
        # Create text that should definitely split
        sentence1 = "This is the first sentence. " * (MAX_CHUNK_SIZE // 40) # Approx MAX_CHUNK_SIZE / 2
        sentence2 = "This is the second sentence that pushes it over. " * (MAX_CHUNK_SIZE // 50)
        long_text = sentence1 + sentence2
        stripped_long_text = long_text.strip()
        assert len(stripped_long_text) > MAX_CHUNK_SIZE, "Test text must be longer than MAX_CHUNK_SIZE"

        handler.process_text(long_text)

        # 1. Check that splitting occurred (more than 1 chunk)
        assert len(handler.text_chunks) >= 2, f"Should split into at least two chunks, but got {len(handler.text_chunks)}"

        # 2. Check that the joined chunks reconstruct the original text
        reconstructed_text = " ".join(handler.text_chunks)
        # Allow for minor whitespace differences introduced by joining/stripping
        assert reconstructed_text.replace(" ", "") == stripped_long_text.replace(" ", ""), \
            "Reconstructed text does not match original (ignoring spaces)"

        # 3. Check that each chunk respects MAX_CHUNK_SIZE
        for i, chunk in enumerate(handler.text_chunks):
            assert len(chunk) <= MAX_CHUNK_SIZE, \
                f"Chunk {i} length ({len(chunk)}) exceeds MAX_CHUNK_SIZE ({MAX_CHUNK_SIZE})"

    def test_process_text_long_sentence(self, handler):
        """Test processing text where a single sentence exceeds MAX_CHUNK_SIZE."""
        # Add a period to ensure nltk recognizes it as a sentence
        long_sentence = "L" * (MAX_CHUNK_SIZE + 10) + "."
        text = f"Short first sentence. {long_sentence} Short third sentence."

        handler.process_text(text)

        # Current implementation puts the too-long sentence in its own chunk
        assert len(handler.text_chunks) == 3, "Should split into three chunks"
        assert handler.text_chunks[0] == "Short first sentence.", "First chunk incorrect"
        assert handler.text_chunks[1] == long_sentence, "Long sentence chunk incorrect"
        assert handler.text_chunks[2] == "Short third sentence.", "Third chunk incorrect"

    def test_process_text_exceeds_max_chunks(self, handler, monkeypatch):
        """Test processing text that generates more chunks than MAX_CHUNKS_PER_ANALYSIS."""
        # Temporarily lower MAX_CHUNKS_PER_ANALYSIS for this test
        test_max_chunks = 2
        monkeypatch.setattr('src.book_to_essay.model_handler.MAX_CHUNKS_PER_ANALYSIS', test_max_chunks)

        # Create text that will generate more than test_max_chunks
        sentence = "This is a sentence. "
        num_sentences = (test_max_chunks + 2) * (MAX_CHUNK_SIZE // (len(sentence) + 5)) # Ensure enough sentences
        text = (sentence * num_sentences)

        # Verify the setup - it should create more than test_max_chunks if not limited
        handler_temp = DeepSeekHandler() # Need a fresh instance for calculation
        handler_temp.text_chunks = []
        sentences = [s for s in text.strip().split('.') if s]
        assert len(sentences) > test_max_chunks, "Test setup failed: not enough sentences generated"

        handler.process_text(text)

        assert len(handler.text_chunks) == test_max_chunks, \
            f"Should truncate chunks to MAX_CHUNKS_PER_ANALYSIS ({test_max_chunks})"

    # --- More tests for process_text will be added below ---
