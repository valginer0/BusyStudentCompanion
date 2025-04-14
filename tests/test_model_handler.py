import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pickle
import hashlib

from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.config import MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS, MODEL_NAME


@pytest.fixture
def handler():
    """Fixture to create a DeepSeekHandler instance with mocked __init__."""
    # Mock __init__ to avoid actual model loading during testing simple methods
    with patch('src.book_to_essay.model_handler.DeepSeekHandler.__init__', return_value=None) as mock_init:
        # Pass dummy model/tokenizer to satisfy __init__ signature if needed,
        # but they won't be used if mocking is correct.
        instance = DeepSeekHandler()
        # Explicitly set model_name as it's set in the real __init__ and used by cache key
        instance.model_name = MODEL_NAME
        # Ensure chunk_cache_dir exists for tests that might need it (like _get_chunk_cache_path)
        instance.chunk_cache_dir = Path('/tmp/test_cache') # Use a temporary path
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

    # --- Tests for Cache Handling ---

    def test_get_cached_chunk_analysis_file_not_exists(self, handler, mocker):
        """Test _get_cached_chunk_analysis when the cache file does not exist."""
        # Mock the path object and its exists method
        mock_path_instance = MagicMock(spec=Path)
        mock_path_instance.exists.return_value = False
        mocker.patch('src.book_to_essay.model_handler.Path', return_value=mock_path_instance)

        # Mock the helper methods called before Path()
        mocker.patch.object(handler, '_get_chunk_cache_key', return_value='dummy_key')
        mocker.patch.object(handler, '_get_chunk_cache_path', return_value=mock_path_instance)

        # Set chunk_cache_dir needed by _get_chunk_cache_path (if not mocked away)
        # Ensure the fixture sets up necessary attributes if not mocking __init__ fully
        handler.chunk_cache_dir = Path('/fake/cache/dir') # Needed by real _get_chunk_cache_path

        result = handler._get_cached_chunk_analysis(
            chunk="test chunk",
            topic="test topic",
            style="test style",
            word_limit=100
        )

        assert result is None, "Should return None when cache file doesn't exist"
        handler._get_chunk_cache_key.assert_called_once_with("test chunk", "test topic", "test style", 100)
        handler._get_chunk_cache_path.assert_called_once_with('dummy_key')
        mock_path_instance.exists.assert_called_once()

    def test_get_cached_chunk_analysis_file_exists(self, handler, mocker):
        """Test _get_cached_chunk_analysis when the cache file exists and is valid."""
        # Mock the path object and its exists method
        mock_path_instance = MagicMock(spec=Path)
        mock_path_instance.exists.return_value = True
        mocker.patch('src.book_to_essay.model_handler.Path', return_value=mock_path_instance)

        # Mock the helper methods called before Path()
        mocker.patch.object(handler, '_get_chunk_cache_key', return_value='dummy_key')
        mocker.patch.object(handler, '_get_chunk_cache_path', return_value=mock_path_instance)
        handler.chunk_cache_dir = Path('/fake/cache/dir')

        # Mock open and pickle.load
        mock_file = MagicMock()
        mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data=b'dummy')) # read_data needed but not used by pickle mock
        mock_pickle_load = mocker.patch('pickle.load', return_value={'analysis': 'cached data'})

        result = handler._get_cached_chunk_analysis(
            chunk="test chunk",
            topic="test topic",
            style="test style",
            word_limit=100
        )

        expected_data = {'analysis': 'cached data'}
        assert result == expected_data, "Should return the cached data"
        handler._get_chunk_cache_key.assert_called_once_with("test chunk", "test topic", "test style", 100)
        handler._get_chunk_cache_path.assert_called_once_with('dummy_key')
        mock_path_instance.exists.assert_called_once()
        mock_open.assert_called_once_with(mock_path_instance, 'rb')
        mock_pickle_load.assert_called_once() # Check it was called, args depend on file handle mock

    def test_get_chunk_cache_key(self, handler):
        """Test that _get_chunk_cache_key produces a consistent SHA256 hash."""
        chunk = "This is a test chunk."
        topic = "Test Topic"
        style = "Academic"
        word_limit = 500

        # Calculate expected key using the handler's actual model_name attribute
        # (Fixture should ensure handler.model_name is set correctly)
        key_string = f"{chunk}-{topic}-{style}-{word_limit}-{handler.model_name}"
        expected_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()

        # Call the method under test
        actual_key = handler._get_chunk_cache_key(chunk, topic, style, word_limit)

        assert actual_key == expected_hash, "Generated cache key does not match expected hash"
        assert len(actual_key) == 64, "Cache key should be a 64-character SHA256 hash"
