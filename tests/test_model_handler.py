import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import pickle
import hashlib
import tempfile

from src.book_to_essay.model_handler import DeepSeekHandler, MODEL_NAME
from src.book_to_essay.config import MAX_CHUNK_SIZE, MAX_CHUNKS_PER_ANALYSIS


# --- Fixture Definition (Module Level) ---
@pytest.fixture
def test_setup_handler():
    """Fixture to set up the handler with mocked dependencies for testing."""
    # 1. Create Mocks needed for configuration and direct assignment
    mock_config = MagicMock()
    mock_config.MODEL_NAME = "test_model" # Needed by factory? Let's keep it.
    # Set config attributes needed by the code under test (e.g., generate_essay)
    mock_config.MIN_ESSAY_LENGTH_THRESHOLD = 50
    mock_config.FILTERING_PATTERNS = ("instruction:", "summary:")
    mock_config.SKIP_PATTERNS = (r'^\s*#', r'^\s*---')
    mock_config.START_PATTERNS = (r'^(The|An|In|Based on)', r'^[A-Z][a-z]+\s')
    # Add other attrs needed by generate_essay if any
    mock_config.MAX_CHUNK_SIZE = 1000 # Example: Assuming generate_essay needs this
    mock_config.CHUNK_OVERLAP = 100   # Example: Assuming generate_essay needs this

    mock_prompt_template = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # 2. Patch dependencies called during DeepSeekHandler.__init__
    # Patching where they are looked up (in model_handler or factory)
    with patch('src.book_to_essay.model_handler.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('src.book_to_essay.model_handler.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
         patch('src.book_to_essay.prompts.factory.PromptTemplateFactory.create', return_value=mock_prompt_template):
        # 3. Initialize the handler (should use mocks now)
        handler = DeepSeekHandler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompt_template=mock_prompt_template,
            max_token_threshold=10,      # very low for test
            truncate_token_target=5,     # very low for test
            min_essay_length=10          # very low for test
        )

    # 4. Manually assign the config object and mocks
    handler.config = mock_config
    handler.model = mock_model # Ensure model mock is assigned
    handler.tokenizer = mock_tokenizer # Ensure tokenizer mock is assigned
    handler.prompt_template = mock_prompt_template # Ensure template mock is assigned

    # CacheManager will be mocked per-test if necessary

    return handler
# -----------------------------------------

@pytest.fixture
def handler():
    """Fixture to create a DeepSeekHandler instance with mocked __init__."""
    # Use a temporary directory for cache that gets cleaned up
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override MODEL_CACHE_DIR
        with patch('src.book_to_essay.model_handler.MODEL_CACHE_DIR', tmpdir):
            # Mock the model and tokenizer loading to avoid actual downloads/loading
            mock_model_instance = MagicMock()
            mock_model_instance.eval = MagicMock() # Ensure the mock model has an eval method
            mock_tokenizer_instance = MagicMock()
            mock_prompt_instance = MagicMock()

            with patch('src.book_to_essay.model_handler.AutoModelForCausalLM.from_pretrained', return_value=mock_model_instance) as mock_model_load, \
                 patch('src.book_to_essay.model_handler.AutoTokenizer.from_pretrained', return_value=mock_tokenizer_instance) as mock_tokenizer_load, \
                 patch('src.book_to_essay.model_handler.PromptTemplateFactory.create', return_value=mock_prompt_instance) as mock_prompt_factory:

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

    # --- Tests for generate_essay --- #

    @patch('src.book_to_essay.model_handler.logger') # Patch logger to suppress output
    def test_generate_essay_basic(
        self,
        mock_logger,           # Only logger mock from decorator remains
        test_setup_handler    # Inject the fixture result
    ):
        """Tests the basic successful flow of generate_essay, including truncation."""
        # 1. Setup
        # Get handler instance from fixture
        handler = test_setup_handler # Assign injected fixture result

        # Define test variables
        test_topic = "The Impact of AI"
        test_limit = 150 # Set a limit that will trigger truncation
        test_style = "Persuasive"
        mock_text_chunks = ["Chunk 1 content.", "Chunk 2 content."]
        # Each string is 6 words, so total is 12 words (threshold is 10)
        mock_analyses = ["word " * 6, "word " * 6]
        combined_analysis = "\n\n".join(mock_analyses)

        # --- Configure mocks directly on the handler instance --- #
        handler._get_cached_chunk_analysis = MagicMock(return_value=None)
        handler._analyze_chunk = MagicMock(side_effect=mock_analyses)
        handler._cache_chunk_analysis = MagicMock()
        final_essay_truncated = "Final essay result." # Expected final result
        handler._truncate_text = MagicMock(side_effect=lambda text, limit: final_essay_truncated)

        # Mock text chunks on the instance
        handler.text_chunks = mock_text_chunks

        # --- Mock interactions not directly part of the handler instance --- #
        # Define what the raw output from the model's tokenizer.decode would be
        # Make it long enough to ensure word count exceeds test_limit
        placeholder_text = "word " * (test_limit + 50) # Ensure significantly over limit
        raw_decoded_essay = (
            "Introduction: AI is impactful.\n\n" # Example starting text
            "Here is the main part. We need this part to be significantly longer "
            "to ensure that after any potential filtering or trimming of initial lines, "
            "the remaining content still meets the hardcoded threshold of 100 characters. "
            "Adding more sentences here to increase the length substantially.\n"
            f"{placeholder_text}\n" 
            "Even more placeholder text to ensure we cross the word limit significantly. "
            "Repeating words helps increase the count quickly for testing purposes. "
            "Word count must exceed 150 words to trigger the truncation logic correctly. "
            "Testing testing one two three. Placeholder placeholder placeholder. "
            "Adding just a bit more to be absolutely sure we are over the limit.\n"
            "\n\n\n"
            "It also has extra newlines and perhaps a summary:\n" # Line to be filtered?
            "Summary: Final thoughts." # Line to be filtered?
        )

        # Mock the model/tokenizer interactions within generate_essay
        handler.tokenizer = MagicMock()
        handler.model = MagicMock()

        # Mock generate() call specifics
        mock_tokenized_output = MagicMock()
        mock_generated_tokens = [1, 2, 3] # Dummy tokens
        handler.tokenizer.return_value = mock_tokenized_output
        mock_tokenized_output.to.return_value = mock_tokenized_output # Simulate .to(device)
        handler.model.generate.return_value = [mock_generated_tokens] # generate returns a list

        # Mock parameter checking for device
        mock_param = MagicMock()
        mock_param.device = 'cpu' # Mock device
        handler.model.parameters.return_value = iter([mock_param])

        # Mock the decoded output and the extraction step
        handler.tokenizer.decode.return_value = final_essay_truncated # Mock decoding

        # Mock the prompt template generation and extraction
        mock_final_prompt = "Final essay prompt for AI impact."
        handler.prompt_template = MagicMock()
        handler.prompt_template.format_essay_prompt.return_value = mock_final_prompt
        # Configure extract_response to return the truncated essay if input matches, else the long essay
        def mock_extract_response(input_str):
            if input_str == final_essay_truncated:
                return final_essay_truncated
            return raw_decoded_essay
        handler.prompt_template.extract_response.side_effect = mock_extract_response

        # 2. Execute
        result = handler.generate_essay(topic=test_topic, word_limit=test_limit, style=test_style)

        # 3. Assert
        # Check that truncation was triggered
        assert handler._truncate_text.called, "_truncate_text was not called!"
        print(f"_truncate_text call args: {handler._truncate_text.call_args}")
        assert result == final_essay_truncated

        # Verify mock calls
        expected_analyze_calls = [
            call(handler.text_chunks[0], test_topic, test_style, test_limit),
            call(handler.text_chunks[1], test_topic, test_style, test_limit)
        ]
        handler._analyze_chunk.assert_has_calls(expected_analyze_calls)

        # Check prompt template call (for final essay)
        handler.prompt_template.format_essay_prompt.assert_called_once_with(
            topic=test_topic,
            style=test_style,
            word_limit=test_limit,
            analysis=final_essay_truncated,  # Truncated value used
            citations=None # Assuming None for MLA citations in basic test
        )

        # Verify the direct model/tokenizer calls within generate_essay for final step
        handler.tokenizer.assert_called_once_with(mock_final_prompt, return_tensors="pt", truncation=True, max_length=4096)
        mock_tokenized_output.to.assert_called_once() # Check device transfer
        handler.model.generate.assert_called_once() # Check generate was called
        # More specific check on generate args if needed:
        # args, kwargs = handler.model.generate.call_args
        # assert kwargs['max_new_tokens'] == test_limit * 2
        # assert kwargs['pad_token_id'] == handler.tokenizer.eos_token_id

        handler.tokenizer.decode.assert_called_once_with(mock_generated_tokens, skip_special_tokens=True)

        # Verify truncate call
        handler._truncate_text.assert_called_once()
        truncate_args, truncate_kwargs = handler._truncate_text.call_args
        # Assert that truncate was called with the direct output of extract_response
        assert truncate_args[0] == combined_analysis
        assert truncate_args[1] == handler.truncate_token_target

        # Assert logger calls (optional, using mock_logger)

    # Removed fixture definition from inside the class
