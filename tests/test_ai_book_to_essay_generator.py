"""Tests for the AI Book Essay Generator."""
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from pytest_mock import MockerFixture
import ebooklib

from src.book_to_essay.ai_book_to_essay_generator import AIBookEssayGenerator

def create_test_file(path: str, content: str = "test content"):
    """Create a test file with content."""
    with open(path, 'w') as f:
        f.write(content)

def create_processor(content: str):
    """Create a processor function that returns the content."""
    def processor(file_path: str) -> str:
        return content
    return processor

def test_generator_initialization():
    """Test generator initialization."""
    generator = AIBookEssayGenerator()
    assert generator is not None
    assert generator.content == ""
    assert generator.sources == []

def test_process_pdf_content(mocker, temp_cache_dir, sample_pdf_content, mock_deepseek_response):
    """Test processing PDF content."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)

    # Create test file
    file_path = os.path.join(temp_cache_dir, "test.pdf")
    create_test_file(file_path, sample_pdf_content)

    # Mock the model response
    mock_model = mocker.patch('src.book_to_essay.model_handler.DeepSeekHandler')
    mock_model.return_value.generate_essay.return_value = mock_deepseek_response

    # Mock PDF processing
    mocker.patch('fitz.open', return_value=mocker.MagicMock())

    processor = create_processor(sample_pdf_content)
    result = generator._process_file_content(file_path, processor)
    assert result is not None

def test_content_caching(mocker, temp_cache_dir, sample_pdf_content):
    """Test content caching during processing."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)

    # Create test file
    file_path = os.path.join(temp_cache_dir, "test.pdf")
    create_test_file(file_path, sample_pdf_content)

    # Process content first time
    mocker.patch('fitz.open', return_value=mocker.MagicMock())
    processor = create_processor(sample_pdf_content)
    generator._process_file_content(file_path, processor)
    assert len(generator.sources) == 1

    # Process same content again
    generator._process_file_content(file_path, processor)
    assert len(generator.sources) == 2  # Should add source again

def test_process_content_async(mocker, temp_cache_dir, sample_pdf_content, mock_deepseek_response):
    """Test asynchronous content processing."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)

    # Create test file
    file_path = os.path.join(temp_cache_dir, "test.pdf")
    create_test_file(file_path, sample_pdf_content)

    # Mock the model response
    mock_model = mocker.patch('src.book_to_essay.model_handler.DeepSeekHandler')
    mock_model.return_value.generate_essay.return_value = mock_deepseek_response

    # Mock PDF processing
    mocker.patch('fitz.open', return_value=mocker.MagicMock())

    # Use the synchronous method since async isn't implemented
    processor = create_processor(sample_pdf_content)
    result = generator._process_file_content(file_path, processor)
    assert result is not None

def test_generate_essay_success(mocker, temp_cache_dir, sample_pdf_content, mock_deepseek_response):
    """Test successful essay generation call."""
    # Create the generator instance first
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)

    # Configure the mock handler
    mock_handler_instance = MagicMock()
    mock_handler_instance.generate_essay.return_value = mock_deepseek_response
    mock_handler_instance.text_chunks = ['dummy chunk'] # Ensure text_chunks exists

    # Assign the mock directly to the internal attribute
    generator._model = mock_handler_instance

    # Mock cache manager to prevent early exit
    mocker.patch.object(generator.cache_manager, 'get_cached_model_output', return_value=None)

    generator.content = sample_pdf_content # Simulate content loading

    # Call generate_essay
    essay = generator.generate_essay("Test Topic", word_limit=100, style="academic")

    # Assertions
    assert essay == mock_deepseek_response
    mock_handler_instance.generate_essay.assert_called_once_with(
        topic="Test Topic",
        word_limit=100,
        style="academic",
        sources=[] # Assuming no file loaded via load_*, only content set directly
    )

def test_generate_essay_handler_error(mocker, temp_cache_dir, sample_pdf_content):
    """Test that errors from the model handler are propagated as ValueErrors."""
    # Create the generator instance first
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)

    # Configure the mock handler
    mock_handler_instance = MagicMock()
    mock_handler_instance.generate_essay.side_effect = RuntimeError("Model generation failed")
    mock_handler_instance.text_chunks = ['dummy chunk'] # Ensure text_chunks exists

    # Assign the mock directly to the internal attribute
    generator._model = mock_handler_instance

    # Mock cache manager to prevent early exit
    mocker.patch.object(generator.cache_manager, 'get_cached_model_output', return_value=None)

    # Set content (needed for the generate_essay call itself)
    generator.content = sample_pdf_content

    # Assert that calling generate_essay raises ValueError wrapping the original error
    with pytest.raises(ValueError, match="Error generating essay: Model generation failed"):
        generator.generate_essay("Test Topic", 100, "academic")

    mock_handler_instance.generate_essay.assert_called_once()

# === Tests for Specific File Types ===

def test_process_txt_file_success(mocker, temp_cache_dir):
    """Test successful processing of a TXT file via load_txt_file."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    sample_txt_content = "This is simple text content."

    # Create test file
    file_path = os.path.join(temp_cache_dir, "test_book.txt")
    create_test_file(file_path, sample_txt_content)

    # Mock dependencies: cache manager and model handler's process_text
    mock_get_cache = mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)
    mock_cache_content = mocker.patch.object(generator.cache_manager, 'cache_content')
    # Mock the model property to return a mock handler
    mock_handler_instance = MagicMock()
    mock_handler_instance.process_text = MagicMock()
    mocker.patch.object(AIBookEssayGenerator, 'model', new_callable=mocker.PropertyMock, return_value=mock_handler_instance)

    # Call the public loading method
    generator.load_txt_file(file_path)

    # Assertions
    mock_get_cache.assert_called_once_with(file_path)
    # Assert that the handler's process_text was called with the content
    mock_handler_instance.process_text.assert_called_once_with(sample_txt_content)
    # Assert cache was updated
    # We need to check the arguments cache_content was called with
    mock_cache_content.assert_called_once()
    call_args = mock_cache_content.call_args[0]
    assert call_args[0] == file_path
    cached_data = call_args[1]
    assert cached_data['content'] == sample_txt_content
    assert cached_data['source']['path'] == file_path
    assert cached_data['source']['name'] == "test_book.txt"
    assert cached_data['source']['type'] == 'txt'
    # Assert generator state was updated
    assert generator.content.strip() == sample_txt_content # strip trailing newline added
    assert len(generator.sources) == 1
    assert generator.sources[0]['path'] == file_path
    assert generator.sources[0]['name'] == "test_book.txt"
    assert generator.sources[0]['type'] == 'txt'

def test_load_pdf_file_success(mocker, temp_cache_dir):
    """Test successful processing of a PDF file via load_pdf_file, mocking fitz."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    sample_pdf_text = "This is text from a PDF page."
    num_pages = 2

    # Create dummy test file path (doesn't need to exist as fitz is mocked)
    file_path = os.path.join(temp_cache_dir, "test_book.pdf")
    # Touch the file so os.path.exists passes in _process_file_content
    Path(file_path).touch()

    # Mock dependencies: cache, model, and fitz (PyMuPDF)
    mock_get_cache = mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)
    mock_cache_content = mocker.patch.object(generator.cache_manager, 'cache_content')
    mock_handler_instance = MagicMock()
    mock_handler_instance.process_text = MagicMock()
    mocker.patch.object(AIBookEssayGenerator, 'model', new_callable=mocker.PropertyMock, return_value=mock_handler_instance)

    # --- Mock fitz (PyMuPDF) --- 
    mock_pdf_page = MagicMock()
    mock_pdf_page.get_text.return_value = sample_pdf_text
    
    mock_pdf_doc = MagicMock()
    mock_pdf_doc.load_page.return_value = mock_pdf_page
    # Configure __len__ for the loop range(len(pdf_document))
    mock_pdf_doc.__len__.return_value = num_pages 
    
    mock_fitz_open = mocker.patch('src.book_to_essay.ai_book_to_essay_generator.fitz.open', return_value=mock_pdf_doc)
    # ---------------------------

    # Call the public loading method
    generator.load_pdf_file(file_path)

    # Assertions
    mock_get_cache.assert_called_once_with(file_path)
    mock_fitz_open.assert_called_once_with(file_path)
    assert mock_pdf_doc.load_page.call_count == num_pages
    assert mock_pdf_page.get_text.call_count == num_pages
    mock_pdf_doc.close.assert_called_once()

    # Assert model processing was called with concatenated text (+ newline per page)
    expected_content = (sample_pdf_text + '\n') * num_pages
    mock_handler_instance.process_text.assert_called_once_with(expected_content)
    
    # Assert cache was updated
    mock_cache_content.assert_called_once()
    call_args = mock_cache_content.call_args[0]
    assert call_args[0] == file_path
    cached_data = call_args[1]
    assert cached_data['content'] == expected_content # Content saved is NOT stripped
    assert cached_data['source']['path'] == file_path
    assert cached_data['source']['name'] == "test_book.pdf"
    assert cached_data['source']['type'] == 'pdf'

    # Assert generator state was updated (content has extra newline added by _process_file_content)
    assert generator.content.strip() == expected_content.strip()
    assert len(generator.sources) == 1
    assert generator.sources[0]['path'] == file_path
    assert generator.sources[0]['name'] == "test_book.pdf"
    assert generator.sources[0]['type'] == 'pdf'

def test_load_epub_file_success(mocker, temp_cache_dir):
    """Test successful processing of an EPUB file via load_epub_file, mocking ebooklib."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    sample_epub_content_part1 = b'<html><body><p>Chapter 1 content.</p></body></html>'
    sample_epub_content_part2 = b'<html><body><p>Chapter 2 content.</p></body></html>'
    expected_text_part1 = "Chapter 1 content."
    expected_text_part2 = "Chapter 2 content."

    # Create dummy test file path (doesn't need to exist as ebooklib is mocked)
    file_path = os.path.join(temp_cache_dir, "test_book.epub")
    Path(file_path).touch()

    # Mock dependencies: cache, model, and ebooklib
    mock_get_cache = mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)
    mock_cache_content = mocker.patch.object(generator.cache_manager, 'cache_content')
    mock_handler_instance = MagicMock()
    mock_handler_instance.process_text = MagicMock()
    mocker.patch.object(AIBookEssayGenerator, 'model', new_callable=mocker.PropertyMock, return_value=mock_handler_instance)

    # --- Mock ebooklib --- 
    mock_item1 = MagicMock()
    mock_item1.get_content.return_value = sample_epub_content_part1
    mock_item1.get_type.return_value = ebooklib.ITEM_DOCUMENT # Configure get_type
    mock_item2 = MagicMock()
    mock_item2.get_content.return_value = sample_epub_content_part2
    mock_item2.get_type.return_value = ebooklib.ITEM_DOCUMENT # Configure get_type
    
    mock_epub_book = MagicMock()
    # Mock get_items() as used by the code, not get_items_of_type()
    mock_epub_book.get_items.return_value = [mock_item1, mock_item2] 
    
    mock_read_epub = mocker.patch('src.book_to_essay.ai_book_to_essay_generator.epub.read_epub', return_value=mock_epub_book)
    # ---------------------

    # Call the public loading method
    generator.load_epub_file(file_path)

    # Assertions
    mock_get_cache.assert_called_once_with(file_path)
    mock_read_epub.assert_called_once_with(file_path)
    # Assert that get_items was called (as used by the code)
    mock_epub_book.get_items.assert_called_once()
    assert mock_item1.get_content.call_count == 1
    assert mock_item1.get_type.call_count == 1 # Check get_type was called
    assert mock_item2.get_content.call_count == 1
    assert mock_item2.get_type.call_count == 1 # Check get_type was called

    # Assert model processing was called with concatenated & cleaned text (+ newline per item)
    expected_content = f"{expected_text_part1}\n{expected_text_part2}\n"
    mock_handler_instance.process_text.assert_called_once_with(expected_content)

    # Assert cache was updated
    mock_cache_content.assert_called_once()
    call_args = mock_cache_content.call_args[0]
    assert call_args[0] == file_path
    cached_data = call_args[1]
    assert cached_data['content'] == expected_content
    assert cached_data['source']['path'] == file_path
    assert cached_data['source']['name'] == "test_book.epub"
    assert cached_data['source']['type'] == 'epub'

    # Assert generator state was updated (content has extra newline added by _process_file_content)
    assert generator.content.strip() == expected_content.strip()
    assert len(generator.sources) == 1
    assert generator.sources[0]['path'] == file_path
    assert generator.sources[0]['name'] == "test_book.epub"
    assert generator.sources[0]['type'] == 'epub'

# === Additional Tests ===

@pytest.mark.parametrize(
    "load_method_name, mock_target, file_extension, error_message_part",
    [
        (
            "load_txt_file",
            "src.book_to_essay.ai_book_to_essay_generator.open", # Target module-specific open
            ".txt",
            "Error reading TXT"
        ),
        (
            "load_pdf_file",
            "src.book_to_essay.ai_book_to_essay_generator.fitz.open",
            ".pdf",
            "Error reading PDF"
        ),
        (
            "load_epub_file",
            "src.book_to_essay.ai_book_to_essay_generator.epub.read_epub",
            ".epub",
            "Error reading EPUB"
        ),
    ]
)
def test_load_file_processing_error(mocker, tmp_path, load_method_name, mock_target, file_extension, error_message_part):
    """Test that errors during file processing are caught and raise ValueError."""
    generator = AIBookEssayGenerator()
    file_path = tmp_path / f"test_file{file_extension}"
    file_path.touch()
    
    # Mock cache miss so processing is attempted
    mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)

    # Special handling for TXT read error vs. PDF/EPUB processing errors
    if load_method_name == "load_txt_file":
        mock_file = MagicMock()
        # Simulate error during file.read() within the 'with' block
        mock_file.read.side_effect = IOError("Simulated read error")
        # Make the mock usable as a context manager
        mock_file.__enter__.return_value = mock_file 
        mock_file.__exit__.return_value = None # Ensure __exit__ doesn't suppress errors
        # Patch the module-specific 'open' to return our mock file object
        mock_open_patch = mocker.patch(mock_target, return_value=mock_file)
        # We need to track the read call for assertion
        mocked_processor_call_check = mock_file.read 
    else:
        # Mock the specific processing function for PDF/EPUB to raise an error
        mocked_processor = mocker.patch(mock_target, side_effect=Exception("Simulated processing error"))
        # Track the processor call for assertion
        mocked_processor_call_check = mocked_processor 

    # Get the actual load method
    load_method = getattr(generator, load_method_name)

    # Assert that ValueError is raised with the expected message
    with pytest.raises(ValueError, match=error_message_part):
        load_method(str(file_path))

    # Ensure the correct mocked part was called
    if load_method_name == "load_txt_file":
        # Check that our patched 'open' was called correctly
        mock_open_patch.assert_called_once_with(str(file_path), 'r', encoding='utf-8')
        # Check that 'read' was called on the mock file object
        mocked_processor_call_check.assert_called_once()
    else:
        # Check that the PDF/EPUB processor function was called
        mocked_processor_call_check.assert_called_once_with(str(file_path))

def test_load_file_cache_hit(mocker, tmp_path):
    """Test that loading the same file twice results in a cache hit on the second call."""
    generator = AIBookEssayGenerator()
    file_path = tmp_path / "cache_test.txt"
    file_content = "This is the content to be cached."
    file_path.write_text(file_content, encoding='utf-8')

    # Mock CacheManager methods
    mock_get_cache = mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)
    mock_cache_content = mocker.patch.object(generator.cache_manager, 'cache_content')
    
    # Mock the actual processing function just to ensure it's called only once
    # We need to mock the nested function within load_txt_file
    # Note: This requires accessing the inner function definition, which is tricky.
    # Instead, let's verify behavior via cache mock calls.
    # We expect process_txt -> cache_content on first call.
    # We expect get_cached_content -> no process_txt -> no cache_content on second call.

    # --- First call (Cache Miss) ---
    generator.load_txt_file(str(file_path))
    
    # Assertions for first call
    mock_get_cache.assert_called_once_with(str(file_path))
    # process_txt should have run and triggered cache_content with the correct structure
    expected_cached_data = {
        "content": file_content,
        "source": {"path": str(file_path), "name": "cache_test.txt", "type": "txt"}
    }
    mock_cache_content.assert_called_once_with(str(file_path), expected_cached_data)
    assert generator.content == file_content + "\n" # Content includes added newline
    
    # --- Prepare for Second Call (Cache Hit) ---
    # Reset relevant mocks and configure get_cache to return data
    mock_get_cache.reset_mock()
    mock_cache_content.reset_mock()
    # Use the same structure for the cached data returned by get_cache
    cached_data = expected_cached_data 
    mock_get_cache.return_value = cached_data
    
    # --- Second call (Cache Hit) ---
    generator.load_txt_file(str(file_path))
    
    # Assertions for second call
    mock_get_cache.assert_called_once_with(str(file_path)) # Called again
    mock_cache_content.assert_not_called() # Should not be called again
    # Content should now be duplicated because it's loaded twice
    assert generator.content == (file_content + "\n") + (file_content + "\n")


def test_generate_essay_model_error(mocker, temp_cache_dir):
    """Test handling of errors raised by the model during essay generation."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    error_message = "Simulated model generation error"
    
    # --- Mocking ---
    # Mock model property and its generate_essay method to raise an error
    mock_model_instance = MagicMock()
    mock_model_instance.generate_essay.side_effect = RuntimeError(error_message) # Simulate underlying model error
    mock_model_instance.text_chunks = ['dummy chunk'] # Ensure text_chunks exists
    generator._model = mock_model_instance # Assign mock directly
    
    # Mock cache manager to prevent early exit (for model output cache)
    mocker.patch.object(generator.cache_manager, 'get_cached_model_output', return_value=None)

    # Add some dummy content/source so generate_essay can be called
    generator.content = "Some processed content\n"
    generator.sources = [{'path': '/fake/source.txt', 'name': 'source.txt', 'type': 'txt'}]
    
    # --- Action & Assertion ---
    # generate_essay should catch the RuntimeError and raise a ValueError
    with pytest.raises(ValueError, match=f"Error generating essay: {error_message}"):
        generator.generate_essay(prompt="Generate on this topic", word_limit=500, style="Academic")
        
    # Ensure model's generate_essay was called
    mock_model_instance.generate_essay.assert_called_once()

def test_generate_essay_empty_result(mocker, temp_cache_dir):
    """Test handling when the model returns an empty result for the essay."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    
    # --- Mocking ---
    # Mock model property and its generate_essay method to return None
    mock_model_instance = MagicMock()
    mock_model_instance.generate_essay.return_value = None # Or could be ""
    # Mock the fallback method on the model instance to also return None
    mock_model_instance.generate_fallback_essay.return_value = None 
    mock_model_instance.text_chunks = ['dummy chunk'] # Ensure text_chunks exists
    generator._model = mock_model_instance # Assign mock directly

    # Mock cache manager to prevent early exit (for model output cache)
    mocker.patch.object(generator.cache_manager, 'get_cached_model_output', return_value=None)
    
    # Add some dummy content/source so generate_essay can be called
    generator.content = "Some processed content\n"
    generator.sources = [{'path': '/fake/source.txt', 'name': 'source.txt', 'type': 'txt'}]

    # --- Action & Assertion ---
    # Expecting the code to use the fallback and then raise ValueError
    with pytest.raises(ValueError, match="Failed to generate essay after fallback."):
        generator.generate_essay(prompt="Generate on this topic", word_limit=500, style="Academic")
        
    # Ensure model's generate_essay and the fallback were called
    mock_model_instance.generate_essay.assert_called_once()
    mock_model_instance.generate_fallback_essay.assert_called_once() # Check fallback was attempted
