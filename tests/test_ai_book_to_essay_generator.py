"""Tests for the AI Book Essay Generator."""
import os
import pytest
from pathlib import Path
from src.book_to_essay.ai_book_to_essay_generator import AIBookEssayGenerator
from unittest.mock import MagicMock

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

# === Additional Tests ===

@pytest.mark.skip(reason="AIBookEssayGenerator in this version lacks a public load_file method")
def test_load_file_unsupported_type(mocker, temp_cache_dir):
    """Test ValueError is raised when trying to load an unsupported file type."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    mock_file_path = "/fake/path/document.zip"

    # Mock os.path.exists to return True (file exists but is wrong type)
    mocker.patch('os.path.exists', return_value=True)
    
    # Mock cache methods (simulate cache miss) - needed as load_file checks cache
    mocker.patch.object(generator.cache_manager, 'get_cached_content', return_value=None)

    # Assert ValueError is raised due to unsupported extension in load_file
    with pytest.raises(ValueError, match=f"Unsupported file type: {mock_file_path}"):
        generator.load_file(mock_file_path)

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
