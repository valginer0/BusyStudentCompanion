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
    """Test that errors from the model handler are propagated."""
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

    # Assert that calling generate_essay raises the expected RuntimeError
    with pytest.raises(RuntimeError, match="Model generation failed"):
        generator.generate_essay("Test Topic", word_limit=100, style="academic")

    mock_handler_instance.generate_essay.assert_called_once()
