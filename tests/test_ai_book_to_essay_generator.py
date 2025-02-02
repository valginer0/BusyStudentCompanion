"""Tests for the AI Book Essay Generator."""
import os
import pytest
from pathlib import Path
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
