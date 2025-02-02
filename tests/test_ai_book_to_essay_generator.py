"""Tests for the AI book to essay generator module."""
import pytest
from pathlib import Path
from src.book_to_essay.ai_book_to_essay_generator import AIBookEssayGenerator

def test_generator_initialization():
    """Test generator initialization."""
    generator = AIBookEssayGenerator()
    assert generator.content == ""
    assert generator.sources == []
    assert generator._model is None

def test_process_pdf_content(mocker, sample_pdf_content, mock_deepseek_response):
    """Test processing PDF content."""
    generator = AIBookEssayGenerator()
    
    # Mock the model response
    mock_model = mocker.patch('src.book_to_essay.model_handler.DeepSeekHandler')
    mock_model.return_value.generate_essay.return_value = mock_deepseek_response
    
    # Mock PDF processing
    mocker.patch('fitz.open', return_value=mocker.MagicMock())
    
    result = generator.process_file("test.pdf", sample_pdf_content)
    assert result["essay"] == mock_deepseek_response["essay"]
    assert result["summary"] == mock_deepseek_response["summary"]

def test_content_caching(mocker, temp_cache_dir, sample_pdf_content):
    """Test content caching during processing."""
    generator = AIBookEssayGenerator()
    generator.cache_manager.cache_dir = Path(temp_cache_dir)
    
    # Process content first time
    mocker.patch('fitz.open', return_value=mocker.MagicMock())
    file_path = "test.pdf"
    
    generator.process_file(file_path, sample_pdf_content)
    
    # Verify content is cached
    cached_content = generator.cache_manager.get_cached_content(file_path)
    assert cached_content is not None
    assert cached_content["content"] == sample_pdf_content

@pytest.mark.asyncio
async def test_async_processing(mocker, sample_pdf_content, mock_deepseek_response):
    """Test asynchronous content processing."""
    generator = AIBookEssayGenerator()
    
    # Mock async operations
    mock_model = mocker.patch('src.book_to_essay.model_handler.DeepSeekHandler')
    mock_model.return_value.generate_essay_async.return_value = mock_deepseek_response
    
    result = await generator.process_file_async("test.pdf", sample_pdf_content)
    assert result["essay"] == mock_deepseek_response["essay"]
