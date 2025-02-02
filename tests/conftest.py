"""Test configuration and fixtures."""
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return "This is a sample book content for testing purposes."

@pytest.fixture
def mock_deepseek_response():
    """Mock response from DeepSeek API."""
    return {
        "essay": "This is a generated essay about the sample book.",
        "summary": "Brief summary of the content.",
        "analysis": "Detailed analysis of the themes and characters."
    }
