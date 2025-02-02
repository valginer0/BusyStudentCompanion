"""Tests for the cache manager."""
import os
import pytest
from pathlib import Path
from src.book_to_essay.cache_manager import CacheManager

def create_test_file(path: str, content: str = "test content"):
    """Create a test file with content."""
    with open(path, 'w') as f:
        f.write(content)

def test_cache_initialization(temp_cache_dir):
    """Test cache initialization."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    assert cache.cache_dir == Path(temp_cache_dir)

def test_cache_content(temp_cache_dir):
    """Test caching and retrieving content."""
    cache = CacheManager(cache_dir=temp_cache_dir)

    # Create a test file
    file_path = os.path.join(temp_cache_dir, "test_book.pdf")
    create_test_file(file_path)

    # Test data
    content = {"content": "Sample book content", "metadata": {"pages": 10}}

    # Cache the content
    cache.cache_content(file_path, content)

    # Verify content is cached
    cached_content = cache.get_cached_content(file_path)
    assert cached_content is not None
    assert cached_content["content"] == content["content"]

def test_cache_missing_content(temp_cache_dir):
    """Test retrieving non-existent cached content."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    
    # Create a test file but don't cache it
    file_path = os.path.join(temp_cache_dir, "test_book.pdf")
    create_test_file(file_path)
    
    assert cache.get_cached_content(file_path) is None

def test_cache_cleanup(temp_cache_dir):
    """Test cleaning up cache files."""
    cache = CacheManager(cache_dir=temp_cache_dir)

    # Create and cache content
    file_path = os.path.join(temp_cache_dir, "test_book.pdf")
    create_test_file(file_path)
    content = {"content": "Sample content"}
    cache.cache_content(file_path, content)

    # Verify cache file exists
    content_cache_dir = Path(temp_cache_dir) / "content"
    cache_files = list(content_cache_dir.glob("*.pkl"))
    assert len(cache_files) > 0

    # Remove cache files manually
    for cache_file in cache_files:
        cache_file.unlink()

    # Verify cache files are gone
    cache_files = list(content_cache_dir.glob("*.pkl"))
    assert len(cache_files) == 0
