"""Tests for the cache manager module."""
import pytest
from pathlib import Path
from src.book_to_essay.cache_manager import CacheManager

def test_cache_initialization(temp_cache_dir):
    """Test cache manager initialization."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    assert Path(temp_cache_dir).exists()
    assert cache.cache_dir == Path(temp_cache_dir)

def test_cache_content(temp_cache_dir):
    """Test caching and retrieving content."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    
    # Test data
    file_path = "test_book.pdf"
    content = {"content": "Sample book content", "metadata": {"pages": 10}}
    
    # Cache the content
    cache.cache_content(file_path, content)
    
    # Retrieve the content
    cached = cache.get_cached_content(file_path)
    assert cached == content

def test_cache_missing_content(temp_cache_dir):
    """Test retrieving non-existent cached content."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    assert cache.get_cached_content("nonexistent.pdf") is None

def test_cache_invalidation(temp_cache_dir):
    """Test cache invalidation."""
    cache = CacheManager(cache_dir=temp_cache_dir)
    
    # Cache some content
    file_path = "test_book.pdf"
    content = {"content": "Sample content"}
    cache.cache_content(file_path, content)
    
    # Invalidate the cache
    cache.invalidate_cache(file_path)
    assert cache.get_cached_content(file_path) is None
