"""Tests for the cache manager."""
import os
import pytest
from pathlib import Path
from src.book_to_essay.cache_manager import CacheManager
from datetime import datetime, timedelta

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

def test_clean_cache(mocker, temp_cache_dir):
    """Test the clear_expired_cache method removes old files based on hardcoded limits."""
    # --- Setup ---
    manager = CacheManager(cache_dir=temp_cache_dir) 
    now = datetime.now()
    one_day_ago = now - timedelta(days=1, seconds=1) # Expired for model
    two_days_ago = now - timedelta(days=2) # Expired for model
    seven_days_ago = now - timedelta(days=7, seconds=1) # Expired for content
    eight_days_ago = now - timedelta(days=8) # Expired for content
    half_day_ago = now - timedelta(hours=12) # Not expired
    six_days_ago = now - timedelta(days=6) # Not expired for content

    # Mock metadata and create dummy files
    content_dir = manager.content_cache_dir
    model_dir = manager.model_cache_dir

    mock_metadata = {
        "content": {
            "file_path_old": {"hash": "hash_content_old", "timestamp": eight_days_ago.isoformat()},
            "file_path_expired": {"hash": "hash_content_expired", "timestamp": seven_days_ago.isoformat()},
            "file_path_new": {"hash": "hash_content_new", "timestamp": six_days_ago.isoformat()}
        },
        "model": {
            "prompt_hash_old": {"timestamp": two_days_ago.isoformat()},
            "prompt_hash_expired": {"timestamp": one_day_ago.isoformat()},
            "prompt_hash_new": {"timestamp": half_day_ago.isoformat()}
        }
    }

    # Create corresponding dummy cache files
    content_files_to_create = {
        "hash_content_old": eight_days_ago,
        "hash_content_expired": seven_days_ago,
        "hash_content_new": six_days_ago
    }
    model_files_to_create = {
        "prompt_hash_old": two_days_ago,
        "prompt_hash_expired": one_day_ago,
        "prompt_hash_new": half_day_ago
    }

    for file_hash, dt in content_files_to_create.items():
        file_path = content_dir / f"{file_hash}.pkl"
        file_path.touch()
        timestamp = dt.timestamp()
        os.utime(file_path, (timestamp, timestamp))

    for prompt_hash, dt in model_files_to_create.items():
        file_path = model_dir / f"{prompt_hash}.pkl"
        file_path.touch()
        timestamp = dt.timestamp()
        os.utime(file_path, (timestamp, timestamp))

    # Write initial metadata
    manager.metadata = mock_metadata
    manager._save_metadata()

    # --- Action ---
    manager.clear_expired_cache()

    # --- Assertions ---
    # Reload metadata to check updates
    updated_metadata = manager._load_metadata()

    # Check content metadata
    assert "file_path_old" not in updated_metadata["content"]
    assert "file_path_expired" not in updated_metadata["content"]
    assert "file_path_new" in updated_metadata["content"]
    assert updated_metadata["content"]["file_path_new"]["hash"] == "hash_content_new"

    # Check model metadata
    assert "prompt_hash_old" not in updated_metadata["model"]
    assert "prompt_hash_expired" not in updated_metadata["model"]
    assert "prompt_hash_new" in updated_metadata["model"]
    assert updated_metadata["model"]["prompt_hash_new"]["timestamp"] == half_day_ago.isoformat()

    # Check files on disk
    assert not (content_dir / "hash_content_old.pkl").exists()
    assert not (content_dir / "hash_content_expired.pkl").exists()
    assert (content_dir / "hash_content_new.pkl").exists()

    assert not (model_dir / "prompt_hash_old.pkl").exists()
    assert not (model_dir / "prompt_hash_expired.pkl").exists()
    assert (model_dir / "prompt_hash_new.pkl").exists()
