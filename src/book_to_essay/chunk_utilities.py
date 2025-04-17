"""
Utility functions for chunk analysis and caching (cache key/path generation).
"""
import hashlib
from pathlib import Path

def get_chunk_cache_key(chunk: str, topic: str, style: str, word_limit: int) -> str:
    """
    Generate a cache key for a chunk analysis based on content and parameters.
    """
    key_data = f"{chunk}|{topic}|{style}|{word_limit}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_chunk_cache_path(cache_key: str, chunk_cache_dir: Path) -> Path:
    """
    Get the cache file path for a chunk analysis given a cache key and directory.
    """
    return chunk_cache_dir / f"{cache_key}.pkl"
