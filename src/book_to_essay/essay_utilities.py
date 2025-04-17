"""
Utility functions for essay-level operations such as cache key generation.
"""
import hashlib
from typing import List, Dict

def get_essay_cache_key(prompt: str, word_limit: int, style: str, sources: List[Dict]) -> str:
    """
    Generate a cache key for an essay based on prompt, word limit, style, and sources.
    Args:
        prompt: The essay topic or prompt
        word_limit: Maximum word count for the essay
        style: Writing style
        sources: List of source dicts (must contain 'name' key)
    Returns:
        A unique string cache key
    """
    source_names = ','.join(sorted([s['name'] for s in sources]))
    key_data = f"{prompt}|{word_limit}|{style}|{source_names}"
    return hashlib.md5(key_data.encode()).hexdigest()
