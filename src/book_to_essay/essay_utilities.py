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
        sources: List of source dicts (must contain 'name' and 'hash' keys)
    Returns:
        A unique string cache key
    """
    # Use both name and hash for each source
    source_keys = ','.join(sorted([f"{s['name']}:{s.get('hash','')}" for s in sources]))
    key_data = f"{prompt}|{word_limit}|{style}|{source_keys}"
    return hashlib.md5(key_data.encode()).hexdigest()
