import pytest
from src.book_to_essay.essay_utilities import get_essay_cache_key

def test_get_essay_cache_key_basic():
    sources = [
        {"name": "Book1.txt"},
        {"name": "Book2.pdf"}
    ]
    key1 = get_essay_cache_key("Prompt", 500, "academic", sources)
    key2 = get_essay_cache_key("Prompt", 500, "academic", sources)
    assert key1 == key2


def test_get_essay_cache_key_different_inputs():
    sources = [{"name": "Book1.txt"}]
    key1 = get_essay_cache_key("PromptA", 500, "academic", sources)
    key2 = get_essay_cache_key("PromptB", 500, "academic", sources)
    key3 = get_essay_cache_key("PromptA", 400, "academic", sources)
    key4 = get_essay_cache_key("PromptA", 500, "argumentative", sources)
    assert len({key1, key2, key3, key4}) == 4


def test_get_essay_cache_key_source_order_independence():
    sources1 = [
        {"name": "BookA.txt"},
        {"name": "BookB.pdf"}
    ]
    sources2 = [
        {"name": "BookB.pdf"},
        {"name": "BookA.txt"}
    ]
    key1 = get_essay_cache_key("Prompt", 1000, "expository", sources1)
    key2 = get_essay_cache_key("Prompt", 1000, "expository", sources2)
    assert key1 == key2


def test_get_essay_cache_key_empty_sources():
    key = get_essay_cache_key("Prompt", 500, "academic", [])
    assert isinstance(key, str)
    assert len(key) == 32  # md5 hex length
