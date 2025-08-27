"""Unit tests for error_utils.py error handling utilities."""
import pytest
from src.book_to_essay.error_utils import log_and_raise

class DummyLogger:
    def __init__(self):
        self.last_msg = None
    def error(self, msg):
        self.last_msg = msg


def test_log_and_raise_raises(monkeypatch):
    # Patch logger to capture error messages
    import src.book_to_essay.error_utils as err_utils
    dummy_logger = DummyLogger()
    monkeypatch.setattr(err_utils, "logger", dummy_logger)
    
    # Should raise ValueError and log
    with pytest.raises(ValueError, match="Test error message"):
        log_and_raise("Test error message")
    assert "Test error message" in dummy_logger.last_msg

    # Should raise RuntimeError and log, with exception chaining
    try:
        try:
            raise KeyError("Inner error")
        except KeyError as e:
            log_and_raise("Outer error", e, RuntimeError)
    except RuntimeError as exc:
        assert "Outer error" in str(exc)
        assert "Inner error" in str(exc)
        assert "Outer error" in dummy_logger.last_msg
        assert "Inner error" in dummy_logger.last_msg
