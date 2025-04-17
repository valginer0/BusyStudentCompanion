"""Error handling utilities for model and essay generation."""
import logging
import traceback

logger = logging.getLogger(__name__)

def log_and_raise(msg: str, exc: Exception = None, error_type=ValueError):
    """
    Log the error message and raise the specified exception type.
    If exc is provided, include traceback info.
    """
    if exc:
        logger.error(f"{msg}: {exc}\n{traceback.format_exc()}")
        raise error_type(f"{msg}: {exc}") from exc
    else:
        logger.error(msg)
        raise error_type(msg)


def log_exception(msg: str, exc: Exception):
    """
    Log the error message and traceback for diagnostics, but do not raise.
    """
    logger.error(f"{msg}: {exc}\n{traceback.format_exc()}")
