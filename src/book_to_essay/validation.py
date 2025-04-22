"""Validation utilities for essay generation inputs."""
import os
import re

SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.epub'}
ALLOWED_STYLES = {'academic', 'analytical', 'argumentative', 'expository'}
WORD_COUNT_RANGE = (100, 5000)


def validate_word_count(word_count: int):
    min_wc, max_wc = WORD_COUNT_RANGE
    if not (min_wc <= word_count <= max_wc):
        raise ValueError(f"Word count must be between {min_wc} and {max_wc} (got {word_count})")


def validate_file_extension(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")


def validate_style(style: str):
    if style.lower() not in ALLOWED_STYLES:
        raise ValueError(f"Invalid style: {style}. Allowed: {', '.join(ALLOWED_STYLES)}")


def validate_filename_for_citation(filename: str):
    # Should match 'Author - Title.ext'
    base = os.path.splitext(os.path.basename(filename))[0]
    # Only warn if the pattern does not match, do not raise
    if not re.match(r"^.+\s-\s.+$", base):
        import warnings
        warnings.warn(
            f"Filename '{filename}' does not match 'Author - Title.ext'. Citation extraction may be less accurate.",
            UserWarning
        )
    # Optionally, return None or metadata if you want to extract when possible
    # else: extract author/title if needed
