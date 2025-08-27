"""Unit tests for validation utilities."""
import pytest
import warnings
from src.book_to_essay.validation import (
    validate_word_count,
    validate_file_extension,
    validate_style,
    validate_filename_for_citation,
)

# --- Word Count ---
def test_validate_word_count_valid():
    for wc in [100, 500, 5000]:
        validate_word_count(wc)

def test_validate_word_count_invalid():
    with pytest.raises(ValueError):
        validate_word_count(99)
    with pytest.raises(ValueError):
        validate_word_count(5001)

# --- File Extension ---
def test_validate_file_extension_valid():
    for fname in ["file.txt", "book.PDF", "novel.epub"]:
        validate_file_extension(fname)

def test_validate_file_extension_invalid():
    with pytest.raises(ValueError):
        validate_file_extension("essay.docx")
    with pytest.raises(ValueError):
        validate_file_extension("image.jpeg")

# --- Style ---
def test_validate_style_valid():
    for style in ["academic", "Analytical", "ARGUMENTATIVE", "expository"]:
        validate_style(style)

def test_validate_style_invalid():
    with pytest.raises(ValueError):
        validate_style("creative")
    with pytest.raises(ValueError):
        validate_style("")

# --- Filename for Citation ---
def test_validate_filename_for_citation_valid():
    import warnings
    valid_names = [
        "Shakespeare - Hamlet.txt",
        "Jane Austen - Pride and Prejudice.pdf",
        "Author - Title.epub",
    ]
    for fname in valid_names:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_filename_for_citation(fname)
            # Should not warn for valid names
            assert not w, f"Unexpected warning for valid filename: {fname}"


def test_validate_filename_for_citation_invalid():
    import warnings
    invalid_names = [
        "badname.txt",
        "NoDashHere.pdf",
        " - MissingAuthor.epub",
    ]
    for fname in invalid_names:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_filename_for_citation(fname)
            # Should warn for invalid names
            assert w, f"Expected warning for invalid filename: {fname}"
            assert issubclass(w[-1].category, UserWarning)
            assert "does not match 'Author - Title.ext'" in str(w[-1].message)
