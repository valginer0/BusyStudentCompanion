"""Unit tests for validation utilities."""
import pytest
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
    for fname in ["Author - Title.txt", "Doe - MyBook.pdf", "Smith - Epic.epub"]:
        validate_filename_for_citation(fname)

def test_validate_filename_for_citation_invalid():
    with pytest.raises(ValueError):
        validate_filename_for_citation("badname.txt")
    with pytest.raises(ValueError):
        validate_filename_for_citation("NoDashHere.pdf")
    with pytest.raises(ValueError):
        validate_filename_for_citation(" - MissingAuthor.epub")
