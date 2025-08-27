import pytest
from src.book_to_essay.utils import truncate_text, filter_analysis, prepare_citations, format_essay_from_analyses

def test_truncate_text_shorter_than_target():
    text = "one two three"
    assert truncate_text(text, 10) == text

def test_truncate_text_exact_target():
    text = "one two three four"
    assert truncate_text(text, 4) == text

def test_truncate_text_minor_truncation():
    text = "a b c d"
    assert truncate_text(text, 3) == "a b c"

def test_truncate_text_significant_truncation():
    text = " ".join([str(i) for i in range(30)])
    result = truncate_text(text, 10)
    assert result.startswith("0 1 2 3 4 5")
    assert "[...]" in result
    assert result.endswith("26 27 28 29")

def test_filter_analysis_removes_instructions():
    analysis = """INSTRUCTIONS: Do not include
1. Extract key points
Some real analysis
Focus ONLY on the topic
"""
    filtered = filter_analysis(analysis)
    assert "INSTRUCTIONS" not in filtered
    assert "Extract key" not in filtered
    assert "Focus ONLY" not in filtered
    assert "Some real analysis" in filtered

def test_prepare_citations_none():
    mla, txt = prepare_citations(None)
    assert mla is None and txt == ""

def test_prepare_citations_empty():
    mla, txt = prepare_citations([])
    assert mla is None and txt == ""

def test_prepare_citations_minimal():
    sources = [{"author": "A", "title": "T"}]
    mla, txt = prepare_citations(sources)
    assert isinstance(mla, list) and len(mla) == 1
    assert "A. T." in mla[0]
    assert txt.strip() == mla[0]

def test_format_essay_from_analyses_basic():
    analyses = ["Intro", "Body"]
    citations = "Cite1\nCite2"
    essay = format_essay_from_analyses(analyses, citations, 500, "Academic")
    assert "Intro" in essay and "Body" in essay
    assert "Works Cited" in essay
    assert "Cite1" in essay and "Cite2" in essay

def test_format_essay_from_analyses_no_citations():
    analyses = ["A", "B"]
    essay = format_essay_from_analyses(analyses, "", 300, "Argumentative")
    assert "Works Cited" not in essay
    assert essay.startswith("A\n\nB")
