def truncate_text(text: str, target_words: int) -> str:
    """Truncate text to a target word count while preserving paragraph structure.
    Args:
        text: The text to truncate
        target_words: Target number of words
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= target_words:
        return text
    # If we need significant truncation, preserve first 60% and last 40%
    if len(words) > target_words * 1.5:
        first_chunk_size = int(target_words * 0.6)
        last_chunk_size = target_words - first_chunk_size
        first_chunk = ' '.join(words[:first_chunk_size])
        last_chunk = ' '.join(words[-last_chunk_size:])
        return f"{first_chunk}\n\n[...]\n\n{last_chunk}"
    # For minor truncation, just take the first N words
    return ' '.join(words[:target_words])

def filter_analysis(analysis: str) -> str:
    """
    Remove instruction lines from chunk analysis.
    Args:
        analysis: The chunk analysis string.
    Returns:
        The filtered analysis string with instructions removed.
    """
    instruction_keywords = [
        "INSTRUCTIONS:", "Extract key", "Identify character", "Note literary",
        "Focus ONLY", "Format your", "Source Materials:", "TEXT EXCERPT:",
        "social media", "data analysis", "YOUR ANALYSIS",
        "do not repeat these instructions", "do not include these instructions",
        "start directly with", "ESSAY", "do not"
    ]
    lines = analysis.split('\n')
    filtered = []
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in instruction_keywords):
            continue
        import re
        if re.match(r'^\d+\.\s+(Extract|Identify|Note|Focus|Format)', line.strip()):
            continue
        filtered.append(line)
    return '\n'.join(filtered)

def prepare_citations(sources):
    """
    Prepare MLA citations and citation text from a list of sources.
    Args:
        sources: List of dicts with citation info (author, title, year, etc.)
    Returns:
        Tuple (mla_citations: List[str] or None, citations_text: str)
    """
    if not sources:
        return None, ""
    mla_citations = []
    for source in sources:
        author = source.get("author", "Unknown Author")
        title = source.get("title", "Unknown Title")
        year = source.get("year", "n.d.")
        publisher = source.get("publisher", "")
        citation = f"{author}. {title}. {publisher}, {year}."
        mla_citations.append(citation)
    citations_text = "\n".join(mla_citations)
    return mla_citations, citations_text

def format_essay_from_analyses(analyses, citations_text, word_limit, style):
    """
    Format the essay body from analyses and citations.
    Args:
        analyses: List of analysis strings
        citations_text: Works Cited string
        word_limit: Target word count
        style: Essay style
    Returns:
        Formatted essay string
    """
    essay_body = "\n\n".join(analyses)
    essay = essay_body
    if citations_text and "Works Cited" not in essay:
        essay += f"\n\nWorks Cited:\n{citations_text}"
    return essay
