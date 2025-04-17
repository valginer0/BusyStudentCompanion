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
