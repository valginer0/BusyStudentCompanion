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
