"""Dataclass for essay prompt configuration."""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PromptConfig:
    analysis_text: Optional[str] = None
    topic: Optional[str] = None
    style: Optional[str] = None
    word_limit: Optional[int] = None
    source_info: Optional[str] = None
    citations: Optional[List[str]] = None
    # Add more fields as requirements grow
