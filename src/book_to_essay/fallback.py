"""Fallback tracking for essay generation."""

from enum import Enum, auto


class FallbackReason(Enum):
    """
    Enumeration of possible reasons for fallback essay generation.

    NOTE: FallbackReason is no longer used to trigger fallback essay generation.
    It is retained for reference, error code mapping, and for generating user-facing error messages.
    Do not use this enum to trigger fallback essays—raise explicit errors instead.
    """
    
    # Input-related reasons
    NO_TEXT_LOADED = auto()              # No text chunks loaded before generation
    EMPTY_CHUNKS = auto()                # Text chunks exist but are empty/invalid
    
    # Chunk analysis reasons
    CHUNK_ANALYSIS_ERROR = auto()        # Error during chunk analysis
    CHUNK_ANALYSIS_EMPTY = auto()        # Chunk analysis yielded empty/invalid results
    CHUNK_ANALYSIS_TIMEOUT = auto()      # Chunk analysis took too long
    
    # Essay generation reasons
    GENERATION_ERROR = auto()            # Error during generation phase
    EXTRACTION_ERROR = auto()            # Error extracting response from generation
    TOKEN_LIMIT_EXCEEDED = auto()        # Analysis text too large for context window
    
    # Post-processing reasons
    FILTERING_ERROR = auto()             # Error during essay filtering phase
    ESSAY_TOO_SHORT = auto()             # Essay too short after filtering
    ESSAY_NO_STRUCTURE = auto()          # Essay lacks proper paragraph structure
    ESSAY_ONLY_INSTRUCTIONS = auto()     # Essay contains only instruction text
    
    # Other reasons
    USER_REQUESTED = auto()              # User explicitly requested a fallback essay
    UNKNOWN = auto()                     # Unknown error occurred
    
    def __str__(self):
        """Convert enum name to readable format."""
        return self.name.replace('_', ' ').title()
