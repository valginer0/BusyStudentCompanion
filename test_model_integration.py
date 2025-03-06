#!/usr/bin/env python
"""Minimal test for model integration with new prompt templates."""

import sys
import logging
from src.book_to_essay.model_handler import DeepSeekHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generate_fallback_essay():
    """Test the fallback essay generation with the new prompt template system."""
    handler = DeepSeekHandler()
    
    # Test the fallback essay generation
    topic = "love and sacrifice in Romeo and Juliet"
    style = "analytical"
    word_limit = 200
    
    print(f"Generating a {word_limit}-word {style} essay on: {topic}")
    print("=" * 50)
    
    # Generate essay
    essay = handler._generate_fallback_essay(topic, style, word_limit)
    
    print(f"Generated essay ({len(essay.split())} words):")
    print("-" * 50)
    print(essay)
    print("-" * 50)
    
    return essay

if __name__ == "__main__":
    print("Starting minimal model integration test...")
    try:
        essay = test_generate_fallback_essay()
        print(f"\nTest completed successfully! Generated {len(essay.split())} words.")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        sys.exit(1)
