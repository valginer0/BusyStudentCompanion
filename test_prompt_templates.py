#!/usr/bin/env python
"""Test script for validating prompt templates."""

from src.book_to_essay.prompts.factory import PromptTemplateFactory
from src.book_to_essay.config import MODEL_NAME

def test_template(model_name):
    """Test prompt templates for a specific model."""
    print(f'\n{"="*20} Testing model: {model_name} {"="*20}')
    
    # Test template creation for model
    template = PromptTemplateFactory.create(model_name)
    print('Prompt template:', template.__class__.__name__)
    
    # Test prompt generation
    chunk_prompt = template.format_chunk_analysis_prompt(
        chunk="This is a test chunk of text to analyze.",
        topic="love and relationships"
    )
    print('\nExample chunk analysis prompt:')
    print('-' * 40)
    print(chunk_prompt[:200] + '...')
    print('-' * 40)
    
    # Test essay generation prompt
    essay_prompt = template.format_essay_generation_prompt(
        analysis_text="Key themes of love include sacrifice and devotion.",
        topic="love and relationships",
        style="analytical",
        word_limit=500
    )
    print('\nExample essay generation prompt:')
    print('-' * 40)
    print(essay_prompt[:200] + '...')
    print('-' * 40)
    
    # Test fallback prompt
    fallback_prompt = template.format_fallback_prompt(
        topic="love and relationships",
        style="analytical",
        word_limit=500
    )
    print('\nExample fallback prompt:')
    print('-' * 40)
    print(fallback_prompt)
    print('-' * 40)

def main():
    """Run validation tests for prompt templates."""
    print('Current configured model:', MODEL_NAME)
    
    # Test current model from config
    test_template(MODEL_NAME)
    
    # Test alternative model (Mistral)
    test_template("mistralai/Mistral-7B-Instruct-v0.1")
    
    print("\nAll prompt template tests completed successfully!")

if __name__ == "__main__":
    main()
