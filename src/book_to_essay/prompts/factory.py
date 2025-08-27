"""Factory for creating prompt templates based on model type."""
from typing import Dict, Type

from src.book_to_essay.prompts.base import PromptTemplate
from src.book_to_essay.prompts.deepseek import DeepSeekPromptTemplate
from src.book_to_essay.prompts.mistral import MistralPromptTemplate


class PromptTemplateFactory:
    """Factory for creating prompt templates based on model type."""
    
    # Registry of prompt templates keyed by model ID prefix
    _templates: Dict[str, Type[PromptTemplate]] = {
        "deepseek": DeepSeekPromptTemplate,
        "mistral": MistralPromptTemplate,
    }
    
    @classmethod
    def create(cls, model_id: str) -> PromptTemplate:
        """Create a prompt template for the given model ID.
        
        Args:
            model_id: The model ID, e.g., 'deepseek-ai/deepseek-llm-7b-base'
            
        Returns:
            A prompt template instance
        """
        # Default to DeepSeek if no match is found
        template_cls = cls._templates.get("deepseek")
        
        # Try to find a more specific template
        for prefix, template in cls._templates.items():
            if prefix in model_id.lower():
                template_cls = template
                break
        
        return template_cls()
    
    @classmethod
    def register(cls, model_prefix: str, template_cls: Type[PromptTemplate]) -> None:
        """Register a new prompt template.
        
        Args:
            model_prefix: The model ID prefix to match
            template_cls: The prompt template class
        """
        cls._templates[model_prefix.lower()] = template_cls
