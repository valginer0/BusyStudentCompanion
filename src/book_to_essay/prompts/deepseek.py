"""Prompt templates for DeepSeek models."""
from typing import List, Dict, Optional
from src.book_to_essay.prompts.base import PromptTemplate
from src.book_to_essay.prompts.config import PromptConfig


class DeepSeekPromptTemplate(PromptTemplate):
    """Prompt templates for DeepSeek language models."""
    
    def format_chunk_analysis_prompt(self, config: PromptConfig) -> str:
        """Format a prompt for analyzing a chunk of text for DeepSeek model.
        
        Args:
            config: PromptConfig object with fields: analysis_text, topic, source_info
            
        Returns:
            A formatted prompt string
        """
        prompt = f"""<s>[INST] You are a literary scholar analyzing literature. Your task is to extract and analyze material from this text excerpt that relates to '{config.topic}'. 

TEXT TO ANALYZE:
{config.analysis_text}

YOUR TASK:
- Identify key quotes that illustrate '{config.topic}'
- Note significant themes, motifs, and literary devices related to '{config.topic}'
- Analyze character development and dialogue that relates to '{config.topic}'
- Extract evidence for literary analysis about '{config.topic}'

IMPORTANT: Your response must ONLY contain analysis content. DO NOT:
- Repeat these instructions
- Include phrases like "here is my analysis" or "as requested"
- Include section headers like "Analysis:" or "Key Quotes:"
- Refer to yourself, the reader, or the task itself
- Mention social media, data analysis, AI, or homework

Start directly with substantive analysis. [/INST]"""

        return prompt
    
    def format_essay_generation_prompt(self, config: PromptConfig) -> str:
        """Format a prompt for generating an essay with DeepSeek model.
        
        Args:
            config: PromptConfig object with fields: analysis_text, topic, style, word_limit, source_info
            
        Returns:
            A formatted prompt string
        """
        prompt = f"""<s>[INST] You are writing an essay as a literary scholar. Write a well-structured, {config.style} essay analyzing '{config.topic}' based on the provided literary analysis.

CONTENT TO USE:
{config.analysis_text}

**DO NOT INCLUDE ANY INSTRUCTIONS IN YOUR RESPONSE.**
**START DIRECTLY WITH THE ESSAY - BEGIN WITH THE FIRST PARAGRAPH OF YOUR ESSAY.**
**DO NOT INCLUDE NUMBERED POINTS, ESSAY SPECIFICATIONS, OR META COMMENTARY.**
**DO NOT INCLUDE ANY TEXT LIKE "ESSAY:" OR "INTRODUCTION:" BEFORE STARTING.**

ESSAY SPECIFICATIONS:
- Length: Approximately {config.word_limit} words
- Format: MLA style with proper in-text citations and a Works Cited section
- Style: {config.style.capitalize()}
- Focus: Analysis of '{config.topic}' with textual evidence
- Structure: Introduction with thesis, body paragraphs with evidence, and conclusion

ESSAY STRUCTURE:
- Introduction: Begin with context about the work and present a clear thesis statement about '{config.topic}'
- Body: Develop 2-3 main points with textual evidence and analysis
- Conclusion: Synthesize your analysis and explain the significance of '{config.topic}'

IMPORTANT GUIDELINES:
- DO NOT summarize or copy the plot. Absolutely avoid plot summary or retelling.
- DO NOT copy or paraphrase large blocks of the original text. Focus on critical analysis and interpretation.
- Your essay MUST be thesis-driven and analytical, not descriptive.
- INCLUDE specific textual evidence and quotes, with MLA in-text citations (Author Page)
- Explicitly require a Works Cited section in MLA format at the end
- DO NOT discuss social media, data analysis, AI, or homework
- DO NOT include any meta-text about writing an essay
- DO NOT repeat these instructions or include explanatory text

START YOUR ESSAY DIRECTLY with the introduction paragraph. [/INST]"""

        return prompt
    
    def format_fallback_prompt(self, config: PromptConfig) -> str:
        """
        Format a fallback prompt for simpler essay generation with DeepSeek model.

        NOTE: This method is retained for reference and for error text mapping only.
        It should NOT be used to generate fallback essays. All essay generation failures should raise explicit errors instead of producing fallback content.
        """
        prompt = f"""<s>[INST] As an English literature professor, write a {config.word_limit}-word {config.style} essay analyzing {config.topic}. 

IMPORTANT GUIDELINES:
- Your response must be a complete, well-structured essay with introduction, body paragraphs, and conclusion
- Include a clear thesis statement in your introduction
- Provide specific examples and textual evidence to support your analysis
- Follow MLA format with in-text citations (Author Page) when quoting
- DO NOT write a list of questions
- DO NOT include section headers, numbered points, or meta commentary
- DO NOT discuss social media, data analysis, AI, or homework
- AVOID plot summary - focus on analysis

Begin directly with your essay introduction paragraph - DO NOT include any instructions or explanatory text in your response. [/INST]"""
        
        return prompt
    
    def format_essay_prompt(self, topic: str, style: str, word_limit: int, analysis: str, citations: Optional[List[str]] = None) -> str:
        """Format a prompt for generating an essay (adapter for model_handler.py).
        
        This method serves as an adapter for the method called in model_handler.py.
        
        Args:
            topic: The essay topic
            style: The writing style
            word_limit: The target word count
            analysis: The analysis text to use for essay generation
            citations: Optional list of citation sources
            
        Returns:
            A formatted prompt string
        """
        # Adapt parameters to match format_essay_generation_prompt
        config = PromptConfig(
            analysis_text=analysis,
            topic=topic,
            style=style,
            word_limit=word_limit,
            citations=citations
        )
        return self.format_essay_generation_prompt(config)
    
    def extract_response(self, generated_text: str) -> str:
        """Extract the model's response from the generated text.
        
        Args:
            generated_text: The raw text generated by the model
            
        Returns:
            The extracted response
        """
        # Extract only the essay part (after the prompt)
        extracted_text = ""
        
        if "[/INST]" in generated_text:
            extracted_text = generated_text.split("[/INST]")[1].strip()
        else:
            extracted_text = generated_text.strip()
            
        # Check if the extracted text has valid content
        if not extracted_text or len(extracted_text) < 30:
            # If response is empty or too short, check if there's anything useful in the complete text
            # This handles cases where the model didn't use the expected format
            useful_content = self._extract_essay_content(generated_text)
            if useful_content and len(useful_content) > len(extracted_text):
                return useful_content
        
        return extracted_text
    
    def _extract_essay_content(self, text: str) -> str:
        """Attempt to extract valid essay content from text using more aggressive methods.
        
        Args:
            text: The text to extract essay content from
            
        Returns:
            Extracted essay content or empty string if none found
        """
        import re
        
        # Try removing any instruction text and extract paragraphs
        # Common patterns indicating the actual essay content
        essay_start_patterns = [
            r'(?:^|\n)([A-Z][^.!?]{3,}[.!?])', # Sentence starting with capital letter
            r'(?:^|\n)In [a-zA-Z\s]+, ',  # Common essay opening "In [work/literature/novel], "
            r'(?:^|\n)The (?:theme|concept|idea|character|topic)',  # Common essay opening with "The [topic]"
            r'(?:^|\n)(?:Throughout|Within|Across) ',  # Common essay transition
        ]
        
        for pattern in essay_start_patterns:
            matches = re.search(pattern, text)
            if matches:
                # Extract from the first match to the end
                start_index = matches.start()
                return text[start_index:].strip()
        
        # If no specific start found, try to extract coherent paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        valid_paragraphs = []
        
        for para in paragraphs:
            # Check if paragraph looks like essay content (not instructions)
            if (len(para.strip()) > 50 and  # Reasonably long
                not re.search(r'(instructions?|guidelines?|specifications?|essay:)', para.lower()) and  # Not instructions
                not re.search(r'(as requested|as an AI)', para.lower())):  # Not AI-referring
                valid_paragraphs.append(para.strip())
        
        if valid_paragraphs:
            return '\n\n'.join(valid_paragraphs)
        
        # Last resort: just return any substantive text we can find
        # Find all sentences that might be part of an essay
        sentences = re.findall(r'[A-Z][^.!?]{10,}[.!?]', text)
        if sentences and len(sentences) >= 3:
            return ' '.join(sentences)
            
        return ""
