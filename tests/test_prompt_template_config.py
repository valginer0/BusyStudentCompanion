from src.book_to_essay.prompts.deepseek import DeepSeekPromptTemplate
from src.book_to_essay.prompts.config import PromptConfig

def test_format_chunk_analysis_prompt_accepts_config():
    template = DeepSeekPromptTemplate()
    config = PromptConfig(
        analysis_text="This is a test chunk.",
        topic="test topic",
        style="analytical",
        word_limit=123,
        source_info="Test Source"
    )
    prompt = template.format_chunk_analysis_prompt(config)
    assert isinstance(prompt, str)
    assert "test chunk" in prompt or "test topic" in prompt
