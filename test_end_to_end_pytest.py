import shutil
from pathlib import Path
import pytest
from src.book_to_essay.ai_book_to_essay_generator import AIBookEssayGenerator

@pytest.mark.slow
def test_end_to_end_generation(tmp_path):
    # Copy excerpt file to a valid Author - Title filename
    project_root = Path(__file__).parent
    src = project_root / "test_data" / "RomeoAndJulietExcerpt.txt"
    assert src.exists(), f"Test data not found at {src}"
    dest = tmp_path / "Shakespeare - Romeo and Juliet.txt"
    shutil.copy(src, dest)

    # Initialize generator and load file
    gen = AIBookEssayGenerator()
    gen.load_txt_file(str(dest))
    assert gen.content, "No content loaded"

    # Generate essay
    essay = gen.generate_essay(prompt="Theme of love", word_limit=100, style="analytical")
    assert isinstance(essay, str)
    assert "Works Cited" in essay, "Missing Works Cited"
    word_count = len(essay.split())
    assert 50 <= word_count <= 200, f"Word count out of bounds: {word_count}"
