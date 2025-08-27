import os
import sys
import subprocess
import pytest

@pytest.mark.slow
def test_end_to_end_generation():
    # Ensure excerpt file exists
    script_dir = os.getcwd()
    excerpt = os.path.join(script_dir, "test_data", "RomeoAndJulietExcerpt.txt")
    assert os.path.exists(excerpt), f"Test data not found at {excerpt}"
    print(f"[E2E] Excerpt exists at: {excerpt}")

    # Run the generation script
    script = os.path.join(script_dir, "test_essay_generation.py")
    print(f"[E2E] Running script: {script}")
    result = subprocess.run(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"[E2E] Script return code: {result.returncode}")
    print(f"[E2E] Script stdout (first 500 chars):\n{result.stdout[:500]}")
    print(f"[E2E] Script stderr (first 500 chars):\n{result.stderr[:500]}")
    assert result.returncode == 0, f"Script failed with: {result.stderr}"
    output = result.stdout

    # Validate output
    print("[E2E] Validating output for headers...")
    assert "GENERATED ESSAY" in output, "Generated essay header missing"
    assert "Works Cited" in output, "Works Cited section missing"

    # Check word count
    word_count = len(output.split())
    print(f"[E2E] Word count: {word_count}")
    assert 100 <= word_count <= 5000, f"Word count {word_count} out of acceptable range"
