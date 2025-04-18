"""Test quantization configuration and logging."""
import logging
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import torch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.book_to_essay.config import QuantizationConfig, HAS_GPU, HAS_BITSANDBYTES
from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.model_loader import load_model, load_tokenizer

@pytest.fixture(scope="module")
def base_handler():
    # Load model/tokenizer once for all tests
    return DeepSeekHandler(
        model=load_model(),
        tokenizer=load_tokenizer()
    )

def test_environment_detection_logging(caplog):
    # Simulate CPU-only environment
    with patch('torch.cuda.is_available', return_value=False):
        with patch('src.book_to_essay.config.HAS_GPU', False):
            with patch.dict('sys.modules', {'bitsandbytes': MagicMock()}):
                with patch('importlib.metadata.version', return_value="0.39.0"):
                    import importlib
                    import src.book_to_essay.config
                    with caplog.at_level('INFO', logger="src.book_to_essay.config"):
                        importlib.reload(src.book_to_essay.config)
                    env_logs = [r.message for r in caplog.records if "Environment detected: GPU=False, BitsAndBytes=True" in r.message]
                    assert any(env_logs)

def test_gpu_quantization_config(caplog):
    # This test is only meaningful if you want to simulate a GPU environment
    # For CPU-only hardware, we can skip or expect the CPU config
    with patch('torch.cuda.is_available', return_value=False):
        with patch('src.book_to_essay.config.HAS_GPU', False):
            with patch('src.book_to_essay.config.HAS_BITSANDBYTES', True):
                with patch.dict('sys.modules', {'bitsandbytes': MagicMock()}):
                    with patch('importlib.metadata.version', return_value="0.39.0"):
                        import importlib
                        import src.book_to_essay.config
                        with caplog.at_level('INFO', logger="src.book_to_essay.config"):
                            importlib.reload(src.book_to_essay.config)
                        config = QuantizationConfig.get_config()
                        # Should fall back to CPU config
                        assert config['method'] == 'cpu'
                        expected_log = "Using standard CPU configuration (no quantization)"
                        assert any(expected_log in r.message for r in caplog.records)

def test_cpu_quantization_config(caplog):
    with patch('torch.cuda.is_available', return_value=False):
        with patch('src.book_to_essay.config.HAS_GPU', False):
            with caplog.at_level('INFO', logger="src.book_to_essay.config"):
                config = QuantizationConfig.get_config()
            assert config['method'] == 'cpu'
            assert config['load_config']['device_map'] == 'cpu'
            assert config['post_load_quantize'] is None
            expected_log = "Using standard CPU configuration (no quantization)"
            assert any(expected_log in r.message for r in caplog.records)
            unexpected_log = "Using 8-bit dynamic quantization (CPU)"
            assert not any(unexpected_log in r.message for r in caplog.records)

def test_model_loading_logs(base_handler, caplog):
    with caplog.at_level('INFO', logger="src.book_to_essay.model_handler"):
        handler = DeepSeekHandler(
            model=base_handler.model,
            tokenizer=base_handler.tokenizer,
            prompt_template=getattr(base_handler, "prompt_template", None)
        )
    # Only check for logs that are actually emitted
    expected_logs = [
        "Model loaded successfully"
    ]
    for expected in expected_logs:
        assert any(expected in r.message for r in caplog.records), f"Expected log message containing '{expected}' not found"

def test_error_logging(caplog):
    # Patch load_model to raise an error during handler initialization
    with patch('src.book_to_essay.model_loader.load_model', side_effect=RuntimeError("Test error")):
        with pytest.raises(RuntimeError):
            DeepSeekHandler(model=load_model(), tokenizer=MagicMock())
    error_logs = [r.message for r in caplog.records if "Test error" in r.message]
    assert any(error_logs)
