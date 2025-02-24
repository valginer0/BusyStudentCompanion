"""Test quantization configuration and logging."""
import logging
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import torch
from src.book_to_essay.config import QuantizationConfig
from src.book_to_essay.model_handler import DeepSeekHandler

class TestQuantization(unittest.TestCase):
    """Test cases for model quantization configuration and logging."""

    def setUp(self):
        # Configure logging to capture log messages
        self.log_messages = []
        
        # Create a custom handler to capture log messages
        class TestLogHandler(logging.Handler):
            def __init__(self, messages):
                super().__init__()
                self.messages = messages

            def emit(self, record):
                self.messages.append(record.getMessage())
                
        # Reset logging configuration
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        
        # Set up our test handler
        self.test_handler = TestLogHandler(self.log_messages)
        logging.getLogger().addHandler(self.test_handler)
        logging.getLogger().setLevel(logging.INFO)

    def test_environment_detection_logging(self):
        """Test that environment detection is properly logged."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('src.book_to_essay.config.HAS_BITSANDBYTES', True):
                # Reset the config module to trigger environment detection
                import importlib
                import src.book_to_essay.config
                importlib.reload(src.book_to_essay.config)
                
                # Check if environment detection was logged
                env_logs = [msg for msg in self.log_messages if "Environment detected" in msg]
                self.assertTrue(any(env_logs))
                self.assertIn("GPU=True", env_logs[0])
                self.assertIn("BitsAndBytes=True", env_logs[0])

    def test_gpu_quantization_config(self):
        """Test 4-bit quantization config when GPU and bitsandbytes are available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('src.book_to_essay.config.HAS_BITSANDBYTES', True):
                with patch('importlib.metadata.version', return_value="0.39.0"):
                    config = QuantizationConfig.get_config()
                    
                    # Check if correct quantization method was logged
                    quant_logs = [msg for msg in self.log_messages if "4-bit quantization" in msg]
                    self.assertTrue(any(quant_logs))
                    self.assertEqual(config['method'], '4bit')

    def test_cpu_quantization_config(self):
        """Test 8-bit quantization config for CPU-only environment."""
        with patch('torch.cuda.is_available', return_value=False):
            config = QuantizationConfig.get_config()
            
            # Check if correct quantization method was logged
            quant_logs = [msg for msg in self.log_messages if "8-bit dynamic quantization" in msg]
            self.assertTrue(any(quant_logs))
            self.assertEqual(config['method'], '8bit_cpu')

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_model_loading_logs(self, mock_model, mock_tokenizer):
        """Test that model loading steps are properly logged."""
        # Mock the model and tokenizer to avoid actual loading
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Initialize model handler
        handler = DeepSeekHandler()
        
        # Check for expected log messages
        expected_logs = [
            "Loading model",
            "Loading tokenizer",
            "Loading model with quantization config",
            "Model loaded successfully"
        ]
        
        for expected in expected_logs:
            matching_logs = [msg for msg in self.log_messages if expected in msg]
            self.assertTrue(
                any(matching_logs), 
                f"Expected log message containing '{expected}' not found"
            )

    def test_error_logging(self):
        """Test that errors during model loading are properly logged."""
        with patch('transformers.AutoTokenizer.from_pretrained', 
                  side_effect=Exception("Test error")):
            with self.assertRaises(RuntimeError):
                handler = DeepSeekHandler()
            
            # Check if error was logged
            error_logs = [msg for msg in self.log_messages if "Error initializing model" in msg]
            self.assertTrue(any(error_logs))
            self.assertIn("Test error", error_logs[0])

    def tearDown(self):
        # Remove our test handler
        logging.getLogger().removeHandler(self.test_handler)

if __name__ == '__main__':
    unittest.main()
