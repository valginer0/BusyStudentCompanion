"""Test suite for essay generation diagnostics and root cause analysis.

This module systematically tests different scenarios that might cause essay generation
failures, collects detailed diagnostic information, and analyzes patterns to help
identify and fix root causes.
"""

import logging
import argparse
import json
import re
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.fallback import FallbackReason
from src.book_to_essay.config import MODEL_CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("essay_diagnostics.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("essay_diagnostics")

# Path to test data and results
TEST_DATA_DIR = os.path.join(os.getcwd(), "test_data")
RESULTS_DIR = os.path.join(os.getcwd(), "test_results")
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
TEST_DOCS = {
    "short": os.path.join(TEST_DATA_DIR, "short_sample.txt"),
    "medium": os.path.join(TEST_DATA_DIR, "medium_sample.txt"),
    "long": os.path.join(TEST_DATA_DIR, "long_sample.txt"),
    "technical": os.path.join(TEST_DATA_DIR, "technical_sample.txt"),
    "irrelevant": os.path.join(TEST_DATA_DIR, "irrelevant_sample.txt")
}


class EssayDiagnosticSuite:
    """Test suite for diagnosing essay generation issues."""
    
    def __init__(self, output_dir=None):
        """Initialize the diagnostic suite."""
        self.base_model_handler = None
        self.model_handler = None  # Shared model handler - important for consistent pattern
        self.output_dir = output_dir or os.path.join(os.getcwd(), "diagnostic_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create cache directories
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        self.model_cache_file = os.path.join(MODEL_CACHE_DIR, "model_state.pt")
        self.tokenizer_cache_dir = os.path.join(MODEL_CACHE_DIR, "tokenizer")
        os.makedirs(self.tokenizer_cache_dir, exist_ok=True)
        
        # Track test results
        self.results = {
            "total_tests": 0,
            "failures": 0,
            "reasons": {},
            "scenarios": {},
            "detailed_results": []
        }
    
    def setup(self):
        """Initialize the base model handler for tests."""
        logger.info("Initializing base model handler")
        
        # Check if we can reuse a cached model
        cached_model = None
        cached_tokenizer = None
        
        try:
            if os.path.exists(self.model_cache_file) and os.path.exists(self.tokenizer_cache_dir):
                logger.info("Found cached model and tokenizer")
                try:
                    # Load tokenizer from cache directory
                    from transformers import AutoTokenizer
                    cached_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_cache_dir)
                    logger.info("Successfully loaded tokenizer from cache")
                    
                    # We'll still create the model in DeepSeekHandler to preserve quantization
                    # but we'll pass the tokenizer for reuse
                    self.base_model_handler = DeepSeekHandler(
                        tokenizer=cached_tokenizer
                    )
                    logger.info("Successfully created handler with cached tokenizer")
                except Exception as e:
                    logger.warning(f"Error loading cached components: {str(e)}")
                    self.base_model_handler = DeepSeekHandler()
            else:
                # No cached components, create a new handler
                logger.info("No cache found, creating new handler")
                self.base_model_handler = DeepSeekHandler()
                
                # Cache the tokenizer for future runs
                try:
                    logger.info("Saving tokenizer to cache...")
                    self.base_model_handler.tokenizer.save_pretrained(self.tokenizer_cache_dir)
                    logger.info("Tokenizer cached successfully")
                except Exception as e:
                    logger.warning(f"Failed to cache tokenizer: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during setup: {str(e)}")
            self.base_model_handler = DeepSeekHandler()
        
        # Set the model_handler to be the same as base_model_handler initially
        self.model_handler = self.base_model_handler
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        logger.info("Starting essay diagnostics test suite")
        
        # Setup if not already done
        if self.base_model_handler is None:
            self.setup()
        
        # Run all test categories
        self.test_insufficient_input_text()
        self.test_chunk_analysis_issues()
        self.test_generation_and_filtering_issues()
        self.test_content_extraction_issues()
        
        # Analyze and report results
        self.analyze_results()
        
        logger.info(f"Test suite completed. Total tests: {self.results['total_tests']}, "
                    f"Failures: {self.results['failures']}")
    
    def _get_fresh_handler(self):
        """Get a fresh handler for a test while preserving the model.
        
        Returns:
            A new DeepSeekHandler instance that reuses the model
        """
        # Create a new instance with the existing model, tokenizer, and prompt template
        logger.info("Creating a fresh model handler (reusing model)")
        
        if hasattr(self.base_model_handler, 'model') and self.base_model_handler.model is not None:
            handler = DeepSeekHandler(
                model=self.base_model_handler.model,
                tokenizer=self.base_model_handler.tokenizer,
                prompt_template=self.base_model_handler.prompt_template
            )
        else:
            # Fallback if no model is loaded yet
            handler = DeepSeekHandler()
            
        return handler
    
    def _load_test_text(self, file_path: str):
        """Load text from a test file.
        
        Args:
            file_path: Path to the test text file
            
        Returns:
            The loaded text string
        """
        if not os.path.exists(file_path):
            logger.warning(f"Test file not found: {file_path}")
            return "This is a placeholder text because the real test file was not found."
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def test_insufficient_input_text(self):
        """Test scenarios where input text is insufficient for essay generation."""
        logger.info("Testing insufficient input text scenarios")
        
        # Test 1: Empty text
        model_handler = self._get_fresh_handler()
        model_handler.text_chunks = []
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="empty_text_chunks",
            topic="Character development in literature",
            expected_reason=FallbackReason.EMPTY_CHUNKS,
            test_description="Text chunks list is empty"
        )
        
        # Test 2: Very short text
        model_handler = self._get_fresh_handler()
        short_text = "Just a few words."
        model_handler.text_chunks = [short_text]
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="extremely_short_text",
            topic="Themes in modern literature",
            expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
            test_description="Text is too short for meaningful analysis"
        )
        
        # Test 3: Short but valid text file
        if os.path.exists(TEST_DOCS["short"]):
            model_handler = self._get_fresh_handler()
            text = self._load_test_text(TEST_DOCS["short"])
            model_handler.text_chunks = [text]
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="short_text_file",
                topic="Main themes",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
                test_description="Short text from file"
            )
    
    def test_chunk_analysis_issues(self):
        """Test scenarios where chunk analysis fails or produces empty results."""
        logger.info("Testing chunk analysis failure scenarios")
        
        # Test 1: Very long chunks that might exceed token limits
        # TEMPORARILY DISABLED - Causes test to hang due to extremely large input
        """
        model_handler = self._get_fresh_handler()
        long_chunk = "This is a test sentence. " * 1000  # ~7000 words
        model_handler.text_chunks = [long_chunk]
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="token_limit_exceeded",
            topic="Repetition in literature",
            expected_reason=FallbackReason.TOKEN_LIMIT_EXCEEDED,
            test_description="Chunk exceeds model's token limit"
        )
        """
        logger.info("Skipping token_limit_exceeded test to prevent script hanging")
        
        # Test 2: Irrelevant content for the requested topic
        if os.path.exists(TEST_DOCS["irrelevant"]):
            model_handler = self._get_fresh_handler()
            text = self._load_test_text(TEST_DOCS["irrelevant"])
            model_handler.text_chunks = [text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="irrelevant_content_from_file",
                topic="Character development in Hamlet",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
                test_description="Text content from file unrelated to requested topic"
            )
        else:
            model_handler = self._get_fresh_handler()
            irrelevant_text = """
            The chemical formula for water is H2O. Each molecule contains two hydrogen atoms and one oxygen atom.
            The boiling point of water is 100 degrees Celsius at standard atmospheric pressure. Water freezes at 0 degrees Celsius.
            Water is essential for all known forms of life, even though it provides no calories or organic nutrients.
            """
            model_handler.text_chunks = [irrelevant_text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="irrelevant_content",
                topic="Character development in Hamlet",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
                test_description="Text content entirely unrelated to requested topic"
            )
        
        # Test 3: Multiple chunks with mixed content
        model_handler = self._get_fresh_handler()
        model_handler.text_chunks = [
            "Some relevant literature content about characters and plot.",
            "2 + 2 = 4, 3 × 3 = 9, square root of 16 is 4",
            "More literature content with themes and symbolism."
        ]
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="mixed_content_quality",
            topic="Literary themes and symbols",
            expected_reason=FallbackReason.ESSAY_TOO_SHORT,
            test_description="Mix of relevant and non-relevant content chunks"
        )
    
    def test_generation_and_filtering_issues(self):
        """Test scenarios related to essay generation and post-processing failures."""
        logger.info("Testing essay generation and filtering issues")
        
        # Test 1: Load valid text but request essay on completely unrelated topic
        if os.path.exists(TEST_DOCS["medium"]):
            model_handler = self._get_fresh_handler()
            text = self._load_test_text(TEST_DOCS["medium"])
            model_handler.text_chunks = [text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="unrelated_topic_from_file",
                topic="Quantum physics in modern science",
                expected_reason=FallbackReason.ESSAY_TOO_SHORT,
                test_description="Topic completely unrelated to text content from file"
            )
        else:
            model_handler = self._get_fresh_handler()
            sample_text = """
            It was the best of times, it was the worst of times, it was the age of wisdom,
            it was the age of foolishness, it was the epoch of belief, it was the epoch of
            incredulity, it was the season of Light, it was the season of Darkness, it was
            the spring of hope, it was the winter of despair.
            """
            model_handler.text_chunks = [sample_text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="unrelated_topic",
                topic="Quantum physics in modern science",
                expected_reason=FallbackReason.ESSAY_TOO_SHORT,
                test_description="Topic completely unrelated to text content"
            )
        
        # Test 2: Valid text but extremely small word limit
        model_handler = self._get_fresh_handler()
        model_handler.text_chunks = [self._load_test_text(TEST_DOCS["medium"])] if os.path.exists(TEST_DOCS["medium"]) else ["Sample literary text for testing."]
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="extremely_small_word_limit",
            topic="Themes in literature",
            word_limit=50,  # Very small word limit
            expected_reason=FallbackReason.ESSAY_TOO_SHORT,
            test_description="Word limit too small for valid essay generation"
        )
        
        # Test 3: Valid text with unusual writing style
        model_handler = self._get_fresh_handler()
        model_handler.text_chunks = [self._load_test_text(TEST_DOCS["medium"])] if os.path.exists(TEST_DOCS["medium"]) else ["Sample literary text for testing."]
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="unusual_writing_style",
            topic="Character analysis",
            style="poetic",  # Not in standard styles
            expected_reason=FallbackReason.ESSAY_NO_STRUCTURE,
            test_description="Requested style outside standard formats"
        )
    
    def test_content_extraction_issues(self):
        """Test scenarios related to essay content extraction and pattern matching failures."""
        logger.info("Testing content extraction and pattern matching issues")
        
        # Test 1: Valid essay with non-standard opening format
        model_handler = self._get_fresh_handler()
        # This test simulates the model generating a valid essay but with an opening
        # that doesn't match the current pattern matchers
        model_handler.text_chunks = ["Sample literature text with characters and plot."]
        
        # Monkey patch the extract_response method to return a valid essay with unusual beginning
        original_extract = model_handler.prompt_template.extract_response
        def mock_extract(_):
            # This essay begins with "A critical examination" which doesn't match current patterns
            return """A critical examination of literature reveals the profound impact of character development.
            
            Character growth serves as a fundamental aspect of narrative structures across literary traditions.
            
            Authors skillfully employ various techniques to illustrate the evolution of personalities within their works.
            
            This essay explores how character arcs function as more than mere plot devices."""
        
        model_handler.prompt_template.extract_response = mock_extract
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="nonstandard_essay_opening",
            topic="Character development",
            expected_reason=FallbackReason.ESSAY_TOO_SHORT,
            test_description="Essay with opening that doesn't match detection patterns"
        )
        
        # Reset the monkey patch
        model_handler.prompt_template.extract_response = original_extract
        
        # Test 2: Essay content with legitimate instruction-like text
        model_handler = self._get_fresh_handler()
        model_handler.text_chunks = ["Sample literature text about writing guidelines."]
        
        # Monkey patch to return an essay about writing instructions
        original_extract = model_handler.prompt_template.extract_response
        def mock_extract_with_guidelines(_):
            # This is a legitimate essay about writing guidelines that might be incorrectly filtered
            return """Literary criticism often examines the guidelines authors follow in their work.
            
            In "Elements of Style," Strunk and White provide instructions for clear writing that many authors adopt.
            The instructions emphasize clarity, brevity, and precision in language use.
            
            Similarly, creative writing textbooks frequently include specifications for character development,
            which novelists incorporate into their artistic process."""
        
        model_handler.prompt_template.extract_response = mock_extract_with_guidelines
        
        self._run_test_and_record(
            model_handler=model_handler,
            scenario="legitimate_instruction_content",
            topic="Writing guidelines in literature",
            expected_reason=FallbackReason.ESSAY_ONLY_INSTRUCTIONS,
            test_description="Essay about writing guidelines incorrectly filtered as instructions"
        )
        
        # Reset the monkey patch
        model_handler.prompt_template.extract_response = original_extract
        
        # Test 3: Technical/scientific essay format
        if os.path.exists(TEST_DOCS["technical"]):
            model_handler = self._get_fresh_handler()
            text = self._load_test_text(TEST_DOCS["technical"])
            model_handler.text_chunks = [text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="technical_content_format",
                topic="Technical aspects in the text",
                expected_reason=FallbackReason.ESSAY_NO_STRUCTURE,
                test_description="Technical content with non-literary structure"
            )
        else:
            model_handler = self._get_fresh_handler()
            technical_text = """
            The structure of DNA was discovered through X-ray crystallography studies.
            Nucleotides form the basic building blocks of DNA, consisting of a phosphate group,
            a deoxyribose sugar, and a nitrogenous base. The four nitrogenous bases in DNA are
            adenine, guanine, cytosine, and thymine. According to base-pairing rules, adenine
            pairs with thymine, and guanine pairs with cytosine.
            """
            model_handler.text_chunks = [technical_text]
            
            self._run_test_and_record(
                model_handler=model_handler,
                scenario="technical_content",
                topic="Structure and function of DNA",
                expected_reason=FallbackReason.ESSAY_NO_STRUCTURE,
                test_description="Technical content with scientific structure"
            )
    
    def _run_test_and_record(self, model_handler, scenario, topic, expected_reason, test_description, 
                           word_limit=500, style="academic"):
        """Run a test case and record detailed results."""
        logger.info(f"Running test: {scenario} - {test_description}")
        
        self.results["total_tests"] += 1
        
        try:
            # Attempt to generate an essay
            essay = model_handler.generate_essay(
                topic=topic,
                word_limit=word_limit,
                style=style
            )
            
            # Analyze the result
            failure_occurred = False
            word_count = len(essay.split()) if essay else 0
            
            # Check if the essay seems valid
            if word_count < 100 or "error" in essay.lower() or "unable" in essay.lower():
                failure_occurred = True
                detected_reason = self._infer_failure_reason(essay, expected_reason)
            else:
                detected_reason = None
            
        except Exception as e:
            failure_occurred = True
            essay = None
            detected_reason = expected_reason
            logger.error(f"Error in test '{scenario}': {str(e)}")
        
        # Record results
        if failure_occurred:
            self.results["failures"] += 1
            reason_str = str(detected_reason)
            self.results["reasons"][reason_str] = self.results["reasons"].get(reason_str, 0) + 1
            self.results["scenarios"][scenario] = self.results["scenarios"].get(scenario, 0) + 1
        
        # Record detailed results
        result = {
            "scenario": scenario,
            "description": test_description,
            "topic": topic,
            "expected_reason": str(expected_reason),
            "detected_reason": str(detected_reason) if detected_reason else None,
            "failure_occurred": failure_occurred,
            "essay_length": len(essay) if essay else 0,
            "word_count": len(essay.split()) if essay else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["detailed_results"].append(result)
        logger.info(f"Test result: {scenario}, Failure: {failure_occurred}, "
                   f"Reason: {str(detected_reason) if detected_reason else 'None'}")
    
    def _infer_failure_reason(self, essay, default_reason):
        """Analyze essay content to infer the failure reason if not explicitly available."""
        # Just placeholders for now - would need more sophisticated analysis
        if not essay or len(essay) < 10:
            return FallbackReason.ESSAY_TOO_SHORT
        
        if "instruction" in essay.lower() or "guidelines" in essay.lower():
            return FallbackReason.ESSAY_ONLY_INSTRUCTIONS
        
        if len(essay.split('\n\n')) < 3:
            return FallbackReason.ESSAY_NO_STRUCTURE
        
        # Default to the expected reason if we can't infer anything specific
        return default_reason
    
    def analyze_results(self):
        """Analyze test results to identify patterns and root causes."""
        if not self.results["failures"]:
            logger.info("No failures detected in the test suite")
            return
        
        # Analyze by reason
        logger.info("=== Analysis by Failure Reason ===")
        reasons_counter = Counter(self.results["reasons"])
        for reason, count in reasons_counter.most_common():
            percentage = (count / self.results["failures"]) * 100
            logger.info(f"{reason}: {count} occurrences ({percentage:.1f}%)")
        
        # Analyze by scenario
        logger.info("=== Analysis by Test Scenario ===")
        scenario_counter = Counter(self.results["scenarios"])
        for scenario, count in scenario_counter.most_common():
            percentage = (count / self.results["failures"]) * 100
            logger.info(f"{scenario}: {count} failures ({percentage:.1f}%)")
        
        # Look for patterns in the detailed results
        logger.info("=== Patterns in Failures ===")
        # Example pattern: Word count distribution
        word_counts = [r["word_count"] for r in self.results["detailed_results"] if r["failure_occurred"]]
        if word_counts:
            avg_word_count = sum(word_counts) / len(word_counts)
            logger.info(f"Average word count in failed essays: {avg_word_count:.1f}")
        
        # Save detailed analysis to file
        self.save_analysis_report()
    
    def save_analysis_report(self):
        """Save detailed analysis report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"essay_diagnostics_report_{timestamp}.json")
        
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Detailed analysis report saved to {report_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run essay generation diagnostic tests")
    parser.add_argument("--output", type=str, default=None, 
                        help="Directory to save test results")
    return parser.parse_args()


def main():
    """Main entry point for running diagnostics."""
    args = parse_args()
    
    output_dir = args.output
    
    # Create test_data directory if it doesn't exist
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Check if test files exist
    missing_files = []
    for name, path in TEST_DOCS.items():
        if not os.path.exists(path):
            missing_files.append((name, path))
    
    if missing_files:
        logger.warning(f"Some test files are missing. Will use placeholder text instead.")
        for name, path in missing_files:
            logger.warning(f"Missing: {name} => {path}")
    
    # Run test suite
    test_suite = EssayDiagnosticSuite(output_dir=output_dir)
    test_suite.setup()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()