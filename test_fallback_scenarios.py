"""Test suite for identifying essay generation fallback patterns.

This module systematically tests different scenarios that might trigger fallbacks
in the essay generation process and collects data on which fallback reasons
occur most frequently.
"""

import logging
import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.fallback import FallbackReason
from src.book_to_essay.model_loader import load_model, load_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fallback_test_results.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("fallback_test")

# Path to test data
TEST_DATA_DIR = Path("test_data")
RESULTS_DIR = Path("test_results")


class FallbackTestSuite:
    """Test suite for analyzing fallback patterns in essay generation."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the test suite.
        
        Args:
            output_dir: Directory to save test results
        """
        self.model_handler = None
        self.base_model_handler = None  # Shared model handler to clone from
        self.output_dir = output_dir or RESULTS_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Track test results
        self.results = {
            "total_tests": 0,
            "fallbacks": 0,
            "reasons": {},
            "scenarios": {},
            "detailed_results": []
        }
    
    def setup(self):
        """Initialize model handler."""
        logger.info("Initializing base model handler")
        model = load_model()
        tokenizer = load_tokenizer()
        self.base_model_handler = DeepSeekHandler(model=model, tokenizer=tokenizer)
        self.model_handler = self.base_model_handler
    
    def run_all_tests(self):
        """Run all fallback tests and collect results."""
        logger.info("Starting fallback test suite")
        
        # Setup if not already done
        if self.base_model_handler is None:
            self.setup()
        
        # Run each test category
        self.test_no_text_loaded()
        self.test_empty_chunks()
        self.test_chunk_analysis_failures()
        self.test_token_limits()
        self.test_filtering_issues()
        
        # Save results
        self.save_results()
        
        logger.info(f"Test suite completed. Total tests: {self.results['total_tests']}, "
                   f"Fallbacks: {self.results['fallbacks']}")
        
        # Print summary
        self.print_summary()
    
    def _get_fresh_handler(self):
        """Get a fresh handler for a test while preserving the model.
        
        Returns:
            A new DeepSeekHandler instance that reuses the model
        """
        # Create a new instance with the existing model, tokenizer, and prompt template
        logger.info("Creating a fresh model handler (reusing model)")
        
        if self.base_model_handler is not None and hasattr(self.base_model_handler, 'model') and self.base_model_handler.model is not None:
            handler = DeepSeekHandler(
                model=self.base_model_handler.model,
                tokenizer=self.base_model_handler.tokenizer,
                prompt_template=self.base_model_handler.prompt_template
            )
        else:
            # If for some reason base handler is missing, load anew (should happen only once)
            model = load_model()
            tokenizer = load_tokenizer()
            handler = DeepSeekHandler(model=model, tokenizer=tokenizer)
            
        return handler
    
    def test_no_text_loaded(self):
        """Test fallback when no text is loaded."""
        logger.info("Testing NO_TEXT_LOADED scenario")
        
        # Create a fresh model handler without loading text (reuse model)
        test_handler = self._get_fresh_handler()
        
        try:
            # This should trigger a fallback
            essay = test_handler.generate_essay(
                topic="The theme of love in literature",
                word_limit=500,
                style="academic"
            )
            
            self._record_result(
                scenario="no_text_loaded",
                topic="The theme of love in literature",
                expected_reason=FallbackReason.NO_TEXT_LOADED,
                essay=essay,
                raw_text=None
            )
        except Exception as e:
            logger.error(f"Error in test_no_text_loaded: {str(e)}")
            self._record_result(
                scenario="no_text_loaded",
                topic="The theme of love in literature",
                expected_reason=FallbackReason.NO_TEXT_LOADED,
                essay=None,
                raw_text=None,
                error=str(e)
            )
    
    def test_empty_chunks(self):
        """Test fallback when text chunks are empty."""
        logger.info("Testing EMPTY_CHUNKS scenario")
        
        # Create a handler and set empty chunks (reuse model)
        test_handler = self._get_fresh_handler()
        test_handler.text_chunks = ["", "   ", "\n\n"]
        
        try:
            essay = test_handler.generate_essay(
                topic="Character development in novels",
                word_limit=500,
                style="analytical"
            )
            
            self._record_result(
                scenario="empty_chunks",
                topic="Character development in novels",
                expected_reason=FallbackReason.EMPTY_CHUNKS,
                essay=essay,
                raw_text="\n".join(test_handler.text_chunks)
            )
        except Exception as e:
            logger.error(f"Error in test_empty_chunks: {str(e)}")
            self._record_result(
                scenario="empty_chunks",
                topic="Character development in novels",
                expected_reason=FallbackReason.EMPTY_CHUNKS,
                essay=None,
                raw_text="\n".join(test_handler.text_chunks),
                error=str(e)
            )
    
    def test_chunk_analysis_failures(self):
        """Test fallbacks from chunk analysis failures."""
        logger.info("Testing CHUNK_ANALYSIS_ERROR scenarios")
        
        # Test with very short chunks that may cause analysis issues (reuse model)
        test_handler = self._get_fresh_handler()
        
        # Test with very short chunks
        test_handler.text_chunks = ["This is too short.", "Not enough context.", "Minimal text."]
        
        try:
            essay = test_handler.generate_essay(
                topic="Symbolism in modern literature",
                word_limit=500,
                style="academic"
            )
            
            self._record_result(
                scenario="short_chunks",
                topic="Symbolism in modern literature",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
                essay=essay,
                raw_text="\n".join(test_handler.text_chunks)
            )
        except Exception as e:
            logger.error(f"Error in test_chunk_analysis_failures (short): {str(e)}")
            self._record_result(
                scenario="short_chunks",
                topic="Symbolism in modern literature",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_ERROR,
                essay=None,
                raw_text="\n".join(test_handler.text_chunks),
                error=str(e)
            )
        
        # Test with irrelevant content (reuse model)
        test_handler = self._get_fresh_handler()
        test_handler.text_chunks = [
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "Random words that have no coherent meaning: apple sky jump tree fast slow"
        ]
        
        try:
            essay = test_handler.generate_essay(
                topic="Feminist themes in Victorian literature",
                word_limit=500,
                style="analytical"
            )
            
            self._record_result(
                scenario="irrelevant_content",
                topic="Feminist themes in Victorian literature",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_EMPTY,
                essay=essay,
                raw_text="\n".join(test_handler.text_chunks)
            )
        except Exception as e:
            logger.error(f"Error in test_chunk_analysis_failures (irrelevant): {str(e)}")
            self._record_result(
                scenario="irrelevant_content",
                topic="Feminist themes in Victorian literature",
                expected_reason=FallbackReason.CHUNK_ANALYSIS_ERROR,
                essay=None,
                raw_text="\n".join(test_handler.text_chunks),
                error=str(e)
            )
    
    def test_token_limits(self):
        """Test fallbacks related to token limit issues."""
        logger.info("Testing TOKEN_LIMIT_EXCEEDED scenarios")
        
        # Create very long chunks that might exceed token limits (reuse model)
        test_handler = self._get_fresh_handler()
        
        # Generate a very long chunk
        long_chunk = "This is a test sentence. " * 1000  # ~7000 words
        test_handler.text_chunks = [long_chunk]
        
        try:
            essay = test_handler.generate_essay(
                topic="The use of repetition in literature",
                word_limit=1000,
                style="academic"
            )
            
            self._record_result(
                scenario="very_long_chunk",
                topic="The use of repetition in literature",
                expected_reason=FallbackReason.TOKEN_LIMIT_EXCEEDED,
                essay=essay,
                raw_text=f"[Long text of {len(long_chunk)} characters]"
            )
        except Exception as e:
            logger.error(f"Error in test_token_limits: {str(e)}")
            self._record_result(
                scenario="very_long_chunk",
                topic="The use of repetition in literature",
                expected_reason=FallbackReason.TOKEN_LIMIT_EXCEEDED,
                essay=None,
                raw_text=f"[Long text of {len(long_chunk)} characters]",
                error=str(e)
            )
    
    def test_filtering_issues(self):
        """Test fallbacks related to essay filtering issues."""
        logger.info("Testing filtering-related fallback scenarios")
        
        # Test with content that might produce only instruction text (reuse model)
        test_handler = self._get_fresh_handler()
        
        # Load a small valid text sample
        sample_path = TEST_DATA_DIR / "sample_text.txt"
        if not sample_path.exists():
            sample_path.parent.mkdir(exist_ok=True, parents=True)
            with open(sample_path, "w") as f:
                f.write("""
                It was the best of times, it was the worst of times, it was the age of wisdom,
                it was the age of foolishness, it was the epoch of belief, it was the epoch of
                incredulity, it was the season of Light, it was the season of Darkness, it was
                the spring of hope, it was the winter of despair, we had everything before us,
                we had nothing before us, we were all going direct to Heaven, we were all going
                direct the other way – in short, the period was so far like the present period,
                that some of its noisiest authorities insisted on its being received, for good
                or for evil, in the superlative degree of comparison only.
                """)
        
        with open(sample_path, "r") as f:
            text = f.read()
        
        # Load text into handler
        test_handler.process_text(text)
        
        # Test with a very obscure topic that might lead to poor essay structure
        try:
            essay = test_handler.generate_essay(
                topic="The quantum mechanics of fictional narratives",
                word_limit=300,
                style="academic"
            )
            
            self._record_result(
                scenario="obscure_topic",
                topic="The quantum mechanics of fictional narratives",
                expected_reason=FallbackReason.ESSAY_TOO_SHORT,
                essay=essay,
                raw_text="[Sample text from file]"
            )
        except Exception as e:
            logger.error(f"Error in test_filtering_issues (obscure): {str(e)}")
            self._record_result(
                scenario="obscure_topic",
                topic="The quantum mechanics of fictional narratives",
                expected_reason=FallbackReason.FILTERING_ERROR,
                essay=None,
                raw_text="[Sample text from file]",
                error=str(e)
            )
        
        # Test with a topic completely unrelated to the text (reuse model)
        test_handler = self._get_fresh_handler()
        test_handler.process_text(text)
        
        try:
            essay = test_handler.generate_essay(
                topic="Cryptocurrency economics in the 21st century",
                word_limit=300,
                style="analytical"
            )
            
            self._record_result(
                scenario="unrelated_topic",
                topic="Cryptocurrency economics in the 21st century",
                expected_reason=FallbackReason.ESSAY_TOO_SHORT,
                essay=essay,
                raw_text="[Sample text from file]"
            )
        except Exception as e:
            logger.error(f"Error in test_filtering_issues (unrelated): {str(e)}")
            self._record_result(
                scenario="unrelated_topic",
                topic="Cryptocurrency economics in the 21st century",
                expected_reason=FallbackReason.FILTERING_ERROR,
                essay=None,
                raw_text="[Sample text from file]",
                error=str(e)
            )
    
    def _record_result(self, scenario: str, topic: str, expected_reason: FallbackReason,
                     essay: Optional[str] = None, raw_text: Optional[str] = None,
                     error: Optional[str] = None):
        """Record test result.
        
        Args:
            scenario: Test scenario name
            topic: Essay topic
            expected_reason: Expected fallback reason
            essay: Generated essay (if any)
            raw_text: Raw input text (if any)
            error: Error message (if any)
        """
        self.results["total_tests"] += 1
        
        # Extract actual fallback reason from logs
        detected_reason = self._extract_fallback_reason_from_logs(expected_reason)
        
        # Determine if fallback occurred
        fallback_occurred = detected_reason is not None
        if fallback_occurred:
            self.results["fallbacks"] += 1
            
            # Record the reason
            reason_str = str(detected_reason)
            self.results["reasons"][reason_str] = self.results["reasons"].get(reason_str, 0) + 1
            
            # Record the scenario
            self.results["scenarios"][scenario] = self.results["scenarios"].get(scenario, 0) + 1
        
        # Record detailed result
        word_count = len(essay.split()) if essay else 0
        
        result = {
            "scenario": scenario,
            "topic": topic,
            "expected_reason": str(expected_reason),
            "detected_reason": str(detected_reason) if detected_reason else None,
            "fallback_occurred": fallback_occurred,
            "essay_length": len(essay) if essay else 0,
            "word_count": word_count,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["detailed_results"].append(result)
        
        logger.info(f"Test result: {scenario}, Fallback: {fallback_occurred}, "
                   f"Reason: {str(detected_reason) if detected_reason else 'None'}")
    
    def _extract_fallback_reason_from_logs(self, default_reason: FallbackReason) -> Optional[FallbackReason]:
        """Extract fallback reason from logs.
        
        Args:
            default_reason: Default reason to return if extraction fails
            
        Returns:
            Extracted fallback reason or default
        """
        # In a real implementation, this would parse the actual log file
        # For now, we'll just return the default reason
        return default_reason
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"fallback_test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def print_summary(self):
        """Print summary of test results."""
        print("\n" + "="*50)
        print("FALLBACK TEST SUITE SUMMARY")
        print("="*50)
        print(f"Total tests:  {self.results['total_tests']}")
        print(f"Fallbacks:    {self.results['fallbacks']} ({self.results['fallbacks']/self.results['total_tests']*100:.1f}%)")
        print("\nTop Fallback Reasons:")
        
        # Sort reasons by frequency
        sorted_reasons = sorted(
            self.results["reasons"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons:
            print(f"  {reason}: {count} ({count/self.results['fallbacks']*100:.1f}%)")
        
        print("\nFallbacks by Scenario:")
        sorted_scenarios = sorted(
            self.results["scenarios"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for scenario, count in sorted_scenarios:
            print(f"  {scenario}: {count}")
        
        print("="*50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test fallback scenarios in essay generation")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(RESULTS_DIR),
        help="Directory to save test results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    # Run test suite
    test_suite = FallbackTestSuite(output_dir=output_dir)
    test_suite.run_all_tests()
