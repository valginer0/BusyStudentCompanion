"""Test script for the enhanced essay generation functionality."""
import logging
import os
import sys
import time
from src.book_to_essay.ai_book_to_essay_generator import AIBookEssayGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set a maximum test duration (2 hours)
MAX_TEST_DURATION = 7200  # seconds

# Set to True to use a smaller test file for faster testing
USE_SMALL_TEST = True

def main():
    """Test the enhanced essay generation functionality."""
    start_time = time.time()
    logger.info("Creating AIBookEssayGenerator instance")
    generator = AIBookEssayGenerator()
    
    # Test file path - using a sample text file from test_data
    test_file = os.path.join("test_data", "RomeoAndJulietFullTxt.txt")
    if USE_SMALL_TEST:
        # Use a smaller test file if available
        small_test_file = os.path.join("test_data", "RomeoAndJulietExcerpt.txt")
        if os.path.exists(small_test_file):
            test_file = small_test_file
            logger.info(f"Using smaller test file for faster testing: {small_test_file}")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
    
    try:
        logger.info(f"Loading test file: {test_file}")
        generator.load_txt_file(test_file)
        
        # Verify content was loaded
        if not generator.content:
            logger.error("No content was loaded from the file")
            return
            
        logger.info(f"Successfully loaded file with {len(generator.content)} characters")
        
        # Fix source information for proper MLA citation
        if generator.sources and 'name' in generator.sources[0]:
            # Update the name to follow the Author - Title format
            if not ' - ' in generator.sources[0]['name']:
                generator.sources[0]['name'] = "Shakespeare - Romeo and Juliet.txt"
                
        logger.info(f"Source information: {generator.sources}")
        
        # Generate essay
        prompt = "the theme of love and its consequences"
        word_limit = 200  # Reduced from 300 for faster testing
        style = "analytical"
        
        logger.info(f"Generating essay with prompt: {prompt}, word_limit: {word_limit}, style: {style}")
        
        # Print the prompt for debugging
        print(f"\nDEBUG - Using prompt: '{prompt}'\n")
        
        # Check if we're approaching the timeout
        if time.time() - start_time > MAX_TEST_DURATION * 0.8:  # 80% of max time
            logger.warning("Test is taking too long, may timeout soon")
            
        essay = generator.generate_essay(prompt, word_limit, style)
        
        # Check if we've exceeded the timeout
        if time.time() - start_time > MAX_TEST_DURATION:
            logger.error(f"Test exceeded maximum duration of {MAX_TEST_DURATION} seconds")
            print("\n" + "="*50 + " TEST TIMEOUT " + "="*50)
            print(f"Test exceeded maximum duration of {MAX_TEST_DURATION} seconds.")
            print("Consider using a smaller test file or model for testing.")
            print("="*120)
            return
            
        logger.info("Essay generation complete")
        print("\n" + "="*50 + " GENERATED ESSAY " + "="*50)
        print(essay)
        print("="*120)
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
