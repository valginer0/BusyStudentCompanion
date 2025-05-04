"""AI Book Essay Generator using DeepSeek model."""
import os
import fitz  # PyMuPDF for reading PDFs
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import List, Optional, Dict, Any
from src.book_to_essay.model_handler import DeepSeekHandler
from src.book_to_essay.cache_manager import CacheManager
from src.book_to_essay.validation import validate_word_count, validate_file_extension, validate_style, validate_filename_for_citation
from src.book_to_essay.error_utils import log_and_raise
import logging
from src.book_to_essay.essay_utilities import get_essay_cache_key

logger = logging.getLogger(__name__)

class AIBookEssayGenerator:
    def __init__(self):
        self.content = ""
        self.sources = []
        self._model = None
        self.cache_manager = CacheManager()

    @property
    def model(self):
        """Get the model handler, creating it if it doesn't exist."""
        if self._model is None:
            try:
                logger.info("Creating new DeepSeekHandler instance")
                # Properly load model and tokenizer before passing to DeepSeekHandler
                from src.book_to_essay.model_loader import load_model, load_tokenizer
                base_model = load_model()
                tokenizer = load_tokenizer()
                self._model = DeepSeekHandler(model=base_model, tokenizer=tokenizer)
            except Exception as e:
                logger.error(f"Error initializing model: {str(e)}")
                raise RuntimeError(f"Failed to initialize model: {str(e)}")
        return self._model

    def extract_metadata_from_text(self, text):
        """Extract Author and Title from the first 40 lines of the text, fallback to None if not found."""
        title, author = None, None
        for line in text.splitlines()[:40]:
            if line.lower().startswith('title:'):
                title = line.split(':', 1)[1].strip()
            elif line.lower().startswith('author:'):
                author = line.split(':', 1)[1].strip()
        return author, title

    def _process_file_content(self, file_path: str, processor_func) -> Dict[str, Any]:
        """Process file content with caching and extract metadata from content or filename."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension and citation filename
        validate_file_extension(file_path)
        validate_filename_for_citation(file_path)
        
        # Check cache first
        file_hash = self.cache_manager._get_file_hash(file_path)
        logger.info(f"[CACHE DEBUG] File: {file_path} | Hash: {file_hash}")
        cached_content = self.cache_manager.get_cached_content(file_path)
        if cached_content is not None:
            self.content += cached_content["content"] + "\n"
            # Ensure hash is present in source dict
            if 'hash' not in cached_content["source"]:
                file_hash = self.cache_manager._get_file_hash(file_path)
                cached_content["source"]['hash'] = file_hash
            logger.info(f"[CACHE DEBUG] Loaded cached content for file {file_path} with hash {file_hash}")
            self.sources.append(cached_content["source"])
            return cached_content

        # Process content if not cached
        content = processor_func(file_path)
        if not content:
            raise ValueError(f"No content could be extracted from {file_path}")

        # Try to extract metadata from content
        author, title = self.extract_metadata_from_text(content)
        logger.debug(f"Extracted metadata - Author: {author}, Title: {title}")
        # Fallback: parse from filename if not found in content
        if not (author and title):
            logger.warning(f"Metadata extraction incomplete. Author: {author}, Title: {title}. Attempting filename fallback.")
            # Parse filename: "Author - Title - Extra.txt" or "Author - Title.txt"
            base = os.path.basename(file_path)
            base = base.rsplit('.', 1)[0]
            import re
            match = re.match(r'^(.*?) - (.*?)(?: - .*)?$', base)
            if match:
                author = author or match.group(1).strip()
                title = title or match.group(2).strip()
        logger.info(f"Final metadata used - Author: {author}, Title: {title}")
        file_hash = self.cache_manager._get_file_hash(file_path)
        source = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'type': os.path.splitext(file_path)[1][1:],
            'author': author,
            'title': title,
            'hash': file_hash
        }
        logger.info(f"[CACHE DEBUG] Source dict to be cached: {source}")
        logger.info(f"[CACHE DEBUG] Caching content for file {file_path} with hash {file_hash}")
        # Cache the processed content
        cache_data = {
            "content": content,
            "source": source
        }
        self.cache_manager.cache_content(file_path, cache_data)
        # Update current state
        self.content += content + "\n"
        self.sources.append(source)
        return cache_data

    def load_txt_file(self, file_path: str):
        """Load content from a text file with caching."""
        def process_txt(file_path: str) -> str:
            """Reads content from a TXT file."""
            logger.info(f"Processing TXT file: {file_path}")
            try:
                # Ensure correct encoding is used
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        text = file.read()
                        logger.info(f"Processing text with model instance {id(self.model)}")
                        self.model.process_text(text)
                        return text
                    except (IOError, UnicodeDecodeError) as e:
                        log_and_raise(f"Error reading TXT file {file_path}", e)
            except FileNotFoundError:
                logger.error(f"TXT file not found: {file_path}")
                raise
            except Exception as e:
                log_and_raise(f"Unexpected error opening or processing TXT file {file_path}", e)
        
        self._process_file_content(file_path, process_txt)
        logger.debug(f"Loaded content length: {len(self.content)}")
        if hasattr(self, 'model') and hasattr(self.model, 'process_text'):
            self.model.process_text(self.content)
            if hasattr(self.model, 'text_chunks'):
                logger.debug(f"Number of text_chunks after process_text: {len(self.model.text_chunks)}")
            else:
                logger.debug("Model has no text_chunks attribute after process_text.")

    def load_pdf_file(self, file_path: str):
        """Load content from a PDF file with caching."""
        def process_pdf(file_path):
            try:
                pdf_document = fitz.open(file_path)
                content = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    content += page.get_text() + "\n"
                pdf_document.close()
                self.model.process_text(content)
                if not content.strip():
                    raise ValueError("PDF appears to be empty or unreadable")
                return content
            except Exception as e:
                log_and_raise(f"Error reading PDF {file_path}", e)
        
        self._process_file_content(file_path, process_pdf)

    def load_epub_file(self, file_path: str):
        """Load content from an EPUB file with caching."""
        def process_epub(file_path):
            try:
                book = epub.read_epub(file_path)
                content = ""
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        content += soup.get_text() + "\n"
                self.model.process_text(content)
                if not content.strip():
                    raise ValueError("EPUB appears to be empty or contains no readable text")
                return content
            except Exception as e:
                log_and_raise(f"Error reading EPUB {file_path}", e)
        
        self._process_file_content(file_path, process_epub)

    def generate_essay(self, prompt: str, word_limit: int = 500, style: str = "academic") -> str:
        """Generate an essay using the DeepSeek model with chunking and MLA formatting.
        
        This method uses text chunking to handle large documents efficiently while
        maintaining proper MLA formatting and citations.
        
        Args:
            prompt: The essay topic or prompt
            word_limit: Maximum word count for the essay
            style: Writing style (academic, analytical, argumentative, expository)
            
        Returns:
            A formatted essay with MLA citations
        """
        # Validate inputs
        validate_word_count(word_limit)
        validate_style(style)
        
        # Ensure we have content to process
        if not self.content:
            raise ValueError("No content has been loaded. Please load at least one file first.")
            
        # Check cache for existing essay with these parameters
        cache_key = get_essay_cache_key(prompt, word_limit, style, self.sources)
        cached_essay = self.cache_manager.get_cached_model_output(cache_key, self.content)
        if cached_essay is not None:
            return cached_essay
        
        try:
            logger.info(f"Generating essay with prompt: {prompt}, word_limit: {word_limit}, style: {style}")
            
            # Ensure the model has processed the text
            if not hasattr(self.model, 'text_chunks') or not self.model.text_chunks:
                logger.info("Processing text content with model")
                self.model.process_text(self.content)
                
            # Double-check that text chunks were created
            if not self.model.text_chunks:
                logger.error("Failed to create text chunks. Content may be empty or too short.")
                raise ValueError("Failed to process text content. Please check the input file.")
                
            logger.info(f"Model has {len(self.model.text_chunks)} text chunks to process")
            
            essay = None
            try:
                essay = self.model.generate_essay(
                    topic=prompt,
                    word_limit=word_limit,
                    style=style.lower(),
                    sources=self.sources
                )
            except Exception as e:
                logger.warning(f"Initial essay generation failed: {str(e)}")
                raise ValueError(f"Error generating essay: {str(e)}") from e

            # If initial generation SUCCEEDED but returned empty, try fallback
            if not essay:
                logger.info("Initial generation returned empty result. Attempting fallback...")
                try:
                    essay = self.model.generate_fallback_essay(
                        topic=prompt,
                        word_limit=word_limit,
                        style=style.lower(),
                        sources=self.sources
                    )
                    if essay:
                        logger.info("Fallback essay generation successful.")
                    else:
                        logger.error("Fallback essay generation also returned empty result.")
                except Exception as fallback_e:
                    logger.error(f"Fallback essay generation also failed: {fallback_e}")
                    # If fallback fails, raise an error referencing the original one if it exists
                    error_msg = f"Error generating essay: {str(fallback_e)}"
                    raise ValueError(error_msg) from fallback_e

            # If fallback also failed or returned empty, raise final error
            if not essay:
                raise ValueError("Failed to generate essay after fallback.")
                
            # Cache the generated essay (only if successful)
            self.cache_manager.cache_model_output(cache_key, self.content, essay)
            
            return essay
        except Exception as e:
            # This outer catch is a safety net, specific errors should be handled above
            # Log the error including traceback for better debugging
            logger.exception(f"Unhandled error during essay generation process: {str(e)}")
            # Raise a generic error, potentially wrapping the caught one
            raise ValueError(f"An unexpected error occurred during essay generation: {str(e)}") from e

    def extract_quotes(self, num_quotes: int = 5) -> List[str]:
        """Extract relevant quotes from the content."""
        sentences = re.split(r'(?<=[.!?]) +', self.content)
        # TODO: Implement more sophisticated quote extraction using semantic similarity
        return sentences[:num_quotes]

    def format_mla_quote(self, quote: str, source: str) -> str:
        """Format a quote in MLA style."""
        return f'"{quote}" ({source}).'

    def get_bibliography(self) -> List[str]:
        """Generate MLA bibliography entries for the sources."""
        bibliography = []
        for source in self.sources:
            # Basic MLA format for different types
            if source['type'] == 'pdf':
                entry = f"{os.path.splitext(source['name'])[0]}. PDF file."
            elif source['type'] == 'epub':
                entry = f"{os.path.splitext(source['name'])[0]}. EPUB file."
            else:
                entry = f"{os.path.splitext(source['name'])[0]}. Text file."
            bibliography.append(entry)
        return bibliography
