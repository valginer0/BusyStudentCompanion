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
import logging

logger = logging.getLogger(__name__)

class AIBookEssayGenerator:
    def __init__(self):
        self.content = ""
        self.sources = []
        self._model = None
        self.cache_manager = CacheManager()

    @property
    def model(self):
        """Lazy loading of the model to save memory until needed."""
        if self._model is None:
            logger.info("Creating new DeepSeekHandler instance")
            self._model = DeepSeekHandler()
        else:
            logger.info("Using existing DeepSeekHandler instance")
        return self._model

    def _process_file_content(self, file_path: str, processor_func) -> Dict[str, Any]:
        """Process file content with caching."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check cache first
        cached_content = self.cache_manager.get_cached_content(file_path)
        if cached_content is not None:
            self.content += cached_content["content"] + "\n"
            self.sources.append(cached_content["source"])
            return cached_content

        # Process content if not cached
        content = processor_func(file_path)
        if not content:
            raise ValueError(f"No content could be extracted from {file_path}")
            
        source = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'type': os.path.splitext(file_path)[1][1:]
        }
        
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
        def process_txt(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    logger.info(f"Processing text with model instance {id(self.model)}")
                    self.model.process_text(text)
                    return text
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            text = file.read()
                            if text.strip():
                                logger.info(f"Processing text with model instance {id(self.model)}")
                                self.model.process_text(text)
                                return text
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"Could not decode file {file_path} with any supported encoding")
        
        self._process_file_content(file_path, process_txt)

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
                raise ValueError(f"Error reading PDF {file_path}: {str(e)}")
        
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
                raise ValueError(f"Error reading EPUB {file_path}: {str(e)}")
        
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
        # Ensure we have content to process
        if not self.content:
            raise ValueError("No content has been loaded. Please load at least one file first.")
            
        # Check cache for existing essay with these parameters
        cache_key = f"{prompt}_{word_limit}_{style}_{','.join([s['name'] for s in self.sources])}"
        cached_essay = self.cache_manager.get_cached_model_output(cache_key, self.content)
        if cached_essay is not None:
            return cached_essay
        
        # Generate essay with the enhanced model method
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
            
            essay = self.model.generate_essay(
                topic=prompt,
                word_limit=word_limit,
                style=style.lower(),
                sources=self.sources
            )
            
            # Cache the generated essay
            self.cache_manager.cache_model_output(cache_key, self.content, essay)
            
            return essay
        except Exception as e:
            logger.error(f"Error generating essay: {str(e)}")
            raise

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
