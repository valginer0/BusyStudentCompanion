"""AI Book Essay Generator using DeepSeek model."""
import os
import fitz  # PyMuPDF for reading PDFs
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from typing import List, Optional, Dict, Any
from .model_handler import DeepSeekHandler
from .cache_manager import CacheManager

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
            self._model = DeepSeekHandler()
        return self._model

    def _process_file_content(self, file_path: str, processor_func) -> Dict[str, Any]:
        """Process file content with caching."""
        # Check cache first
        cached_content = self.cache_manager.get_cached_content(file_path)
        if cached_content is not None:
            self.content += cached_content["content"] + "\n"
            self.sources.append(cached_content["source"])
            return cached_content

        # Process content if not cached
        content = processor_func(file_path)
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
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        self._process_file_content(file_path, process_txt)

    def load_pdf_file(self, file_path: str):
        """Load content from a PDF file with caching."""
        def process_pdf(file_path):
            pdf_document = fitz.open(file_path)
            content = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                content += page.get_text() + "\n"
            return content
        
        self._process_file_content(file_path, process_pdf)

    def load_epub_file(self, file_path: str):
        """Load content from an EPUB file with caching."""
        def process_epub(file_path):
            book = epub.read_epub(file_path)
            content = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    content += soup.get_text() + "\n"
            return content
        
        self._process_file_content(file_path, process_epub)

    def generate_essay(self, prompt: str, word_limit: int = 500, style: str = "academic") -> str:
        """Generate an essay using the DeepSeek model with caching."""
        # Prepare context with source information
        context = f"Sources:\n"
        for source in self.sources:
            context += f"- {source['name']} ({source['type']})\n"
        context += f"\nContent:\n{self.content}\n"
        
        # Check cache for existing essay
        cached_essay = self.cache_manager.get_cached_model_output(prompt, context)
        if cached_essay is not None:
            return cached_essay
        
        # Add style instructions
        style_instructions = {
            "academic": "Write in a formal academic style with proper citations.",
            "analytical": "Provide a detailed analysis with supporting evidence.",
            "argumentative": "Present a clear argument with supporting evidence.",
            "expository": "Explain the topic clearly and informatively."
        }
        
        # Calculate approximate token limit based on word limit
        token_limit = word_limit * 1.5  # Approximate tokens per word
        
        # Generate essay
        essay = self.model.generate_essay(
            context=context,
            prompt=f"{prompt}\n{style_instructions.get(style.lower(), style_instructions['academic'])}",
            max_length=int(token_limit)
        )
        
        # Cache the generated essay
        self.cache_manager.cache_model_output(prompt, context, essay)
        
        return essay

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
