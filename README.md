# BusyStudentCompanion

A powerful AI-powered tool that helps students analyze books and generate well-structured essays. Using state-of-the-art AI technology (DeepSeek model), this companion tool streamlines the process of understanding literature and creating thoughtful essay content.

## Features

- **Multi-format Support**: Process various document formats including PDF, EPUB, and DOCX
- **AI-Powered Analysis**: Leverages the DeepSeek model for intelligent text understanding
- **Essay Generation**: Creates well-structured essays based on book content
- **Smart Caching**: Efficient content caching system for improved performance
- **Web Interface**: User-friendly Streamlit-based web interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/valginer0/BusyStudentCompanion.git
cd BusyStudentCompanion
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys and configuration
```

## Usage

1. Start the web interface:
```bash
streamlit run src/book_to_essay/streamlit_app.py
```

2. Upload your book file (PDF, EPUB, or DOCX format)
3. The AI will analyze the content and generate an essay
4. Review and download the generated essay

## Project Structure

```
BusyStudentCompanion/
├── src/
│   └── book_to_essay/
│       ├── ai_book_to_essay_generator.py  # Main AI processing logic
│       ├── cache_manager.py              # Content caching system
│       ├── config.py                     # Configuration management
│       ├── model_handler.py              # DeepSeek model integration
│       └── streamlit_app.py             # Web interface
├── .env.example                         # Example environment variables
├── requirements.txt                     # Python dependencies
└── README.md                           # Project documentation
```

## Configuration

The following environment variables can be configured in `.env`:

- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `CACHE_DIR`: Directory for cached content (default: `.cache`)
- `MAX_CACHE_SIZE`: Maximum cache size in MB (default: 1000)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek for their powerful language model
- Streamlit for the web interface framework
- PyMuPDF and ebooklib for document processing capabilities