"""Main Streamlit application for Book to Essay Generator."""
import streamlit as st
from pathlib import Path
import tempfile
from ai_book_to_essay_generator import AIBookEssayGenerator
from config import SUPPORTED_FORMATS, MAX_UPLOAD_SIZE_MB

st.set_page_config(
    page_title="Book to Essay Generator",
    page_icon="📚",
    layout="wide"
)

def main():
    st.title("📚 Book to Essay Generator")
    
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state.generator = AIBookEssayGenerator()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("📥 Upload Books")
        uploaded_files = st.file_uploader(
            "Upload your books",
            accept_multiple_files=True,
            type=[fmt.replace('.', '') for fmt in SUPPORTED_FORMATS]
        )
        
        st.caption(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        st.caption(f"Max file size: {MAX_UPLOAD_SIZE_MB}MB")

    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Essay Parameters")
        topic = st.text_area("Essay Topic/Prompt", height=100)
        word_limit = st.number_input("Word Limit", min_value=100, max_value=5000, value=500, step=100)
        style = st.selectbox("Writing Style", ["Academic", "Analytical", "Argumentative", "Expository"])
        
        if st.button("Generate Essay", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one book first!")
                return
            if not topic:
                st.error("Please provide an essay topic!")
                return
                
            with st.spinner("Processing books and generating essay..."):
                # Process uploaded files
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        if tmp_file.name.endswith('.pdf'):
                            st.session_state.generator.load_pdf_file(tmp_file.name)
                        elif tmp_file.name.endswith('.txt'):
                            st.session_state.generator.load_txt_file(tmp_file.name)
                        elif tmp_file.name.endswith('.epub'):
                            st.session_state.generator.load_epub_file(tmp_file.name)
                
                # Generate essay
                essay = st.session_state.generator.generate_essay(topic, word_limit)
                st.session_state.essay = essay
                st.session_state.quotes = st.session_state.generator.extract_quotes()

    with col2:
        st.header("Generated Essay")
        if 'essay' in st.session_state:
            st.markdown(st.session_state.essay)
            
            st.subheader("Supporting Quotes")
            for quote in st.session_state.quotes:
                st.markdown(f"> {quote}")
            
            col_word, col_pdf = st.columns(2)
            with col_word:
                st.download_button(
                    "Download as Word",
                    st.session_state.essay,
                    file_name="generated_essay.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            with col_pdf:
                st.download_button(
                    "Download as PDF",
                    st.session_state.essay,
                    file_name="generated_essay.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
