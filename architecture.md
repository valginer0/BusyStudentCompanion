# BusyStudentCompanion – Architecture & Design Overview

This document describes the **system architecture, key design decisions, and AI-related dependencies** for **BusyStudentCompanion**.

---

## 1. System Goals

* Transform large book files (TXT / PDF / EPUB) into concise, MLA-compliant essays.
* Run entirely **locally** on CPU or GPU, using 4-bit quantisation for low memory.
* Provide both **Streamlit** (GUI) and **FastAPI** (programmatic) interfaces.
* Cache heavy assets (LLM weights, chunk analyses, generated essays) for speed.

---

## 2. High-Level Architecture

```mermaid
flowchart TD
    A[Book File<br/>(txt / pdf / epub)] -->|PyMuPDF / EbookLib| B[Text Extractor]
    B --> C[AI Book-to-Essay Generator]
    C -->|split text| D[Chunk Analysis Manager]
    D -->|cached? yes| E[Chunk Cache]
    D -->|no| F[DeepSeekHandler]
    F -->|transformers + bitsandbytes| G[Local LLM<br/>(Mistral / DeepSeek)]
    G --> F
    F --> H[Chunk Analyses]
    H --> C
    C --> I[Essay Formatter<br/>+ MLA Citations]
    I --> J[Essay Cache]
    I --> K[Streamlit UI / FastAPI]
```

* **Text Extractor** – Uses **PyMuPDF** (PDF) & **EbookLib** (EPUB) to pull raw text.
* **AI Book-to-Essay Generator (`ai_book_to_essay_generator.py`)** – Orchestrates loading, caching and final essay assembly.
* **Chunk Analysis Manager** – Splits text into ~500-word segments and reuses cached analyses.
* **DeepSeekHandler** – Wraps **transformers** model; builds prompts, calls the LLM, performs post-processing.
* **Cache Layers**
  * **Model Weights** – Hugging Face weights cached in `~/.cache/busy_student_companion/models/*`
  * **Chunk Analyses** – `.pkl` files per chunk in `chunk_cache/`
  * **Essay Outputs** – Final essays cached by `(file_hash, prompt, style)`

---

## 3. Core Python Packages & Their Roles

| Package | Purpose | Where Used |
|---------|---------|------------|
| **transformers** | Runs 7-B LLMs locally (Mistral / DeepSeek) | `model_loader.py`, `model_handler.py` |
| **bitsandbytes** | 4-/8-bit weight loading / quantisation | `model_loader.py` |
| **accelerate** | Device placement; manages CPU ↔ GPU | `model_loader.py` |
| **nltk** | Sentence segmentation for chunking | `chunk_analysis_manager.py` |
| **PyMuPDF (fitz)** | PDF text extraction | `ai_book_to_essay_generator.py` |
| **EbookLib** | EPUB parsing | `ai_book_to_essay_generator.py` |
| **BeautifulSoup4** | HTML sanitisation inside EPUBs | `ai_book_to_essay_generator.py` |
| **streamlit** | Interactive GUI | `streamlit_app.py` |
| **fastapi** | REST API (optional) | `api.py` |
| **psutil** | Memory monitoring | `utils/memory.py` |
| **pytest / pytest-mock** | Testing framework | `tests/` |

> All AI inference happens **locally**; no paid APIs required.

---

## 4. Repository Structure (key files)

```
BusyStudentCompanion/
├─ src/
│   └─ book_to_essay/
│       ├─ ai_book_to_essay_generator.py  # Orchestrator
│       ├─ model_loader.py                # Downloads & loads LLM
│       ├─ model_handler.py               # DeepSeekHandler logic
│       ├─ chunk_analysis_manager.py      # Chunk caching / splitting
│       ├─ cache_manager.py               # Disk-based caches
│       ├─ prompts/                       # Prompt templates & extraction utils
│       ├─ streamlit_app.py               # Web interface
│       └─ api.py                         # FastAPI endpoints (optional)
├─ architecture.md                        # ← current file
└─ README.md
```

---

## 5. Processing Pipeline

1. **File Upload / Selection**
2. **Extraction** – Raw text + metadata (Author, Title) parsed; filename fallback.
3. **Chunking** – `ChunkAnalysisManager` splits text via **nltk**.
4. **Cache Check** – Each chunk hashed; existing analyses loaded.
5. **Prompt Generation** – `DeepSeekHandler` builds analysis prompt from template.
6. **LLM Inference** – Quantised model generates chunk analysis and, later, combined essay.
7. **MLA Formatting** – Citations & bibliography generated; essay trimmed to word limit.
8. **Delivery** – Essay shown in Streamlit or returned via FastAPI; cached for future calls.

---

## 6. Caching Strategy

| Cache | Path | Invalidation |
|-------|------|--------------|
| Model Weights | `~/.cache/busy_student_companion/models/*` | Manual (rare) |
| Chunk Analyses | `~/.cache/busy_student_companion/models/chunk_cache/*.pkl` | Clear when prompt logic changes |
| Essays | `~/.cache/busy_student_companion/essays/*.pkl` | Hash-based uniqueness |

---

## 7. Design Principles

1. **Local-First & Privacy-Aware** – Student content never leaves machine.
2. **Modular Components** – Extraction, analysis, generation, UI are decoupled.
3. **Graceful Fallback** – Secondary generation path if first LLM call fails.
4. **Efficient** – 4-bit quantisation; per-chunk caching; streaming.
5. **Test-Driven** – Extensive pytest suite ensures regressions are caught.

---

## 8. Extending the System

* **Switch Model** – Change `MODEL_NAME` in `.env` or `config.py`; loader auto-downloads.
* **GPU vs CPU** – `accelerate` auto-detects CUDA.
* **Add Prompt Template** – Extend `prompts/factory.py` and wire into `DeepSeekHandler`.
* **Support New Formats** – Implement parser in `ai_book_to_essay_generator.py` and register its extension.

---

## 9. Roadmap Ideas

| Idea | Benefit |
|------|---------|
| Vector store of past essays | Self-plagiarism checks & retrieval |
| ReAct-style agent refinement | Multi-step draft → refine loops |
| Collaborative workspace | Real-time editing / sharing |
| Grammar / style checker | Integrate `language-tool-python` |

---

© 2025 BusyStudentCompanion – MIT License
