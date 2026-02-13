# Multimodal RAG Workflow Studio (Text + Vision)

This project is an end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system for realistic document + image intelligence workflows.

It supports grounded Q&A over:

- Text documents (TXT, PDF)
- Images (charts, diagrams, screenshots)

---

## What's New in the Updated UI

- Redesigned UI with a realistic workflow layout and upgraded background/theme
- Clear 3-stage flow: **Ingestion -> Query -> Workflow Output**
- Feature cards that expose active multimodal capabilities
- Build-time reporting (files processed, chunks indexed, index build latency)
- New workflow output panel with:
  - Answer style controls (Concise / Detailed)
  - Modalities used indicator (text/image)
  - Feature trace for transparency
  - Extended source diagnostics table
- Hybrid reranking diagnostics (semantic score, lexical score, hybrid score)

---

## Core Features

- Text chunking with overlap controls
- Vision-based image analysis via Groq vision model
- Embeddings via Jina Embeddings v4
- FAISS vector retrieval with modality filter (all / text / image)
- Hybrid reranking (semantic distance + lexical overlap)
- Grounded generation with context-only instruction
- Session chat history and export

---

## Architecture Overview

1. User uploads documents and images
2. Text is chunked with configurable overlap
3. Images are converted into semantic descriptions
4. Text and image chunks are embedded with Jina
5. FAISS retrieves top-k chunks
6. Hybrid reranker reorders candidates using semantic + lexical signals
7. LLM produces grounded answer from selected context
8. UI shows workflow diagnostics and feature trace

---

## Project Structure

```bash
Multimodal_RAG-main/
|-- app.py
|-- requirements.txt
|-- config.py
|-- README.md
`-- rag/
    |-- __init__.py
    |-- embeddings.py
    |-- retriever.py
    |-- chunking.py
    |-- vision.py
    |-- reranker.py
    `-- llm.py
```
