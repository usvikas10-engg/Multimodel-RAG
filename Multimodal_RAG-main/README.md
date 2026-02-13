# Enterprise Multimodal RAG (Text + Vision)

This project is an end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that enables question answering over both:

- Text documents (TXT, PDF)
- Images (diagrams, charts, screenshots)

It combines:

- Document retrieval using embeddings + FAISS
- Image understanding using Groq Vision models
- Grounded answer generation using Groq LLMs
- A modern Streamlit user interface

---

## Live Demo

The application is deployed and accessible here:

https://rag-with-multimodality.streamlit.app/

---

## Key Features

- Text-based RAG over one or many uploaded TXT/PDF files  
- Multimodal RAG support with one or many images  
- Vision-based image captioning using:

  `meta-llama/llama-4-scout-17b-16e-instruct`

- Embedding generation using Jina Embeddings v4 API  
- Vector similarity retrieval with FAISS  
- Configurable chunking and retrieval controls (chunk size, overlap, top-k, context-k)  
- Reranking with source diagnostics (distance + lexical score)  
- Guardrails to reduce hallucinations (context-only answering)  
- Session memory with recent chat history  
- Latency tracking displayed in the UI  
- Metadata filtering (retrieve text-only, image-only, or both)
- Downloadable chat history export

---

## Architecture Overview

1. User uploads one or more documents/images
2. Content is converted into text chunks with configurable overlap
3. Image files are converted into semantic descriptions using Groq Vision
4. All chunks are embedded using Jina Embeddings v4
5. FAISS retrieves relevant chunks with optional type filtering
6. Chunks are reranked and assembled into grounded context
7. Groq LLM generates an answer grounded in retrieved context
8. UI displays diagnostics and supports chat history export

---

## Project Structure

```bash
multimodal-rag-jina4/
├── app.py
├── requirements.txt
├── config.py
├── README.md
└── rag/
    ├── __init__.py
    ├── embeddings.py
    ├── retriever.py
    ├── chunking.py
    ├── vision.py
    ├── reranker.py
    └── llm.py
