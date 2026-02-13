import streamlit as st
import time
import hashlib

from pypdf import PdfReader

from rag.embeddings import get_jina_embeddings
from rag.vision import describe_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank_with_scores
from rag.llm import ask_llm


st.set_page_config(
    page_title="Multimodal RAG Assistant",
    layout="wide"
)


st.markdown(
    """
    <style>
    body {
        font-family: "Inter", sans-serif;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    .panel {
        padding: 18px;
        border-radius: 14px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .section-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #111827;
    }

    .stButton button {
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<div class='main-title'>Enterprise Multimodal RAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Retrieval-Augmented Generation over documents and images using Jina Embeddings and Groq Vision</div>",
    unsafe_allow_html=True
)


if "history" not in st.session_state:
    st.session_state.history = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []
if "kb_metadata" not in st.session_state:
    st.session_state.kb_metadata = []
if "kb_retriever" not in st.session_state:
    st.session_state.kb_retriever = None
if "kb_signature" not in st.session_state:
    st.session_state.kb_signature = None


with st.sidebar:
    st.header("Configuration")

    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )

    filter_type = st.radio(
        "Retrieval Scope",
        ["all", "text", "image"],
        horizontal=True
    )

    chunk_size = st.slider("Chunk Size (words)", 100, 800, 400, 50)
    overlap = st.slider("Chunk Overlap (words)", 0, 250, 80, 10)
    top_k = st.slider("Retrieve Top-K", 3, 15, 7, 1)
    context_k = st.slider("Context Chunks for LLM", 1, 8, 4, 1)

    st.divider()


col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Document Uploads</div>", unsafe_allow_html=True)
    txt_files = st.file_uploader(
        "Upload one or more TXT/PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Image Uploads</div>", unsafe_allow_html=True)
    img_files = st.file_uploader(
        "Upload one or more PNG/JPG files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _safe_extract_text(file_obj):
    if file_obj.name.lower().endswith(".pdf"):
        reader = PdfReader(file_obj)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return "\n".join(pages)

    return file_obj.read().decode("utf-8", errors="ignore")


def _files_signature(doc_files, image_files, chunk_sz, overlap_sz):
    digest = hashlib.sha256()
    digest.update(str(chunk_sz).encode("utf-8"))
    digest.update(str(overlap_sz).encode("utf-8"))

    for f in doc_files + image_files:
        digest.update(f.name.encode("utf-8"))
        digest.update(str(f.size).encode("utf-8"))

    return digest.hexdigest()


build_clicked = st.button("Build / Refresh Knowledge Base", use_container_width=True)

if build_clicked:
    if not jina_key:
        st.error("Add Jina API key before building the knowledge base.")
    elif not (txt_files or img_files):
        st.error("Upload at least one document or image.")
    elif overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        signature = _files_signature(txt_files or [], img_files or [], chunk_size, overlap)

        if signature == st.session_state.kb_signature and st.session_state.kb_ready:
            st.info("Knowledge base is already up to date with current uploads and settings.")
        else:
            with st.spinner("Processing knowledge sources..."):
                chunks = []
                metadata = []

                try:
                    for text_file in txt_files or []:
                        text_file.seek(0)
                        raw_text = _safe_extract_text(text_file)
                        text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                        chunks.extend(text_chunks)
                        metadata.extend(
                            [{"type": "text", "source": text_file.name} for _ in text_chunks]
                        )
                except Exception as e:
                    st.error(f"Failed to parse document: {e}")
                    st.stop()

                if img_files:
                    if not groq_key:
                        st.error("Add Groq API key to include images in the knowledge base.")
                        st.stop()

                    try:
                        for image_file in img_files:
                            image_file.seek(0)
                            image_bytes = image_file.read()
                            vision_text = describe_image(image_bytes, groq_key)
                            if vision_text:
                                chunks.append("Image description: " + vision_text)
                                metadata.append({"type": "image", "source": image_file.name})
                    except Exception as e:
                        st.error(f"Failed to process image: {e}")
                        st.stop()

                if not chunks:
                    st.error("No usable content found in uploaded files.")
                    st.stop()

                try:
                    embeddings = get_jina_embeddings(chunks, jina_key)
                    retriever = FAISSRetriever(embeddings, metadata)
                except Exception as e:
                    st.error(f"Embedding or indexing failed: {e}")
                    st.stop()

                st.session_state.kb_chunks = chunks
                st.session_state.kb_metadata = metadata
                st.session_state.kb_retriever = retriever
                st.session_state.kb_ready = True
                st.session_state.kb_signature = signature
                st.success(f"Knowledge base ready with {len(chunks)} indexed chunks.")

if st.session_state.kb_ready:

    chunks = st.session_state.kb_chunks
    metadata = st.session_state.kb_metadata
    retriever = st.session_state.kb_retriever

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Query Interface</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Enter your question",
        placeholder="Example: What does the uploaded image explain?"
    )

    run = st.button("Run Retrieval and Generate Answer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run and query:
        if not groq_key or not jina_key:
            st.error("Add Groq and Jina API keys before running a query.")
            st.stop()

        start = time.time()

        try:
            query_emb = get_jina_embeddings([query], jina_key)
        except Exception as e:
            st.error(f"Query embedding failed: {e}")
            st.stop()

        f = None if filter_type == "all" else filter_type
        results = retriever.search(query_emb, top_k=top_k, filter_type=f)

        if not results:
            st.warning("No chunks matched the selected retrieval scope.")
            st.stop()

        retrieved_docs = [chunks[item["id"]] for item in results]
        reranked = simple_rerank_with_scores(query, retrieved_docs)
        selected = reranked[:context_k]

        context_parts = []
        source_rows = []
        for row in selected:
            idx = row["doc_id"]
            base = results[idx]
            source = base["metadata"].get("source", "unknown")
            dtype = base["metadata"].get("type", "unknown")
            text = row["doc"]
            context_parts.append(f"[{source} | {dtype}] {text}")
            source_rows.append(
                {
                    "source": source,
                    "type": dtype,
                    "distance": round(base["score"], 4),
                    "lexical_score": row["lexical_score"]
                }
            )

        context = "\n\n".join(context_parts)

        try:
            answer = ask_llm(context, query, groq_key, model)
        except Exception as e:
            st.error(f"Answer generation failed: {e}")
            st.stop()

        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Answer</div>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Performance</div>", unsafe_allow_html=True)
        st.metric("Latency (seconds)", latency)
        st.metric("Retrieved Chunks", len(results))
        st.metric("Context Chunks Used", len(selected))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Retrieved Context"):
            st.text(context)

        with st.expander("Source Diagnostics"):
            st.dataframe(source_rows, use_container_width=True, hide_index=True)

        with st.expander("Recent Chat History"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"Question: {q}")
                st.markdown(f"Answer: {a}")
                st.divider()

        chat_export = "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a in st.session_state.history]
        )
        st.download_button(
            "Download Chat History",
            data=chat_export.encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("Upload files, add API keys, and build the knowledge base to begin.")
