import hashlib
import time

import streamlit as st
from pypdf import PdfReader

from rag.chunking import chunk_text
from rag.embeddings import get_jina_embeddings
from rag.llm import ask_llm
from rag.reranker import hybrid_rerank_with_scores
from rag.retriever import FAISSRetriever
from rag.vision import describe_image


st.set_page_config(
    page_title="Multimodal RAG Studio",
    page_icon="??",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-1: #061c17;
        --bg-2: #0e3b34;
        --bg-3: #f2f6ee;
        --ink-1: #10211d;
        --ink-2: #35524b;
        --line: #d6e2d6;
        --accent: #0f766e;
        --accent-soft: #d6f1ea;
        --warn: #8b3f1f;
    }

    .stApp {
        font-family: "Manrope", sans-serif;
        background:
            radial-gradient(1000px 500px at 8% -10%, rgba(147, 197, 253, 0.18), transparent 70%),
            radial-gradient(900px 450px at 90% 0%, rgba(74, 222, 128, 0.14), transparent 68%),
            linear-gradient(170deg, var(--bg-1) 0%, var(--bg-2) 40%, var(--bg-3) 100%);
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
    }

    .hero {
        padding: 24px 26px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(214, 226, 214, 0.9);
        backdrop-filter: blur(2px);
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 700;
        line-height: 1.1;
        color: var(--ink-1);
        margin-bottom: 6px;
    }

    .hero-sub {
        color: var(--ink-2);
        font-size: 15px;
        margin-bottom: 0;
    }

    .panel {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .section-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--ink-1);
        margin-bottom: 8px;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 11px;
        background: var(--accent-soft);
        color: #0b4f49;
        font-size: 12px;
        border: 1px solid #bce6dc;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }

    .feature-item {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px;
        background: #f8fcf9;
        font-size: 13px;
        color: #21423a;
    }

    .status-ok {
        color: var(--accent);
        font-weight: 700;
    }

    .status-warn {
        color: var(--warn);
        font-weight: 700;
    }

    .stButton button {
        border-radius: 11px;
        border: 1px solid #0f766e;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e, #115e59);
        color: #ffffff;
    }

    @media (max-width: 900px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='hero'>
        <div class='hero-title'>Multimodal RAG Workflow Studio</div>
        <p class='hero-sub'>Build one grounded knowledge layer from documents + images, then inspect how retrieval, reranking, and generation contributed to every answer.</p>
    </div>
    """,
    unsafe_allow_html=True
)


def _init_session_state():
    defaults = {
        "history": [],
        "kb_ready": False,
        "kb_chunks": [],
        "kb_metadata": [],
        "kb_retriever": None,
        "kb_signature": None,
        "kb_build_report": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


with st.sidebar:
    st.header("Workflow Controls")

    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox(
        "Generation Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )

    retrieval_mode = st.radio(
        "Retrieval Scope",
        ["all", "text", "image"],
        horizontal=True
    )

    chunk_size = st.slider("Chunk Size (words)", 100, 900, 420, 40)
    overlap = st.slider("Chunk Overlap (words)", 0, 280, 90, 10)
    top_k = st.slider("Initial Retrieval Top-K", 3, 20, 8, 1)
    context_k = st.slider("Final Context Chunks", 1, 10, 5, 1)

    response_style = st.selectbox("Answer Style", ["Concise", "Detailed"]) 

    st.caption("Reranker blend")
    semantic_weight = st.slider("Semantic Weight", 0.1, 0.9, 0.7, 0.05)
    lexical_weight = round(1.0 - semantic_weight, 2)
    st.caption(f"Lexical Weight: {lexical_weight}")

    st.divider()


left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>1) Ingestion Inputs</div>", unsafe_allow_html=True)
    txt_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    img_files = st.file_uploader(
        "Upload image files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Active Features</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='feature-grid'>
            <div class='feature-item'><b>Hybrid Reranking</b><br/>Semantic distance + lexical overlap blended for stable retrieval.</div>
            <div class='feature-item'><b>Multimodal Sources</b><br/>Text chunks and vision-generated image descriptions in one index.</div>
            <div class='feature-item'><b>Workflow Diagnostics</b><br/>Build report, source table, and score transparency for each answer.</div>
            <div class='feature-item'><b>Grounded Generation</b><br/>Answers restricted to retrieved context to limit hallucination.</div>
        </div>
        """,
        unsafe_allow_html=True
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


build_clicked = st.button("Build / Refresh Multimodal Index", use_container_width=True)

if build_clicked:
    if not jina_key:
        st.error("Add Jina API key before building the index.")
    elif not (txt_files or img_files):
        st.error("Upload at least one document or image.")
    elif overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        signature = _files_signature(txt_files or [], img_files or [], chunk_size, overlap)

        if signature == st.session_state.kb_signature and st.session_state.kb_ready:
            st.info("Index already matches current files and chunk settings.")
        else:
            build_start = time.time()
            with st.spinner("Building multimodal index..."):
                chunks = []
                metadata = []

                text_file_count = 0
                image_file_count = 0
                text_chunks_count = 0
                image_chunks_count = 0

                try:
                    for text_file in txt_files or []:
                        text_file.seek(0)
                        raw_text = _safe_extract_text(text_file)
                        text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                        if text_chunks:
                            text_file_count += 1
                        text_chunks_count += len(text_chunks)
                        chunks.extend(text_chunks)
                        metadata.extend(
                            [{"type": "text", "source": text_file.name} for _ in text_chunks]
                        )
                except Exception as e:
                    st.error(f"Failed to parse document: {e}")
                    st.stop()

                if img_files:
                    if not groq_key:
                        st.error("Add Groq API key to include images in the index.")
                        st.stop()

                    try:
                        for image_file in img_files:
                            image_file.seek(0)
                            image_bytes = image_file.read()
                            vision_text = describe_image(image_bytes, groq_key)
                            if vision_text:
                                image_file_count += 1
                                image_chunks_count += 1
                                chunks.append("Image analysis: " + vision_text)
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

                elapsed = round(time.time() - build_start, 2)
                report = {
                    "text_files": text_file_count,
                    "image_files": image_file_count,
                    "text_chunks": text_chunks_count,
                    "image_chunks": image_chunks_count,
                    "total_chunks": len(chunks),
                    "build_time": elapsed,
                }

                st.session_state.kb_chunks = chunks
                st.session_state.kb_metadata = metadata
                st.session_state.kb_retriever = retriever
                st.session_state.kb_ready = True
                st.session_state.kb_signature = signature
                st.session_state.kb_build_report = report

                st.success(f"Index ready: {len(chunks)} chunks across text and image modalities.")


if st.session_state.kb_ready:
    chunks = st.session_state.kb_chunks
    metadata = st.session_state.kb_metadata
    retriever = st.session_state.kb_retriever
    report = st.session_state.kb_build_report

    a, b, c, d = st.columns(4)
    a.metric("Text Files", report.get("text_files", 0))
    b.metric("Image Files", report.get("image_files", 0))
    c.metric("Indexed Chunks", report.get("total_chunks", 0))
    d.metric("Build Time (s)", report.get("build_time", 0))

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2) Query and Generate</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Ask a question",
        placeholder="Example: Summarize what the chart in the image suggests about quarterly revenue."
    )

    run = st.button("Run Multimodal Workflow", use_container_width=True)
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

        selected_filter = None if retrieval_mode == "all" else retrieval_mode
        results = retriever.search(query_emb, top_k=top_k, filter_type=selected_filter)

        if not results:
            st.warning("No chunks matched the selected retrieval scope.")
            st.stop()

        retrieved_docs = [chunks[item["id"]] for item in results]
        distances = [item["score"] for item in results]

        reranked = hybrid_rerank_with_scores(
            query=query,
            docs=retrieved_docs,
            distances=distances,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight
        )

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
                    "vector_distance": round(base["score"], 4),
                    "semantic_score": round(row["semantic_score"], 4),
                    "lexical_score": row["lexical_score"],
                    "hybrid_score": round(row["hybrid_score"], 4)
                }
            )

        context = "\n\n".join(context_parts)

        style_instruction = "Provide a concise answer in bullet points." if response_style == "Concise" else "Provide a detailed answer with clear reasoning steps."
        styled_query = f"{query}\n\nOutput style requirement: {style_instruction}"

        try:
            answer = ask_llm(context, styled_query, groq_key, model)
        except Exception as e:
            st.error(f"Answer generation failed: {e}")
            st.stop()

        latency = round(time.time() - start, 2)
        modalities_used = sorted({row["type"] for row in source_rows})

        st.session_state.history.append((query, answer))

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>3) Workflow Output</div>", unsafe_allow_html=True)
        st.write(answer)

        status_class = "status-ok" if len(modalities_used) > 1 else "status-warn"
        modality_text = ", ".join(modalities_used)

        st.markdown(
            f"""
            <div class='pill-row'>
                <span class='pill'>Response style: {response_style}</span>
                <span class='pill'>Retrieval scope: {retrieval_mode}</span>
                <span class='pill'>Top-K: {top_k}</span>
                <span class='pill'>Context used: {len(selected)}</span>
                <span class='pill {status_class}'>Modalities used: {modality_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency (s)", latency)
        m2.metric("Retrieved Chunks", len(results))
        m3.metric("Context Chunks", len(selected))
        m4.metric("Sources Used", len({row['source'] for row in source_rows}))

        with st.expander("Feature Trace"):
            st.markdown("- Multimodal indexing: text + image descriptions")
            st.markdown(f"- Hybrid reranking: semantic={semantic_weight:.2f}, lexical={lexical_weight:.2f}")
            st.markdown("- Guardrail prompting: context-only answering")
            st.markdown(f"- Answer style control: {response_style}")

        with st.expander("Source Diagnostics"):
            st.dataframe(source_rows, use_container_width=True, hide_index=True)

        with st.expander("Retrieved Context"):
            st.text(context)

        with st.expander("Recent Chat History"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"Question: {q}")
                st.markdown(f"Answer: {a}")
                st.divider()

        chat_export = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.history])
        st.download_button(
            "Download Chat History",
            data=chat_export.encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
else:
    st.info("Upload files, add API keys, and build the multimodal index to begin.")
