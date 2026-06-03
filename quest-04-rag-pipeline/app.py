"""
Streamlit app for the RAG Pipeline demo.

Allows users to query the document corpus and see retrieved passages
with relevance scores, before and after cross-encoder re-ranking.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve import Retriever

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Pipeline — Quest 04",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────

INDEX_DIR = PROJECT_ROOT / "results" / "faiss_index"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

SAMPLE_QUESTIONS = [
    "What is the Transformer architecture?",
    "How do Convolutional Neural Networks work?",
    "What is transfer learning?",
    "How does reinforcement learning differ from supervised learning?",
    "What is Retrieval-Augmented Generation?",
    "What is BERT and how is it pre-trained?",
    "How does gradient descent work in machine learning?",
    "What is the attention mechanism in neural networks?",
]

# ── Session state ────────────────────────────────────────────────────────────

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🔍 RAG Pipeline")
st.sidebar.markdown("**Quest 04** — Retrieval-Augmented Generation")

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")

initial_k = st.sidebar.slider(
    "Documents to retrieve (initial)",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="Number of documents retrieved by dense embedding search before re-ranking",
)

final_k = st.sidebar.slider(
    "Documents after re-ranking",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Number of top documents kept after cross-encoder re-ranking",
)

use_rerank = st.sidebar.checkbox("Use cross-encoder re-ranking", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    """
This RAG pipeline demonstrates:
- **Dense retrieval** with MiniLM embeddings + FAISS
- **Cross-encoder re-ranking** for improved precision
- **Explainable retrieval** — see scores and document text

**Data**: SQuAD v2 context passages
**Embedding**: `all-MiniLM-L6-v2`
**Re-ranker**: `ms-marco-MiniLM-L-6-v2`
"""
)

# ── Load retriever ───────────────────────────────────────────────────────────


def load_retriever():
    """Load the retriever (with spinner feedback)."""
    if not INDEX_DIR.exists():
        st.error(
            f"FAISS index not found at `{INDEX_DIR}`.\n\n"
            "Please run the following steps:\n"
            "1. `python data/download.py` — download the SQuAD dataset\n"
            "2. `python src/embed_and_index.py` — build the FAISS index"
        )
        return None

    try:
        retriever = Retriever(str(INDEX_DIR))
        return retriever
    except Exception as e:
        st.error(f"Failed to load retriever: {e}")
        return None


# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("🔍 RAG Retrieval Pipeline")
st.markdown(
    "Ask a question and see how the retriever finds relevant passages from the "
    "document corpus using dense embeddings + cross-encoder re-ranking."
)

# Load retriever on first interaction
if not st.session_state.index_loaded:
    with st.spinner("Loading FAISS index and embedding models..."):
        retriever = load_retriever()
        if retriever is not None:
            st.session_state.retriever = retriever
            st.session_state.index_loaded = True
            st.rerun()
else:
    retriever = st.session_state.retriever

# Show document count if loaded
if st.session_state.index_loaded and retriever:
    st.info(
        f"📚 **{retriever.num_documents:,} documents** in the index • dim={retriever._dimension}"
    )

# Quick sample questions
if st.session_state.index_loaded:
    cols = st.columns(4)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 4]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.query = q

# Query input
query = st.text_input(
    "**Enter your question:**",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What is the Transformer architecture?",
    key="query_input",
)

# ── Search button ────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

if search_clicked and query and st.session_state.index_loaded:
    with st.spinner("Retrieving and re-ranking..."):
        t0 = time.time()

        # Step 1: Dense retrieval
        initial_results = retriever.retrieve(query, k=initial_k)
        t_retrieve = time.time() - t0

        # Step 2: Re-ranking
        if use_rerank and initial_results:
            final_results = retriever.rerank(query, initial_results, top_k=final_k)
        else:
            final_results = initial_results[:final_k]
        t_total = time.time() - t0

    # Store in history
    st.session_state.query_history.append(
        {
            "query": query,
            "initial_results": initial_results,
            "final_results": final_results,
            "retrieval_time": t_retrieve,
            "total_time": t_total,
        }
    )

    # ── Results display ────────────────────────────────────────────────────────

    st.markdown("---")

    # Metrics row
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    with met_col1:
        st.metric("Retrieval time", f"{t_retrieve * 1000:.1f} ms")
    with met_col2:
        st.metric("Total time", f"{t_total * 1000:.1f} ms")
    with met_col3:
        st.metric("Initial candidates", len(initial_results))
    with met_col4:
        st.metric("Final results", len(final_results))

    st.markdown("---")

    # Display final results
    st.subheader(f"🏆 Top {len(final_results)} Results")

    for rank, doc in enumerate(final_results, 1):
        score = doc.get("rerank_score", doc.get("score", 0))
        score_type = "Re-rank" if "rerank_score" in doc else "Cosine"

        with st.expander(
            f"**#{rank}** — [{score_type} Score: {score:.4f}]  "
            f"`{doc.get('doc_id', 'unknown')}`",
            expanded=(rank <= 3),
        ):
            st.markdown(doc.get("text", ""))

            # Show initial retrieval score too
            if "rerank_score" in doc:
                st.caption(
                    f"Initial cosine similarity: {doc.get('score', 0):.4f}  •  "
                    f"Re-rank score: {doc['rerank_score']:.4f}"
                )

    # Optional: show all initial results
    if use_rerank and len(initial_results) > len(final_results):
        with st.expander(
            f"📋 View all {len(initial_results)} initial retrieval results"
        ):
            for rank, doc in enumerate(initial_results, 1):
                st.markdown(
                    f"**#{rank}** — Score: `{doc['score']:.4f}`  •  `{doc.get('doc_id', '')}`"
                )
                st.markdown(
                    f"{doc.get('text', '')[:200]}..."
                    if len(doc.get("text", "")) > 200
                    else doc.get("text", "")
                )
                if rank < len(initial_results):
                    st.markdown("---")

elif search_clicked and not st.session_state.index_loaded:
    st.warning(
        "Please build the index first by running `python src/embed_and_index.py`"
    )

# ── Query history ────────────────────────────────────────────────────────────

if st.session_state.query_history:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Recent Queries")
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            q = entry["query"][:50] + ("..." if len(entry["query"]) > 50 else "")
            st.markdown(f"**{len(st.session_state.query_history) - i}.** {q}")
            st.caption(
                f"{entry['total_time'] * 1000:.0f}ms • {len(entry['final_results'])} results"
            )
