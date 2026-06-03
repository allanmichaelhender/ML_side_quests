"""
Data utilities for the RAG pipeline.
Handles loading documents and queries from SQuAD and other sources.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_documents(path: str) -> List[Dict[str, str]]:
    """Load documents from a JSON file.

    Expected format: list of dicts with keys "id" and "text".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents


def load_queries(path: str) -> List[Dict]:
    """Load query-answer pairs from a JSON file.

    Expected format: list of dicts with keys "question", "answer", "context_id".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries from {path}")
    return queries


def save_documents(documents: List[Dict[str, str]], path: str) -> None:
    """Save documents to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(documents)} documents to {path}")


def save_queries(queries: List[Dict], path: str) -> None:
    """Save queries to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(queries)} queries to {path}")


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
    doc_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Split a long document into overlapping chunks.

    Each chunk gets an id like '{doc_id}_chunk_{i}'.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = f"{doc_id}_chunk_{chunk_idx}" if doc_id else f"chunk_{chunk_idx}"
        chunks.append({"id": chunk_id, "text": chunk_text})
        chunk_idx += 1
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def extract_documents_from_squad(dataset) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """Extract unique context passages and query-answer pairs from a SQuAD dataset.

    Returns:
        (documents, queries)
        - documents: list of {"id": str, "text": str}
        - queries: list of {"question": str, "answer": str, "context_id": str}
    """
    seen_contexts: Dict[str, str] = {}  # context_text -> doc_id
    documents: List[Dict[str, str]] = []
    queries: List[Dict] = []

    for example in dataset:
        context = example["context"]
        question = example["question"]

        # Extract the first answer if available
        answers = example.get("answers", {})
        answer_texts = answers.get("text", [])
        answer = answer_texts[0] if answer_texts else ""

        # Skip if no answer and not answerable (SQuAD 2.0)
        if not answer and not example.get("is_impossible", False):
            continue

        # Deduplicate contexts
        if context not in seen_contexts:
            doc_id = f"doc_{len(documents):05d}"
            seen_contexts[context] = doc_id
            documents.append({"id": doc_id, "text": context})

        context_id = seen_contexts[context]
        queries.append(
            {
                "question": question,
                "answer": answer,
                "context_id": context_id,
            }
        )

    logger.info(
        f"Extracted {len(documents)} unique documents and "
        f"{len(queries)} queries from SQuAD"
    )
    return documents, queries
