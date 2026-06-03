"""
Embed documents and build a FAISS index for retrieval.

This is the main entry point (also referenced in Dockerfile CMD).
Usage:
    python src/embed_and_index.py                         # full dataset
    python src/embed_and_index.py --use-sample            # sample subset
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_utils import load_documents, save_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "model"
INDEX_DIR = RESULTS_DIR / "faiss_index"

# Default model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_dirs():
    """Create output directories."""
    for d in [MODEL_DIR, INDEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a sentence transformer model."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model dimension: {model.get_embedding_dimension()}")
    return model


def embed_documents(
    model: SentenceTransformer,
    documents: List[dict],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed all documents and return a numpy array of embeddings."""
    texts = [doc["text"] for doc in documents]
    logger.info(f"Embedding {len(texts)} documents with batch size {batch_size}...")

    iterator = (
        tqdm(range(0, len(texts), batch_size), desc="Embedding")
        if show_progress
        else range(0, len(texts), batch_size)
    )

    all_embeddings = []
    for i in iterator:
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings)

    # Normalize for cosine similarity search
    faiss.normalize_L2(embeddings)

    logger.info(f"Embedding shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index (inner product = cosine similarity for normalized vectors)."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors, dimension {dimension}")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    documents: List[dict],
    index_dir: str,
    model_name: str,
) -> None:
    """Save the FAISS index and document metadata."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(index_dir / "index.faiss"))
    logger.info(f"Index saved to {index_dir / 'index.faiss'}")

    # Save document metadata with their index positions
    metadata = []
    for i, doc in enumerate(documents):
        metadata.append(
            {
                "index_position": i,
                "doc_id": doc["id"],
                "text": doc["text"],
            }
        )

    with open(index_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save model info
    with open(index_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "dimension": index.d,
                "num_documents": len(documents),
            },
            f,
            indent=2,
        )

    logger.info(f"Metadata saved to {index_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed documents and build FAISS index"
    )
    parser.add_argument("--use-sample", action="store_true", help="Use sample dataset")
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size"
    )
    args = parser.parse_args()

    ensure_dirs()

    # Determine which documents to use
    if args.use_sample:
        doc_path = SAMPLE_DIR / "documents_sample.json"
        logger.info(f"Using SAMPLE documents from {doc_path}")
    else:
        doc_path = DATA_DIR / "documents.json"
        logger.info(f"Using FULL documents from {doc_path}")

    if not doc_path.exists():
        logger.error(
            f"Documents file not found: {doc_path}\n"
            f"Run 'python data/download.py' first to download the dataset."
        )
        sys.exit(1)

    # Load documents
    documents = load_documents(str(doc_path))

    # Load model
    model = load_embedding_model(args.model)

    # Embed
    t0 = time.time()
    embeddings = embed_documents(model, documents, batch_size=args.batch_size)
    t_embed = time.time() - t0
    logger.info(
        f"Embedding took {t_embed:.2f}s ({t_embed / len(documents):.2f}s per doc)"
    )

    # Build index
    t0 = time.time()
    index = build_faiss_index(embeddings)
    t_index = time.time() - t0
    logger.info(f"Indexing took {t_index:.2f}s")

    # Save
    save_index_and_metadata(index, documents, str(INDEX_DIR), args.model)

    logger.info(
        f"\n{'=' * 60}\n"
        f"Embedding and indexing complete!\n"
        f"  Documents: {len(documents)}\n"
        f"  Embedding dim: {embeddings.shape[1]}\n"
        f"  Time: {t_embed + t_index:.2f}s total\n"
        f"  Index saved to: {INDEX_DIR}\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
