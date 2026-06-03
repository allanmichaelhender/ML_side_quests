"""
Download script for the RAG pipeline.

Downloads SQuAD v2 dataset from Hugging Face, extracts unique context
passages as the document corpus, and saves everything locally.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.data_utils import extract_documents_from_squad, save_documents, save_queries

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
SAMPLE_DIR = DATA_DIR / "sample"


def download_squad(subset_size: int = 500) -> None:
    """Download SQuAD v2, extract documents and queries."""
    logger.info("Loading SQuAD v2 dataset from Hugging Face...")
    dataset = load_dataset("rajpurkar/squad_v2", split="train")

    logger.info(f"Dataset has {len(dataset)} examples total")

    # Extract documents and queries
    documents, queries = extract_documents_from_squad(dataset)

    # Save full set
    save_documents(documents, DATA_DIR / "documents.json")
    save_queries(queries, DATA_DIR / "queries.json")

    # Save a subset for quick testing
    n_sample = min(subset_size, len(documents))
    sample_docs = documents[:n_sample]
    sample_queries = [
        q for q in queries if q["context_id"] in {d["id"] for d in sample_docs}
    ]

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    save_documents(sample_docs, SAMPLE_DIR / "documents_sample.json")
    save_queries(sample_queries, SAMPLE_DIR / "queries_sample.json")

    logger.info(
        f"\n{'=' * 60}\n"
        f"Download complete!\n"
        f"  Full set:  {len(documents)} documents, {len(queries)} queries\n"
        f"  Sample:    {len(sample_docs)} documents, {len(sample_queries)} queries\n"
        f"  Saved to:  {DATA_DIR}\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SQuAD dataset for RAG pipeline"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=500,
        help="Number of sample documents to keep (default: 500)",
    )
    args = parser.parse_args()

    download_squad(subset_size=args.subset_size)
