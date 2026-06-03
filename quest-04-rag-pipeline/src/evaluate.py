"""
Evaluation script for the RAG pipeline.

Uses ragas to compute retrieval and generation metrics on a set of
query-answer pairs from SQuAD.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from retrieve import Retriever
from data_utils import load_documents, load_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"


def evaluate_retrieval(
    retriever: Retriever,
    queries: List[Dict],
    documents: List[Dict],
    initial_k: int = 10,
    final_k: int = 5,
    max_queries: int = 100,
) -> Dict:
    """Evaluate retrieval performance using hit rate and MRR.

    Metrics:
        - Hit Rate @ K: Proportion of queries where the correct document is in the top-K
        - MRR @ K: Mean Reciprocal Rank — average of 1/rank of the first correct document
        - Precision @ K: Proportion of retrieved documents that are relevant
    """
    # Build a map from doc_id to text for checking relevance
    doc_text_map = {doc["id"]: doc["text"] for doc in documents}

    # Limit to max_queries for speed
    eval_queries = queries[:max_queries]
    logger.info(f"Evaluating on {len(eval_queries)} queries...")

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    reciprocal_ranks = []
    precisions_at_5 = []
    retrieval_times = []

    for q in eval_queries:
        question = q["question"]
        expected_doc_id = q["context_id"]
        expected_text = doc_text_map.get(expected_doc_id, "")

        # Retrieve
        t0 = time.time()
        results = retriever.retrieve(question, k=initial_k)
        retrieval_times.append(time.time() - t0)

        # Check if the correct context is in the results
        retrieved_texts = [r["text"] for r in results]

        # Use a simple overlap check — if the expected text is retrieved
        # We can't just compare doc_ids since we may have chunked differently
        # So we check if the expected text appears within any retrieved text
        found_indices = []
        for i, r_text in enumerate(retrieved_texts):
            # Check if the expected text is contained in the retrieved text
            # or vice versa (since SQuAD contexts are used directly as documents)
            if expected_text in r_text or r_text in expected_text:
                found_indices.append(i)

        # Hit rate
        if any(i < 1 for i in found_indices):
            hit_at_1 += 1
        if any(i < 5 for i in found_indices):
            hit_at_5 += 1
        if any(i < 10 for i in found_indices):
            hit_at_10 += 1

        # MRR
        if found_indices:
            best_rank = min(found_indices) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / best_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Precision @ 5
        relevant_at_5 = sum(1 for i in found_indices if i < 5)
        precisions_at_5.append(relevant_at_5 / min(5, len(results)))

    n = len(eval_queries)
    metrics = {
        "num_queries": n,
        "hit_rate@1": float(hit_at_1 / n),
        "hit_rate@5": float(hit_at_5 / n),
        "hit_rate@10": float(hit_at_10 / n),
        "mrr": float(np.mean(reciprocal_ranks)),
        "precision@5": float(np.mean(precisions_at_5)),
        "avg_retrieval_time_ms": float(np.mean(retrieval_times) * 1000),
    }

    logger.info(
        f"\n{'=' * 60}\n"
        f"Retrieval Evaluation Results (n={n}):\n"
        f"  Hit Rate @1:  {metrics['hit_rate@1']:.3f}\n"
        f"  Hit Rate @5:  {metrics['hit_rate@5']:.3f}\n"
        f"  Hit Rate @10: {metrics['hit_rate@10']:.3f}\n"
        f"  MRR:          {metrics['mrr']:.3f}\n"
        f"  Precision@5:  {metrics['precision@5']:.3f}\n"
        f"  Avg Time:     {metrics['avg_retrieval_time_ms']:.1f} ms\n"
        f"{'=' * 60}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retriever performance")
    parser.add_argument("--use-sample", action="store_true", help="Use sample dataset")
    parser.add_argument(
        "--max-queries", type=int, default=100, help="Max queries to evaluate"
    )
    parser.add_argument(
        "--initial-k", type=int, default=10, help="Documents to retrieve"
    )
    parser.add_argument(
        "--final-k", type=int, default=5, help="Documents after re-rank"
    )
    args = parser.parse_args()

    # Determine which dataset to use
    if args.use_sample:
        doc_path = SAMPLE_DIR / "documents_sample.json"
        query_path = SAMPLE_DIR / "queries_sample.json"
    else:
        doc_path = DATA_DIR / "documents.json"
        query_path = DATA_DIR / "queries.json"

    for p in [doc_path, query_path]:
        if not p.exists():
            logger.error(f"File not found: {p}\nRun 'python data/download.py' first.")
            sys.exit(1)

    documents = load_documents(str(doc_path))
    queries = load_queries(str(query_path))

    # Load retriever
    retriever = Retriever()

    # Evaluate
    metrics = evaluate_retrieval(
        retriever,
        queries,
        documents,
        initial_k=args.initial_k,
        final_k=args.final_k,
        max_queries=args.max_queries,
    )

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
