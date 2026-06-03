"""
Retriever module for the RAG pipeline.

Provides the Retriever class that loads a pre-built FAISS index and
supports dense retrieval + cross-encoder re-ranking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

# Default paths relative to the results directory
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "results" / "faiss_index"

# Re-ranker model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    """Dense retriever with optional cross-encoder re-ranking."""

    def __init__(
        self,
        index_dir: str = str(DEFAULT_INDEX_DIR),
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name: Optional[str] = RERANKER_MODEL,
    ):
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name

        # Lazy-loaded attributes
        self._index: Optional[faiss.Index] = None
        self._documents: Optional[List[Dict]] = None
        self._doc_map: Optional[Dict[int, Dict]] = None
        self._embed_model: Optional[SentenceTransformer] = None
        self._rerank_model: Optional[CrossEncoder] = None
        self._dimension: Optional[int] = None

        self._load_index()

    def _load_index(self) -> None:
        """Load the FAISS index and document metadata."""
        index_path = self.index_dir / "index.faiss"
        doc_path = self.index_dir / "documents.json"
        model_info_path = self.index_dir / "model_info.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run 'python src/embed_and_index.py' first."
            )
        if not doc_path.exists():
            raise FileNotFoundError(f"Document metadata not found at {doc_path}")

        self._index = faiss.read_index(str(index_path))
        self._dimension = self._index.d

        with open(doc_path, "r", encoding="utf-8") as f:
            self._documents = json.load(f)

        # Build a map from index position to document
        self._doc_map = {doc["index_position"]: doc for doc in self._documents}

        # Load model info
        if model_info_path.exists():
            with open(model_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self.embed_model_name = info.get("model_name", self.embed_model_name)

        logger.info(
            f"Loaded FAISS index with {len(self._documents)} documents "
            f"(dim={self._dimension})"
        )

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    @property
    def rerank_model(self) -> Optional[CrossEncoder]:
        if self.rerank_model_name and self._rerank_model is None:
            logger.info(f"Loading cross-encoder re-ranker: {self.rerank_model_name}")
            self._rerank_model = CrossEncoder(self.rerank_model_name)
        return self._rerank_model

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> List[Dict]:
        """Retrieve the top-k most similar documents for a query.

        Args:
            query: The query text.
            k: Number of documents to retrieve.

        Returns:
            List of dicts with keys: "doc_id", "text", "score".
        """
        # Embed the query
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._doc_map.get(int(idx), {})
            results.append(
                {
                    "doc_id": doc.get("doc_id", str(idx)),
                    "text": doc.get("text", ""),
                    "score": float(score),
                }
            )

        return results

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Re-rank retrieved documents using a cross-encoder.

        Args:
            query: The original query.
            documents: List of documents (from retrieve()).
            top_k: Number of top documents to return after re-ranking.

        Returns:
            Re-ranked list of dicts with added "rerank_score".
        """
        if self.rerank_model is None:
            logger.warning("No re-ranker model loaded, returning original order")
            return documents[:top_k]

        if not documents:
            return []

        # Prepare pairs for the cross-encoder
        pairs = [[query, doc["text"]] for doc in documents]

        # Get relevance scores
        scores = self.rerank_model.predict(pairs, show_progress_bar=False)

        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return sorted_docs[:top_k]

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 10,
        final_k: int = 5,
    ) -> Dict:
        """Full pipeline: retrieve then re-rank.

        Args:
            query: The query text.
            initial_k: Number of documents to retrieve before re-ranking.
            final_k: Number of documents to return after re-ranking.

        Returns:
            Dict with keys:
                - "query": the original query
                - "initial_results": results from dense retrieval
                - "final_results": results after cross-encoder re-ranking
                - "num_documents": total documents in the index
        """
        # Step 1: Dense retrieval
        initial_results = self.retrieve(query, k=initial_k)

        # Step 2: Cross-encoder re-ranking
        final_results = self.rerank(query, initial_results, top_k=final_k)

        return {
            "query": query,
            "initial_results": initial_results,
            "final_results": final_results,
            "num_documents": len(self._documents) if self._documents else 0,
        }

    @property
    def num_documents(self) -> int:
        return len(self._documents) if self._documents else 0
