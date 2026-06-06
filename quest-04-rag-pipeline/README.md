# 🔍 Quest 04 — RAG Retrieval Pipeline

**Domain**: Retrieval-Augmented Generation  
**Model**: MiniLM (`all-MiniLM-L6-v2`) + Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)  
**Vector Store**: FAISS (cosine similarity)

## Problem

Build a retrieval pipeline that can find relevant passages from a large document corpus given a natural language query. This is the retrieval half of a RAG system — the retrieved passages can then be fed to an LLM for answer generation.

## Pipeline

```
User Query
    │
    ▼
┌──────────────────┐
│  Query Embedding │  (MiniLM sentence transformer)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   FAISS Search   │  (cosine similarity — top-K)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Cross-Encoder    │  (re-rank top candidates)
│   Re-ranking     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Top-N Results   │  (final retrieved passages)
└──────────────────┘
```

## Data

Uses the **SQuAD v2** dataset from Hugging Face. Unique context passages become the document corpus (~19k documents). Each passage has associated question-answer pairs for evaluation.

## Getting Started

### 1. Download the data

```bash
cd quest-04-rag-pipeline
python data/download.py
```

This downloads SQuAD v2, extracts unique context passages as documents, and saves a sample subset. Results:

- `data/documents.json` — full document corpus
- `data/queries.json` — query-answer pairs for evaluation
- `data/sample/documents_sample.json` — 500-doc sample for quick testing
- `data/sample/queries_sample.json` — sample query pairs

### 2. Build the index

```bash
# Full dataset
python src/embed_and_index.py

# Or with sample for quick testing
python src/embed_and_index.py --use-sample
```

Downloads MiniLM, embeds all documents, builds a FAISS index, and saves to `results/faiss_index/`.

### 3. Run the Streamlit demo

```bash
streamlit run app.py
```

### 4. Evaluate

```bash
python src/evaluate.py
```

## Results

| Metric              | Value                                   |
| ------------------- | --------------------------------------- |
| Document corpus     | 18,777 SQuAD v2 passages                |
| Query set           | 86,821 QA pairs                         |
| Embedding model     | `all-MiniLM-L6-v2`                      |
| Embedding dimension | 384                                     |
| Re-ranker           | `ms-marco-MiniLM-L-6-v2`                |
| Index type          | FAISS `IndexFlatIP` (cosine similarity) |
| Embedding time      | ~10 min (CPU)                           |
| **Hit Rate @1**     | **42.5%**                               |
| **Hit Rate @5**     | **74.5%**                               |
| **Hit Rate @10**    | **87.0%**                               |
| **MRR**             | **0.567**                               |
| **Avg retrieval**   | **53 ms**                               |

## Key Findings

- Dense retrieval with MiniLM captures semantic similarity well, even with exact keyword mismatches
- Cross-encoder re-ranking significantly improves precision@K (typically +10–20% over raw retrieval)
- FAISS `IndexFlatIP` with L2-normalized embeddings gives cosine similarity search with exact results (no approximation loss)
- For large-scale deployments, consider `IndexIVFFlat` for faster search at the cost of some accuracy

## Potential Improvements

The current pipeline is entirely **zero-shot inference** — both MiniLM and the cross-encoder are used off-the-shelf with no fine-tuning. There are several avenues to improve retrieval performance:

### 1. Fine-tune the embedding model (highest impact)

The 86k query-answer pairs in `data/queries.json` are only used for evaluation. They can be repurposed for **contrastive learning** (e.g. `MultipleNegativesRankingLoss` from `sentence-transformers`) to pull query embeddings closer to their correct documents and push them away from incorrect ones. Estimated gain: **+10–15% Hit Rate@5**.

### 2. Hard negative mining

Use the current retriever to find top-ranked-but-wrong documents per query, then retrain with those as explicit negatives. This forces the model to learn finer-grained distinctions.

### 3. Enable document chunking

A `chunk_text()` function exists in `src/data_utils.py` but is never called. Currently each SQuAD context is stored as a single document, but some are long paragraphs. Chunking into overlapping word windows (256 words, 32 overlap) would improve precision by matching queries to specific passage snippets rather than whole paragraphs.

### 4. Fine-tune the cross-encoder re-ranker

The cross-encoder was trained on MS MARCO (Bing search data). Fine-tuning it on your own query-document pairs — especially with hard negatives from the FAISS index — would make re-ranking scores more discriminative for this corpus.

### 5. Hybrid retrieval (dense + sparse)

Adding a BM25/TF-IDF sparse retriever alongside MiniLM and fusing results captures exact keyword matches that dense retrieval can miss. The SQuAD data is ideal for learning the optimal fusion weight.

### 6. Query expansion

Use the top initial retrieval results to extract key terms and re-query, improving recall without any model changes.

## Files

```
quest-04-rag-pipeline/
├── app.py                    # Streamlit demo
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   ├── download.py           # Download SQuAD v2 dataset
│   └── sample/               # Small sample for quick testing
├── results/
│   ├── faiss_index/          # Built FAISS index + metadata
│   ├── metrics.json          # Evaluation results
│   └── figures/              # Performance visualizations
└── src/
    ├── data_utils.py         # Document/query loading utilities
    ├── embed_and_index.py    # Embed documents + build FAISS index
    ├── retrieve.py           # Retriever class (retrieve + rerank)
    └── evaluate.py           # Retrieval evaluation
```
