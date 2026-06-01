# ML Side Quests

A collection of six industry-focused ML projects demonstrating breadth across computer vision, NLP, generative AI, retrieval, and reinforcement learning.

---

## Quest 1 — Computer Vision: Plant Disease Classification

### Data Source

- **Dataset**: PlantVillage Dataset via [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 54,000+ labelled leaf images across 38 disease/healthy classes
- Covers 14 crop species (tomato, potato, corn, grape, etc.)
- Pre-split into train/val sets, ready to use with no labelling required

### Model

- **Architecture**: MobileNetV2 pretrained on ImageNet
- Chosen for its CPU-friendly inference profile (depthwise separable convolutions)
- Final classification head replaced with a 38-class dense layer

### Pipeline & Training Steps

1. Download and extract PlantVillage from Kaggle
2. Apply data augmentation: random horizontal/vertical flips, rotation (±15°), brightness/contrast jitter
3. Normalise to ImageNet mean/std (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`)
4. Freeze all MobileNetV2 backbone layers; train classification head only for 5 epochs (transfer learning phase)
5. Unfreeze the last 2 convolutional blocks; fine-tune end-to-end at a lower learning rate (1e-4) for a further 10 epochs
6. Evaluate on held-out validation set: accuracy, per-class precision/recall, confusion matrix
7. Generate Grad-CAM visualisations to show which leaf regions activate the model's decision
8. Export model to ONNX for lightweight inference in Streamlit demo

### Hardware Note

Trains on CPU in approximately 30–60 minutes with frozen backbone. Full fine-tune phase adds ~1–2 hours.

---

## Quest 2 — NLP: Sentiment Analysis with DistilBERT

### Data Source

- **Dataset**: Amazon Reviews (Multi-Domain) via [Hugging Face Datasets](https://huggingface.co/datasets/amazon_reviews_multi)
- Millions of product reviews across categories, labelled with 1–5 star ratings
- Stars mapped to 3 classes: Negative (1–2), Neutral (3), Positive (4–5)
- Subsample of 10,000 reviews used for CPU-feasible fine-tuning

### Model

- **Architecture**: DistilBERT (`distilbert-base-uncased`)
- 40% smaller and 60% faster than BERT-base with ~97% of its performance
- Sequence classification head added for 3-class output

### Pipeline & Training Steps

1. Load dataset via `datasets.load_dataset('amazon_reviews_multi', 'en')`
2. Map star ratings to 3-class labels; stratified sample to 10k rows (balanced classes)
3. Tokenise with `DistilBertTokenizerFast`, max length 256 tokens
4. Fine-tune using Hugging Face `Trainer` API: 3 epochs, batch size 16, AdamW optimiser, linear warmup scheduler
5. Evaluate: accuracy, macro F1, per-class precision/recall, confusion matrix
6. Visualise attention weights on example reviews to show which tokens drive predictions
7. Build Streamlit demo: text input → sentiment label + confidence score + attention heatmap

### Hardware Note

Fine-tuning on 10k samples runs in approximately 1–2 hours on CPU.

---

## Quest 3 — Generative AI: Energy Earnings Call Analyst

### Data Source

- **Primary**: SEC EDGAR full-text search — 8-K filings (Item 2.02: Results of Operations)
  - Companies: ExxonMobil (XOM), Chevron (CVX), Shell (SHEL), BP (BP)
  - Timeframe: 2018–2024 (approx. 24–28 transcripts per company)
  - Access via EDGAR API: `https://efts.sec.gov/LATEST/search-index?q=%22earnings+call%22&dateRange=custom`
- **Silver labelling**: GPT-4o API used to generate sentiment + claim labels on parsed transcript chunks (this becomes the fine-tuning training set)
- **Reference datasets**: Optionally supplement with [EarningsCall dataset on Hugging Face](https://huggingface.co/datasets/lamini/earnings-calls-qa) for additional examples

### Model

- **Architecture**: TinyLlama-1.1B or Phi-2 (2.7B) with LoRA adapters via PEFT
- LoRA config: rank=16, alpha=32, target modules = `q_proj`, `v_proj`
- Quantised to 4-bit (GGUF via `llama.cpp`) for CPU-feasible fine-tuning

### Pipeline & Training Steps

1. Scrape and parse 8-K filings from EDGAR: extract earnings call transcript text, segment by speaker turn
2. Chunk transcripts into ~300-token sections (CEO prepared remarks, CFO financials, Q&A)
3. **Silver labelling pass**: prompt GPT-4o to label each chunk with:
   - Sentiment: `bullish` / `neutral` / `bearish`
   - Extracted claims: guidance figures, capex commitments, production targets, forward-looking statements (structured JSON)
4. Format labelled chunks into instruction-tuning format: `[INST] Analyse this earnings call excerpt... [/INST] {"sentiment": ..., "claims": [...]}`
5. Fine-tune with LoRA using Hugging Face `trl` SFTTrainer: 3 epochs, batch size 4, gradient accumulation 8
6. Evaluate: sentiment classification F1 vs GPT-4o baseline; claim extraction precision/recall
7. Build Streamlit demo: paste transcript excerpt → model returns structured JSON with sentiment + claims table

### Hardware Note

LoRA fine-tuning on TinyLlama with 4-bit quantisation runs on CPU overnight (6–10 hours). Phi-2 adds approximately 2–3 hours.

---

## Quest 4 — RAG: Retrieval Evaluation Pipeline

### Data Source

- **Corpus**: arXiv abstracts via [Hugging Face Datasets](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) or direct arXiv bulk access
  - Focus on a single domain (e.g. ML papers, cs.LG + cs.AI) for coherent retrieval
  - ~50,000–100,000 abstracts for a manageable but realistic corpus size
- **QA pairs for evaluation**: Synthetically generated — use GPT-4o to create question/answer pairs grounded in specific abstracts (this becomes the eval benchmark)

### Models

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — fast, CPU-friendly, strong retrieval performance
- **Vector store**: FAISS (local) or ChromaDB (persistent)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking top-K retrieved chunks
- **Generation**: Claude or GPT-4o API (offloaded — no local GPU needed)

### Pipeline & Training Steps

1. Download and preprocess arXiv abstracts; chunk into ~200-token passages with 20-token overlap
2. Embed all passages using `sentence-transformers`; index into FAISS
3. Generate synthetic eval set: 200–300 question/answer pairs grounded in specific passages (GPT-4o)
4. **Baseline retrieval**: BM25 sparse retrieval (keyword-based) — establish baseline metrics
5. **Dense retrieval**: sentence-transformer embeddings + FAISS ANN search — compare to BM25
6. **Reranking**: retrieve top-20 with dense retrieval, rerank to top-5 with cross-encoder
7. **Hybrid retrieval**: combine BM25 + dense scores (RRF fusion) — compare all four approaches
8. Evaluate all pipeline variants across: Recall@K (K=1,3,5,10), MRR, NDCG, Context Precision (RAGAS)
9. Visualise retrieval quality metrics in a comparison dashboard (Streamlit)
10. Add generation layer: retrieved context → LLM answer → faithfulness score (RAGAS)

### Hardware Note

Embedding generation and FAISS indexing run entirely on CPU. Generation is API-based. No GPU required.

---

## Quest 5 — NLP: Support Ticket Routing

### Data Source

- **Primary**: [Kaggle Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 3M+ real support interactions with company/topic labels
- **Secondary**: GitHub Issues public dump via GH Archive (`https://www.gharchive.org/`) — real bug reports and feature requests with label tags
- Target: 6–8 routing categories (e.g. billing, technical, account, shipping, returns, general)

### Models

- **Baseline**: TF-IDF vectoriser + Logistic Regression (scikit-learn)
- **ML model**: DistilBERT fine-tuned classifier (`distilbert-base-uncased`)
- **LLM zero-shot**: GPT-4o or Claude with a classification prompt — no fine-tuning, compare out of the box

### Pipeline & Training Steps

1. Download and clean dataset; map existing labels to 6–8 unified routing categories
2. Exploratory analysis: class distribution, average token length, common terms per class
3. **Baseline**: fit TF-IDF (unigram + bigram, max 20k features) + Logistic Regression; evaluate on held-out test set
4. **DistilBERT fine-tune**: tokenise with `DistilBertTokenizerFast`, fine-tune classification head for 3 epochs using Hugging Face `Trainer`
5. **Zero-shot LLM**: send each test ticket to GPT-4o with a structured classification prompt; record predictions
6. Compare all three approaches: accuracy, macro F1, per-class precision/recall, latency, cost (for LLM)
7. Generate confusion matrices for all three models side by side
8. Build Streamlit demo: ticket text input → routing prediction with confidence from all three models

### Hardware Note

Baseline trains in seconds. DistilBERT fine-tune on 5k samples runs in under 30 minutes on CPU.

---

## Quest 6 — Reinforcement Learning: Energy Grid Load Balancing Agent

### Data Source

- **Demand curves**: EIA (US Energy Information Administration) hourly electricity demand by region — [EIA Open Data API](https://www.eia.gov/opendata/)
- **Generation mix data**: NREL (National Renewable Energy Laboratory) — [NREL Data Catalog](https://data.nrel.gov/) — solar/wind generation profiles by region and season
- **Spot pricing**: EIA wholesale electricity prices (hourly, by ISO region)
- Data used to parameterise the simulation environment (not fed directly to the RL agent as a dataset)

### Model

- **Algorithm**: PPO (Proximal Policy Optimization) via Stable Baselines3
- **Environment**: Custom OpenAI Gym environment wrapping the grid simulation
- **State space**: Current demand, available capacity per source (coal, gas, solar, wind, hydro), current prices, time of day, day of week
- **Action space**: Discrete dispatch decisions — allocate MW from each generation source
- **Reward function**: Weighted combination of cost minimisation, emissions penalty, and reliability (unmet demand penalty)

### Pipeline & Training Steps

1. Pull historical demand and generation data from EIA/NREL APIs; fit demand and renewable availability distributions per hour-of-day and season
2. Build custom `gym.Env` class: `GridDispatchEnv` — step function advances one hour, applies dispatch action, returns next state + reward
3. Implement reward shaping: reward = −(dispatch cost) − λ×(CO₂ emissions) − μ×(unmet demand)²
4. Train PPO agent with Stable Baselines3: 500k–1M timesteps, MLP policy, entropy bonus for exploration
5. Establish baselines to beat: rule-based merit order dispatch (cheapest source first), random policy
6. Evaluate trained policy: average cost per MWh, emissions per MWh, supply reliability %, vs baselines
7. Ablation: vary λ (emissions weight) to show cost/emissions tradeoff curve — agent learns different policies
8. Visualise: 24-hour dispatch schedule plots, learning curves, reward breakdown by component
9. Build Streamlit demo: select season + demand scenario → agent dispatches grid in real time, animated hourly breakdown

### Hardware Note

PPO on a small state/action space trains entirely on CPU. 1M timesteps completes in approximately 30–60 minutes with Stable Baselines3.

---

## Portfolio Architecture

Each quest follows a consistent repo structure:

```
quest-name/
  Dockerfile         — containerised training & demo
  README.md          — problem statement, approach, results, key findings
  data/
    download.py      — reproducible data acquisition script
    sample/          — small sample for quick testing (committed to repo)
  src/
    train.py         — model training
    evaluate.py      — metrics and evaluation
    predict.py       — inference
  app.py             — Streamlit demo
  requirements.txt
  results/
    figures/         — plots, confusion matrices, learning curves
    metrics.json     — final evaluation numbers
```

### Directory Structure

```
ml_side_quests/
├── ml_side_quests.md
├── quest-01-cv-plant-disease/       — Computer Vision: Plant Disease Classification
├── quest-02-nlp-sentiment-analysis/ — NLP: Sentiment Analysis with DistilBERT
├── quest-03-genai-earnings-call/    — Generative AI: Energy Earnings Call Analyst
├── quest-04-rag-pipeline/           — RAG: Retrieval Evaluation Pipeline
├── quest-05-nlp-ticket-routing/     — NLP: Support Ticket Routing
└── quest-06-rl-grid-balancing/      — RL: Energy Grid Load Balancing Agent
```

## Hardware Summary

| Quest                 | Estimated CPU Training Time    |
| --------------------- | ------------------------------ |
| 1 — Plant disease CV  | 1–3 hours (transfer learning)  |
| 2 — Sentiment BERT    | 1–2 hours (10k samples)        |
| 3 — Earnings call LLM | 6–10 hours overnight (LoRA)    |
| 4 — RAG pipeline      | No training — indexing ~30 min |
| 5 — Ticket routing    | Under 30 minutes               |
| 6 — Grid RL agent     | 30–60 minutes (1M timesteps)   |

## Key Libraries

```
torch torchvision                  # CV and NLP model training
transformers datasets peft trl     # Hugging Face ecosystem
sentence-transformers faiss-cpu    # RAG embeddings and retrieval
chromadb ragas                     # Vector store and RAG evaluation
stable-baselines3 gymnasium        # Reinforcement learning
scikit-learn lightgbm xgboost      # Classical ML baselines
streamlit                         # Demo interfaces
edgar-sec requests                 # EDGAR data acquisition
```

## Docker

Each quest has its own `Dockerfile`. A root `docker-compose.yml` orchestrates all six.

```bash
# Build and run a single quest
cd quest-01-cv-plant-disease
docker build -t quest-01-cv .
docker run --rm -v "$(pwd)/results:/app/results" quest-01-cv

# Run all Streamlit demos simultaneously
cd ..
docker compose up -d
# → http://localhost:8501 (CV)
# → http://localhost:8502 (NLP - Sentiment)
# → http://localhost:8503 (GenAI - Earnings)
# → http://localhost:8504 (RAG)
# → http://localhost:8505 (NLP - Tickets)
# → http://localhost:8506 (RL)

# Train inside a container with persisted results
docker compose run --rm quest-01-cv python src/train.py
```
