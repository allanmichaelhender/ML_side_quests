# 🎫 Quest 05: Support Ticket Routing

**Domain:** NLP — Multi-class Text Classification  
**Dataset:** [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (77 intent categories)  
**Approaches:** TF‑IDF + Logistic Regression · DistilBERT · DeepSeek V4 Flash (zero-shot)  
**Demo:** `http://localhost:8505`

---

## Problem Statement

Customer support teams receive thousands of tickets daily. Manually routing each ticket to the right department is slow, error-prone, and doesn't scale.

**Goal:** Build a system that automatically classifies a support ticket into one of 77 fine-grained intent categories — so it can be instantly routed to the right team with zero human intervention.

**Example tickets:**
| Ticket | Correct Route |
|---|---|
| "I haven't received my new card yet" | `card_arrival` |
| "Someone used my card without permission" | `fraud_report` |
| "Can you increase my withdrawal limit?" | `card_limits` |
| "The ATM didn't give me the cash" | `atm_support` |

---

## Dataset: Banking77

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) is a carefully curated dataset of **13,083** customer service queries from the banking domain, labelled with **77 distinct intents**.

| Split    | Samples |
| -------- | ------- |
| Training | 10,003  |
| Test     | 3,080   |

Intents cover realistic banking scenarios: card issues, fraud, transfers, payments, account management, loans, ATM problems, and more.

---

## Three Approaches

### 1️⃣ TF‑IDF + Logistic Regression _(Baseline)_

Classical NLP pipeline:

- Convert text to TF‑IDF vectors (5k features, 1–2 ngrams)
- Train multinomial logistic regression
- **Training time:** ~30 seconds
- **Expected accuracy:** ~85–88%

**Pros:** Fast, interpretable, works on any hardware  
**Cons:** No understanding of word order or context

### 2️⃣ DistilBERT Fine-tuning _(Transformer)_

Fine-tune DistilBERT (66M params) on the Banking77 training set:

- Sequence classification head on `distilbert-base-uncased`
- 128 token max length, 3 epochs
- **Training time:** ~15–25 minutes (CPU) / ~3 minutes (GPU)
- **Expected accuracy:** ~92–94%

**Pros:** Context-aware, state-of-the-art for text classification  
**Cons:** Larger model, requires training

### 3️⃣ DeepSeek V4 Flash _(Zero-shot Frontier LLM — offline evaluation only)_

No training — just prompt an LLM with the list of intents and the ticket text:

- Uses DeepSeek's OpenAI-compatible API (`deepseek-chat`)
- Zero-shot: the model has never seen Banking77 before
- **Evaluated offline** on the full test set (not available for live queries in the demo)

**Pros:** No training needed, most flexible, can adapt to new intents instantly  
**Cons:** Requires API key, cost per query, no guaranteed format

---

## Pipeline

```
                      ┌─────────────────────┐
                      │   Raw Ticket Text    │
                      └──────────┬──────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐
          │  TF-IDF  │    │DistilBERT│
          │ Vectorize│    │ Tokenize │
          └────┬─────┘    └────┬─────┘
               ▼               ▼
          ┌──────────┐    ┌──────────┐
          │Logistic  │    │ Fine-    │
          │Regression│    │ tuned    │
          └────┬─────┘    └────┬─────┘
               ▼               ▼
          ┌──────────────────────────┐
          │  Predicted Intent + Conf │
          └──────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- `DEEPSEEK_API_KEY` in `.env` (root directory of the monorepo) — only needed for offline evaluation

### Local

```bash
cd quest-05-nlp-ticket-routing

# Create venv (optional — use root .venv or local one)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Train models (TF-IDF ~30s + DistilBERT ~20min)
python src/train.py

# Evaluate on test set
python src/evaluate.py

# Launch Streamlit demo
streamlit run app.py
# → http://localhost:8505
```

### Docker

```bash
docker compose up -d quest-05-ticket
# → http://localhost:8505
```

> Train first if no model exists in `results/`.

---

## Results

| Approach          | Test Accuracy | Macro F1 | Match Rate | Training Time |
| ----------------- | ------------- | -------- | ---------- | ------------- |
| TF‑IDF + LR       | **83.6%**     | 0.269    | 100%       | 1s            |
| DistilBERT        | **85.2%**     | 0.333    | 100%       | 53 min        |
| DeepSeek V4 Flash | **63.0%** †   | 0.499    | 69.8%      | None          |

> † DeepSeek uses Pydantic `Literal` to enforce exact intent names. It only predicts
> on **349/500** samples where it's confident enough to return a valid category —
> on those, accuracy is **90.3%**. The 151 skipped tickets are cases where the model
> couldn't find a good match among the 77 intents, effectively saying "I don't know."
> This honest-uncertainty behaviour is a strength of frontier LLMs vs. trained models
> that always guess.

---

## Key Findings

1. **TF‑IDF is shockingly good** for a 1-second train time — it captures keyword-level patterns that differentiate intents well.
2. **DistilBERT captures nuance** that TF‑IDF misses: "My card was declined" vs "I want to decline a charge" get different routes.
3. **DeepSeek knows when it doesn't know** — with Pydantic `Literal` enforcement, it achieves 90% accuracy on confident predictions but declines ~30% of tickets. This makes it ideal as a triage layer before falling back to trained models.

---

## File Structure

```
quest-05-nlp-ticket-routing/
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py                     # Streamlit demo
├── data/
│   └── sample/
│       └── tickets.json       # 20 sample tickets
├── results/
│   ├── label_info.json        # 77 intent labels
│   ├── metrics.json           # Training + evaluation metrics
│   ├── model/                 # Saved DistilBERT
│   ├── tfidf_model/           # Saved TF-IDF pipeline
│   ├── checkpoints/           # HF Trainer checkpoints
│   └── figures/               # Evaluation plots
└── src/
    ├── data_utils.py          # Data loading, preprocessing
    ├── train.py               # TF-IDF + DistilBERT training
    └── evaluate.py            # Test-set evaluation & plots
```

---

## What I Learned

- Banking77 is a well-structured dataset but some intents are very similar (`card_arrival` vs `getting_spare_card`) — a challenge even for BERT.
- TF‑IDF + LR is a strong baseline that often beats simple deep learning approaches on well-separated classes.
- Zero-shot LLM classification is powerful but unpredictable — controlling output format is the hardest part.
- For production ticket routing, a two-stage approach (fast model → fallback to LLM for low-confidence cases) is ideal.
