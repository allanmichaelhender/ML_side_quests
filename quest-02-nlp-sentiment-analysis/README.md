# 📈 Quest 02 — Sentiment Analysis

**Domain:** NLP — Text Classification (Sentiment / Emotion)  
**Models:** DistilBERT (fine-tuned), TF‑IDF + Logistic Regression (baseline)  
**Demo:** `http://localhost:8502`

---

## Problem Statement

Classify text into sentiment/emotion categories — a foundational NLP task with applications in social media monitoring, customer feedback analysis, and brand reputation management.

---

## Dataset

Uses a subset of the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) dataset
(or a custom labelled corpus of ~20k samples) with multiple preprocessing strategies
explored during experimentation.

| Split      | Samples |
| ---------- | ------- |
| Training   | ~16,000 |
| Validation | ~2,000  |
| Test       | ~2,000  |

---

## Approaches

### TF‑IDF + Logistic Regression _(Baseline)_

- TF‑IDF vectorization with unigram + bigram features
- Multinomial logistic regression classifier
- Fast training (~seconds), fully interpretable

### DistilBERT Fine-tuning _(Transformer)_

- `distilbert-base-uncased` with sequence classification head
- Trained with Hugging Face `Trainer` API
- ~20 minute training time on CPU

---

## Usage

### Local

```bash
cd quest-02-nlp-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py

# Evaluate
python src/evaluate.py

# Launch Streamlit demo
streamlit run app.py
# → http://localhost:8502
```

### Docker

```bash
docker compose up quest-02-sentiment
# → http://localhost:8502
```

---

## Results

| Approach          | Test Accuracy | Macro F1 | Training Time |
| ----------------- | ------------- | -------- | ------------- |
| TF‑IDF + LR (5k)  | ~72%          | —        | ~2s           |
| TF‑IDF + LR (20k) | ~85%          | —        | ~8s           |
| DistilBERT        | ~92%          | —        | ~20 min       |

> See `results/metrics.json` and `results/figures/` for detailed per-run metrics and visualizations.
> Multiple experiment configs are available (`metrics_5k.json`, `metrics_20k.json`, etc.).

---

## Project Structure

```
quest-02-nlp-sentiment-analysis/
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py                     # Streamlit demo
├── data/
│   └── sample/
├── results/
│   ├── label_info.json
│   ├── metrics.json
│   ├── metrics_5k.json
│   ├── metrics_20k.json
│   ├── metrics_20k_training.json
│   ├── checkpoints/
│   ├── figures/
│   ├── model/
│   └── model_5k/
└── src/
    ├── data_utils.py          # Data loading & preprocessing
    ├── train.py               # Model training
    └── evaluate.py            # Evaluation & visualization
```
