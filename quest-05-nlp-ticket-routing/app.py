"""Streamlit demo for Support Ticket Routing.

Compares two trained approaches:
1. TF-IDF + Logistic Regression
2. DistilBERT fine-tuned
"""

import json
import pickle
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from data_utils import load_label_info

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Routing — NLP",
    page_icon="🎫",
    layout="wide",
)

RESULTS_DIR = HERE / "results"
MODEL_DIR = RESULTS_DIR / "model"
TFIDF_DIR = RESULTS_DIR / "tfidf_model"

# Load label info
label_info = (
    load_label_info(RESULTS_DIR) if (RESULTS_DIR / "label_info.json").exists() else None
)
label_names = label_info["label_names"] if label_info else []

# Fallback: load label names directly from dataset if no local info
if not label_names:
    try:
        from datasets import load_dataset

        ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        label_names = ds.features["label"].names
    except Exception:
        label_names = []


# ── Model loading (cached) ──────────────────────────────────
@st.cache_resource
def load_tfidf():
    """Load TF-IDF vectorizer + classifier."""
    if not (TFIDF_DIR / "classifier.pkl").exists():
        return None, None
    with open(TFIDF_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(TFIDF_DIR / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    return vectorizer, clf


@st.cache_resource
def load_distilbert():
    """Load DistilBERT model + tokenizer."""
    if not (MODEL_DIR / "config.json").exists():
        return None, None, None
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    return model, tokenizer, device


# ── Prediction functions ────────────────────────────────────
def predict_tfidf(text: str, vectorizer, clf) -> dict:
    """Predict using TF-IDF model."""
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Get top-5 predictions
    top_k = min(5, len(probs))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {"label": label_names[i], "confidence": float(probs[i])} for i in top_indices
    ]

    return {
        "label": pred_label,
        "index": pred_idx,
        "confidence": confidence,
        "probs": probs,
        "top_k": top_predictions,
    }


def predict_distilbert(text: str, model, tokenizer, device) -> dict:
    """Predict using DistilBERT model."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = label_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Get top-5
    top_k = min(5, len(probs))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {"label": label_names[i], "confidence": float(probs[i])} for i in top_indices
    ]

    return {
        "label": pred_label,
        "index": pred_idx,
        "confidence": confidence,
        "probs": probs,
        "top_k": top_predictions,
    }


# ── Visualisation ───────────────────────────────────────────
def plot_confidence_bars(top_k, title: str, color: str):
    """Plot top-K predictions as horizontal bars."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = [p["label"].replace("_", " ").title() for p in top_k]
    scores = [p["confidence"] for p in top_k]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_k)))

    bars = ax.barh(
        range(len(labels)),
        scores,
        color=colors,
        height=0.6,
        edgecolor="gray",
        linewidth=0.5,
    )

    for bar, score, label in zip(bars, scores, labels):
        ax.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1%} — {label}",
            va="center",
            fontsize=10,
        )

    ax.set_yticks([])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=color)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return fig


# ── Sample tickets ──────────────────────────────────────────
SAMPLE_TICKETS = [
    "I haven't received my new card yet and it's been 2 weeks since I ordered it.",
    "Can you tell me why my last transaction was declined? I have enough funds.",
    "Someone used my card to make a purchase I didn't authorise. Please help!",
    "I need to update my phone number on my account.",
    "There's a charge on my statement that I don't recognise from last week.",
    "How do I set up a direct debit for my monthly rent payment?",
    "I forgot my online banking password and can't log in.",
    "Can you increase my daily withdrawal limit?",
    "I want to close my savings account and transfer the balance.",
    "My card was lost somewhere, I need a replacement urgently.",
    "The money I transferred hasn't arrived in the other account yet.",
    "I need a statement for my account for the last 3 months.",
    "Can I get a loan approval status update?",
    "The ATM charged me but didn't dispense any cash.",
    "My contactless payment isn't working on the new terminal.",
]


# ── UI ──────────────────────────────────────────────────────
st.title("🎫 Support Ticket Routing")
st.markdown(
    """
    Automatically route customer support tickets to the right team using
    **TF‑IDF + Logistic Regression** and fine‑tuned **DistilBERT**.

    Trained on the **[Banking77](https://huggingface.co/datasets/PolyAI/banking77)**
    dataset — 77 fine-grained intent categories for banking queries.

    > Sidebar shows test-set metrics for both models. DeepSeek V4 Flash
    > has been evaluated offline for comparison (see README).
    """
)

# Load models
vec_tfidf, clf_tfidf = load_tfidf()
model_bert, tokenizer_bert, device_bert = load_distilbert()

models_loaded = {
    "TF‑IDF": vec_tfidf is not None,
    "DistilBERT": model_bert is not None,
}

st.sidebar.header("🔧 Models Available")
for name, loaded in models_loaded.items():
    icon = "✅" if loaded else "❌"
    st.sidebar.markdown(f"{icon} **{name}**")

if not any(models_loaded.values()):
    st.warning(
        "No models found. Run `python src/train.py` locally to train them. "
        "The demo will still show sample predictions if you proceed."
    )

# ── Input ───────────────────────────────────────────────────
st.subheader("📝 Enter a support ticket")

col1, col2 = st.columns([3, 1])
with col2:
    use_sample = st.button("🎲 Random sample ticket")

if use_sample or "ticket_text" not in st.session_state:
    if use_sample:
        st.session_state.ticket_text = np.random.choice(SAMPLE_TICKETS)
    else:
        st.session_state.ticket_text = ""

ticket_text = st.text_area(
    "Describe your issue:",
    value=st.session_state.ticket_text,
    height=100,
    placeholder="e.g. I lost my card and need a replacement...",
    label_visibility="collapsed",
)
st.session_state.ticket_text = ticket_text

# ── Predict ─────────────────────────────────────────────────
if ticket_text.strip():
    st.markdown("---")
    st.subheader("🔮 Predictions")

    results_cols = st.columns(2)

    # ── TF-IDF ───────────────────────────────────────────────
    if vec_tfidf is not None:
        result_tfidf = predict_tfidf(ticket_text, vec_tfidf, clf_tfidf)
        with results_cols[0]:
            st.markdown("#### 📊 TF‑IDF + Logistic Regression")
            st.markdown(
                f"**Prediction:** `{result_tfidf['label'].replace('_', ' ').title()}`"
            )
            st.metric("Confidence", f"{result_tfidf['confidence']:.1%}")
            fig = plot_confidence_bars(
                result_tfidf["top_k"][:5],
                "Top-5 Predictions — TF‑IDF",
                "#2e86ab",
            )
            st.pyplot(fig)
            plt.close()

    # ── DistilBERT ───────────────────────────────────────────
    if model_bert is not None:
        result_bert = predict_distilbert(
            ticket_text, model_bert, tokenizer_bert, device_bert
        )
        col_idx = 1 if vec_tfidf is not None else 0
        with results_cols[col_idx]:
            st.markdown("#### 🤖 DistilBERT")
            st.markdown(
                f"**Prediction:** `{result_bert['label'].replace('_', ' ').title()}`"
            )
            st.metric("Confidence", f"{result_bert['confidence']:.1%}")
            fig = plot_confidence_bars(
                result_bert["top_k"][:5],
                "Top-5 Predictions — DistilBERT",
                "#a23b72",
            )
            st.pyplot(fig)
            plt.close()

    # ── Agreement indicator ──────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Model Agreement")

    predictions = []
    if vec_tfidf is not None:
        predictions.append(
            ("TF‑IDF", result_tfidf["label"], result_tfidf["confidence"])
        )
    if model_bert is not None:
        predictions.append(
            ("DistilBERT", result_bert["label"], result_bert["confidence"])
        )

    unique_labels = set(p[1] for p in predictions)
    if len(unique_labels) == 1 and len(predictions) >= 2:
        st.success("✅ Both models agree on the route!")
    elif len(predictions) >= 2:
        st.warning("⚠️ Models disagree — consider manual review")

    for name, label, conf in predictions:
        st.markdown(f"- **{name}**: `{label.replace('_', ' ').title()}` ({conf:.1%})")

    # ── Interpretability: TF-IDF top words ───────────────────
    st.markdown("---")
    st.subheader("🔍 What's driving the prediction?")

    if vec_tfidf is not None:
        with st.expander("📌 TF‑IDF top keywords"):
            feature_names = vec_tfidf.get_feature_names_out()
            row = vec_tfidf.transform([ticket_text])
            if row.nnz > 0:
                coef_idx = row.indices
                coef_data = row.data
                sorted_idx = np.argsort(coef_data)[-10:][::-1]
                keywords = [
                    (feature_names[coef_idx[i]], coef_data[i]) for i in sorted_idx
                ]
                for word, score in keywords:
                    st.markdown(f"- **{word}**: `{score:.4f}`")
            else:
                st.caption("No TF‑IDF features found (text too short?)")

else:
    st.info("👆 Enter a support ticket above to see routing predictions.")


# ── Sidebar: metrics ────────────────────────────────────────
st.sidebar.header("📈 Results")
metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.sidebar.subheader("Model Performance")
    for approach_key in ["tfidf", "distilbert"]:
        if approach_key in metrics:
            m = metrics[approach_key]
            name = "TF‑IDF" if approach_key == "tfidf" else "DistilBERT"
            acc = m.get("val_accuracy", m.get("accuracy", "—"))
            st.sidebar.metric(
                f"{name} Accuracy", f"{acc:.1%}" if isinstance(acc, float) else acc
            )

    if "evaluation" in metrics:
        st.sidebar.subheader("Test Set Results")
        for m in metrics["evaluation"]:
            st.sidebar.metric(
                f"{m['approach']} Test",
                f"{m['accuracy']:.1%}",
                delta=f"F1: {m['macro_f1']:.1%}",
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"77 intent categories · Banking77 dataset")
else:
    st.sidebar.info(
        "Run `python src/train.py && python src/evaluate.py` to populate results."
    )

# ── Footer ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Built with:** scikit-learn, Transformers, PyTorch, Streamlit  
    **Port:** 8505
    """
)
