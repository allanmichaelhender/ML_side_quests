"""Streamlit demo for Support Ticket Routing.

Compares two trained approaches:
1. TF-IDF + Logistic Regression
2. DistilBERT fine-tuned
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
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

# Load models
vec_tfidf, clf_tfidf = load_tfidf()
model_bert, tokenizer_bert, device_bert = load_distilbert()

models_loaded = {
    "TF‑IDF": vec_tfidf is not None,
    "DistilBERT": model_bert is not None,
}

# Load metrics
metrics = None
metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

tab1, tab2 = st.tabs(["🔮 Live Inference", "📊 Metrics"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Live Inference
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
    Three approaches were evaluated on the **[Banking77](https://huggingface.co/datasets/PolyAI/banking77)**
    dataset (77 intent categories):

    - **TF‑IDF + Logistic Regression** — trained locally
    - **DistilBERT** (fine-tuned) — trained locally
    - **DeepSeek V4 Flash** — zero-shot via API (see Metrics tab)

    This tab lets you test the two trained models (TF-IDF/LR and DistilBERT) live. Enter a ticket below
    to compare their predictions side by side.
        """
    )

    if not any(models_loaded.values()):
        st.warning(
            "No models found. Run `python src/train.py` locally to train them. "
            "The demo will still show sample predictions if you proceed."
        )

    # ── Input ───────────────────────────────────────────────
    # Seed a random ticket on first load
    if "ticket_text" not in st.session_state:
        st.session_state.ticket_text = np.random.choice(SAMPLE_TICKETS)

    st.subheader("📝 Enter a support ticket")

    use_sample = st.button("🎲 New random ticket")

    if use_sample:
        st.session_state.ticket_text = np.random.choice(SAMPLE_TICKETS)

    ticket_text = st.text_area(
        "Describe your issue:",
        value=st.session_state.ticket_text,
        height=100,
        placeholder="e.g. I lost my card and need a replacement...",
        label_visibility="collapsed",
    )
    st.session_state.ticket_text = ticket_text

    # ── Predict ─────────────────────────────────────────────
    if ticket_text.strip():
        st.markdown("---")
        st.subheader("🔮 Predictions")

        results_cols = st.columns(2)

        # ── TF-IDF ───────────────────────────────────────────
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

        # ── DistilBERT ───────────────────────────────────────
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

        # ── Agreement indicator ──────────────────────────────
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
            st.markdown(
                f"- **{name}**: `{label.replace('_', ' ').title()}` ({conf:.1%})"
            )

        # ── Interpretability ─────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 What's driving the prediction?")

        interp_cols = st.columns(2)

        # ── TF-IDF top keywords ──────────────────────────────
        with interp_cols[0]:
            if vec_tfidf is not None:
                with st.expander("📌 TF‑IDF top keywords"):
                    feature_names = vec_tfidf.get_feature_names_out()
                    row = vec_tfidf.transform([ticket_text])
                    if row.nnz > 0:
                        coef_idx = row.indices
                        coef_data = row.data
                        sorted_idx = np.argsort(coef_data)[-10:][::-1]
                        keywords = [
                            (feature_names[coef_idx[i]], coef_data[i])
                            for i in sorted_idx
                        ]
                        for word, score in keywords:
                            st.markdown(f"- **{word}**: `{score:.4f}`")
                    else:
                        st.caption("No TF‑IDF features found (text too short?)")

        # ── DistilBERT token importance ──────────────────────
        with interp_cols[1]:
            if model_bert is not None and ticket_text.strip():
                with st.expander("🤖 DistilBERT influential tokens"):
                    with st.spinner("Analysing token impact..."):
                        encoding = tokenizer_bert(
                            ticket_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=128,
                            padding="max_length",
                        )
                        enc = {k: v.to(device_bert) for k, v in encoding.items()}

                        with torch.no_grad():
                            base_out = model_bert(**enc)
                        base_probs = F.softmax(base_out.logits, dim=-1).squeeze(0).cpu()
                        pred_class = int(torch.argmax(base_probs))
                        base_conf = float(base_probs[pred_class])

                        input_ids = encoding["input_ids"][0].to(device_bert)
                        mask_id = tokenizer_bert.mask_token_id
                        tokens = tokenizer_bert.convert_ids_to_tokens(input_ids)
                        # Only mask real content tokens (skip special tokens)
                        content_indices = [
                            i
                            for i, t in enumerate(tokens)
                            if t not in ("[CLS]", "[SEP]", "[PAD]")
                        ][:25]

                        importance = []
                        for idx in content_indices:
                            masked_ids = input_ids.clone()
                            masked_ids[idx] = mask_id
                            with torch.no_grad():
                                m_out = model_bert(
                                    input_ids=masked_ids.unsqueeze(0),
                                    attention_mask=enc["attention_mask"],
                                )
                            m_probs = F.softmax(m_out.logits, dim=-1).squeeze(0).cpu()
                            m_conf = float(m_probs[pred_class])
                            drop = base_conf - m_conf
                            importance.append((tokens[idx], drop))

                        importance.sort(key=lambda x: -x[1])
                        st.caption(
                            f"Impact on `{label_names[pred_class].replace('_', ' ').title()}` "
                            f"(masking each token)"
                        )
                        for rank, (token, drop) in enumerate(importance[:10]):
                            display = token.replace("##", "")
                            bg = "#2d7d46" if rank == 0 else "#555"
                            st.markdown(
                                f"<span style='background:{bg}; color:white; padding:2px 8px; "
                                f"border-radius:4px; font-size:0.9em'>"
                                f"**{display}**</span> &nbsp; ∆ {drop:.1%}",
                                unsafe_allow_html=True,
                            )

    else:
        st.info("👆 Enter a support ticket above to see routing predictions.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Metrics
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        """
    Model performance comparison across approaches.
    Trained on the **[Banking77](https://huggingface.co/datasets/PolyAI/banking77)**
    dataset — 77 fine-grained intent categories for banking queries.
        """
    )

    if metrics is None:
        st.info(
            "Run `python src/train.py && python src/evaluate.py` to populate results."
        )
    else:
        rows = []

        def _fmt_acc(acc):
            return f"{acc:.1%}" if isinstance(acc, float) else str(acc)

        # TF-IDF
        if "tfidf" in metrics:
            m = metrics["tfidf"]
            f1 = m.get("test_macro_f1")
            rows.append(
                {
                    "Model": "TF‑IDF + Logistic Regression",
                    "Type": "Linear classification pipeline using TF-IDF features",
                    "Accuracy": _fmt_acc(m.get("val_accuracy", "—")),
                    "F1 Score": f"{f1:.1%}" if f1 else "—",
                    "Train Time": f"{m.get('training_time_s', '—')}s",
                }
            )

        # DistilBERT
        if "distilbert" in metrics:
            m = metrics["distilbert"]
            rows.append(
                {
                    "Model": "DistilBERT (fine-tuned)",
                    "Type": "Transformer (66M params)",
                    "Accuracy": _fmt_acc(m.get("val_accuracy", "—")),
                    "F1 Score": f"{m.get('val_macro_f1', 0):.1%}",
                    "Train Time": f"{m.get('training_time_min', '—'):.0f} min",
                }
            )

        # Evaluation approaches (DeepSeek etc.)
        if "evaluation" in metrics:
            for m in metrics["evaluation"]:
                rows.append(
                    {
                        "Model": m["approach"],
                        "Type": "Zero-shot LLM",
                        "Accuracy": f"{m['accuracy']:.1%}",
                        "F1 Score": f"{m.get('macro_f1', 0):.1%}",
                        "Train Time": "—",
                    }
                )

        if rows:
            st.subheader("📊 Model Comparison")
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # ── Bar chart comparison ─────────────────────────
            st.subheader("📈 Accuracy & F1 Comparison")

            model_names = [r["Model"] for r in rows]
            short_names = [n.split("(")[0].strip() for n in model_names]
            n_models = len(rows)
            x = range(n_models)
            width = 0.35

            fig, ax = plt.subplots(figsize=(9, 4))

            acc_vals = []
            f1_vals = []
            for r in rows:
                acc = (
                    float(r["Accuracy"].rstrip("%")) / 100
                    if r["Accuracy"] != "—"
                    else None
                )
                f1 = (
                    float(r["F1 Score"].rstrip("%")) / 100
                    if r["F1 Score"] != "—"
                    else None
                )
                acc_vals.append(acc)
                f1_vals.append(f1)

            bars_acc = ax.bar(
                [p - width / 2 for p in x],
                [v if v is not None else 0 for v in acc_vals],
                width,
                label="Accuracy",
                color="#2e86ab",
            )
            bars_f1 = ax.bar(
                [p + width / 2 for p in x],
                [v if v is not None else 0 for v in f1_vals],
                width,
                label="F1 Score",
                color="#a23b72",
            )

            for bars, vals in [(bars_acc, acc_vals), (bars_f1, f1_vals)]:
                for bar, v in zip(bars, vals):
                    if v is not None:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.1%}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

            ax.set_xticks(list(x))
            ax.set_xticklabels(short_names, fontsize=10)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=10)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No metrics data available.")
