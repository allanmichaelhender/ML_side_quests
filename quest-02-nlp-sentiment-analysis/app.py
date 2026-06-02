"""Streamlit demo for Sentiment Analysis with DistilBERT."""

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

from data_utils import LABEL_NAMES

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — DistilBERT",
    page_icon="📝",
    layout="centered",
)

MODEL_DIR = HERE / "results" / "model"


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer (cached)."""
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    if not (MODEL_DIR / "config.json").exists():
        st.error(
            f"Model not found at `{MODEL_DIR}`. "
            "Please run `python src/train.py` first to train the model."
        )
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    return model, tokenizer, device


def predict_sentiment(text: str, model, tokenizer, device):
    """Predict sentiment for a given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


def plot_confidence_bars(probs):
    """Plot confidence scores as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#e74c3c", "#2ecc71"]
    y_pos = np.arange(len(LABEL_NAMES))

    bars = ax.barh(
        y_pos, probs, color=colors, height=0.6, edgecolor="gray", linewidth=0.5
    )

    for bar, prob in zip(bars, probs):
        ax.text(
            prob + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(LABEL_NAMES, fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_title("Sentiment Confidence Scores", fontsize=14)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return fig


def plot_attention_heatmap(text: str, model, tokenizer, device):
    """Generate an attention heatmap showing which tokens the model focused on."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )

    # Last layer attention, averaged over heads
    attentions = outputs.attentions[-1]  # (1, heads, seq_len, seq_len)
    avg_attention = attentions.mean(dim=1).squeeze(0)  # (seq_len, seq_len)
    cls_attention = avg_attention[0, 1:].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = tokens[1:]  # skip [CLS]

    # Truncate to actual tokens (remove padding)
    actual_len = attention_mask[0].sum().item() - 1  # exclude [CLS]
    tokens = tokens[:actual_len]
    cls_attention = cls_attention[:actual_len]

    # Normalize
    if cls_attention.max() > cls_attention.min():
        cls_attention = (cls_attention - cls_attention.min()) / (
            cls_attention.max() - cls_attention.min()
        )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    tokens_display = [t.replace("Ġ", " ") for t in tokens]

    colors = plt.cm.Blues(cls_attention)
    ax.bar(
        range(len(tokens_display)),
        cls_attention,
        color=colors,
        edgecolor="gray",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(tokens_display)))
    ax.set_xticklabels(tokens_display, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Attention", fontsize=11)
    ax.set_title("Token-Level Attention (from [CLS] token)", fontsize=13)
    ax.set_ylim(0, 1.05)
    sns.despine()
    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────
st.title("📝 Sentiment Analysis with DistilBERT")
st.markdown(
    """
    Fine-tuned on **Amazon Polarity** — classifies product reviews as
    **Negative** 😠 or **Positive** 😊.

    *Powered by DistilBERT — 40% smaller than BERT, 60% faster, ~97% of the performance.*
    """
)

model, tokenizer, device = load_model()

# ── Input ────────────────────────────────────────────────────
st.subheader("Enter a review")
input_mode = st.radio(
    "Input mode:", ["Type a review", "Example reviews"], horizontal=True
)

example_reviews = [
    ("Amazing product, exceeded expectations! I love it.", "positive"),
    ("Works great, very happy with my purchase. Would buy again!", "positive"),
    ("Terrible quality, broke after one use. Complete waste of money.", "negative"),
    (
        "Very disappointed, product arrived damaged and customer support was unhelpful.",
        "negative",
    ),
]

if input_mode == "Example reviews":
    selected = st.selectbox(
        "Choose an example:",
        [
            f'"{r[0][:60]}..." — {r[1]}' if len(r[0]) > 60 else f'"{r[0]}" — {r[1]}'
            for r in example_reviews
        ],
    )
    # Find which example was selected
    idx = [
        f'"{r[0][:60]}..." — {r[1]}' if len(r[0]) > 60 else f'"{r[0]}" — {r[1]}'
        for r in example_reviews
    ].index(selected)
    text = example_reviews[idx][0]
else:
    text = st.text_area(
        "Write your review:",
        height=120,
        placeholder="e.g. This product is amazing! I love it.",
    )

analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ── Results ──────────────────────────────────────────────────
if analyze_btn and text.strip():
    with st.spinner("Analysing sentiment..."):
        pred_label, confidence, probs = predict_sentiment(
            text, model, tokenizer, device
        )

    # Sentiment display
    emoji_map = {"negative": "😠", "positive": "😊"}
    color_map = {"negative": "#e74c3c", "positive": "#2ecc71"}

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px;
                         background-color: {color_map[pred_label]}22;
                         border: 2px solid {color_map[pred_label]};">
                <span style="font-size: 48px;">{emoji_map[pred_label]}</span>
                <h3 style="color: {color_map[pred_label]}; margin: 5px 0;">
                    {pred_label.upper()}
                </h3>
                <p style="font-size: 24px; font-weight: bold; margin: 0;">
                    {confidence:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.pyplot(plot_confidence_bars(probs))

    # Attention heatmap
    with st.expander(
        "🧠 Show attention heatmap (which words matter most?)", expanded=True
    ):
        st.caption(
            "The bar height shows how much the model 'paid attention' to each token "
            "when making its classification decision."
        )
        try:
            attn_fig = plot_attention_heatmap(text, model, tokenizer, device)
            st.pyplot(attn_fig)
        except Exception as e:
            st.warning(f"Could not generate attention plot: {e}")

elif analyze_btn:
    st.warning("Please enter some text to analyze.")

# ── Sidebar info ─────────────────────────────────────────────
with st.sidebar:
    st.header("About the Model")
    st.markdown(
        """
        **Architecture**: DistilBERT (`distilbert-base-uncased`)
        - 6 transformer layers (vs BERT's 12)
        - 12 attention heads per layer
        - ~67M parameters

        **Training**:
        - Dataset: Amazon Polarity
        - 10,000 stratified samples
        - 3 epochs, batch size 16
        - AdamW optimiser, linear warmup

        **Classes**:
        - **Negative** 😠 (0 stars)
        - **Positive** 😊 (1 stars)
        """
    )

    st.header("Results")
    results_dir = HERE / "results"
    if (results_dir / "metrics.json").exists():
        import json

        with open(results_dir / "metrics.json") as f:
            metrics = json.load(f)
        if "accuracy" in metrics:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        if "macro_f1" in metrics:
            st.metric("Macro F1", f"{metrics['macro_f1']:.4f}")

    if (results_dir / "figures" / "confusion_matrix.png").exists():
        st.image(
            str(results_dir / "figures" / "confusion_matrix.png"),
            caption="Confusion Matrix",
            use_container_width=True,
        )
