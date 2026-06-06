"""Streamlit demo for Sentiment Analysis with DistilBERT."""

import logging
import sys
import warnings
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

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
HF_REPO = "allanhender/sentiment-distilbert-amazon"


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer (cached).

    Loads from local disk if available, otherwise falls back to Hugging Face Hub.
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    if (MODEL_DIR / "config.json").exists():
        model_path = str(MODEL_DIR)
    else:
        model_path = HF_REPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
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

    return pred_label, confidence, probs, pred_idx


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


def plot_integrated_gradients(text: str, model, tokenizer, device, target_class: int):
    """Generate an Integrated Gradients attribution plot showing each token's contribution
    to the predicted class. Positive = pushes toward the class, negative = pushes away."""
    from captum.attr import LayerIntegratedGradients

    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Baseline: all [PAD] tokens (zero information)
    baselines = torch.full_like(input_ids, tokenizer.pad_token_id)

    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        additional_forward_args=(attention_mask,),
        target=target_class,
        return_convergence_delta=True,
        n_steps=50,
    )

    # attributions: (1, seq_len, hidden_dim) → sum over hidden_dim → (seq_len,)
    attr = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Trim to actual tokens (exclude padding)
    actual_len = attention_mask[0].sum().item()
    tokens = tokens[:actual_len]
    attr = attr[:actual_len]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    tokens_display = [t.replace("Ġ", " ") for t in tokens]

    # Colour: green for positive contribution, red for negative
    colours = ["#2ecc71" if v >= 0 else "#e74c3c" for v in attr]
    ax.bar(
        range(len(tokens_display)),
        attr,
        color=colours,
        edgecolor="gray",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(tokens_display)))
    ax.set_xticklabels(tokens_display, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Attribution", fontsize=11)
    ax.set_title(
        f"Token Contributions — Integrated Gradients (target: {LABEL_NAMES[target_class]})",
        fontsize=11,
    )
    sns.despine()
    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────
st.title("📝 Sentiment Analysis with DistilBERT")
st.markdown(
    """
    Fine-tuned on **Amazon Polarity** — classifies product reviews as
    **Negative** 😠 or **Positive** 😊.

    *Powered by DistilBERT*
    """
)

# ── Sidebar: evaluation metrics ──────────────────────────────
metrics_path = HERE / "results" / "metrics.json"
if metrics_path.exists():
    import json

    with open(metrics_path) as f:
        metrics = json.load(f)
    if "accuracy" in metrics:
        with st.sidebar:
            st.header("📊 Test Set Performance")
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Macro F1", f"{metrics['macro_f1']:.2%}")

            st.subheader("Per Class")
            cols = st.columns(2)
            for i, name in enumerate(LABEL_NAMES):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    st.markdown(
                        f"Precision: {metrics['per_class']['precision'][i]:.2%}  \n"
                        f"Recall:    {metrics['per_class']['recall'][i]:.2%}  \n"
                        f"F1:        {metrics['per_class']['f1'][i]:.2%}"
                    )

            # Confusion matrix
            cm = np.array(metrics["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax_cm.set_xticks(range(len(LABEL_NAMES)))
            ax_cm.set_yticks(range(len(LABEL_NAMES)))
            ax_cm.set_xticklabels(LABEL_NAMES, fontsize=9)
            ax_cm.set_yticklabels(LABEL_NAMES, fontsize=9)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            ax_cm.set_xlabel("Predicted", fontsize=9)
            ax_cm.set_ylabel("True", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

model, tokenizer, device = load_model()

if model is None:
    st.warning(
        "Model not found. Run `python src/train.py` locally to train it. "
        "The results sidebar will still show evaluation metrics if available."
    )

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

analyze_btn = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True,
    disabled=(model is None),
)

# ── Results ──────────────────────────────────────────────────
if analyze_btn and text.strip() and model is not None:
    with st.spinner("Analysing sentiment..."):
        pred_label, confidence, probs, pred_idx = predict_sentiment(
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

    # Token attribution (Integrated Gradients)
    with st.expander(
        "🧠 Show token attributions (which words influenced the decision?)",
        expanded=True,
    ):
        st.caption(
            "**Integrated Gradients** — green bars push toward the predicted class, "
            "red bars push against it. Taller bars = stronger influence."
        )
        try:
            ig_fig = plot_integrated_gradients(text, model, tokenizer, device, pred_idx)
            st.pyplot(ig_fig)
        except Exception as e:
            st.warning(f"Could not generate attribution plot: {e}")
        except Exception as e:
            st.warning(f"Could not generate attribution plot: {e}")

    # ── Model comparison ─────────────────────────────────────
    st.markdown("### 📊 Model Comparison: 5k vs 20k Training Samples")
    st.markdown(
        """
        | Metric | 5k model | 20k model |
        |---|---|---|
        | **Training samples** | 5,000 | 20,000 |
        | **Training time** | ~25 min | ~5h 19min |
        | **Test accuracy** | 84.20% | **94.36%** |
        | **Macro F1** | 84.19% | **94.36%** |
        | **Negative F1** | 84.59% | 94.30% |
        | **Positive F1** | 83.78% | 94.42% |
        """
    )

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
        - **Negative** 😠
        - **Positive** 😊 
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
