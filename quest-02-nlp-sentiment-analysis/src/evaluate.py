"""Evaluation and visualisation for the sentiment analysis model."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
import torch.nn.functional as F

from data_utils import (
    load_amazon_reviews,
    get_tokenizer,
    tokenize_dataset,
    load_label_info,
    LABEL_NAMES,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model"


def load_model_and_tokenizer(
    model_dir: Path = DEFAULT_MODEL, device: torch.device = None
):
    """Load a trained model and tokenizer from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_info = load_label_info(model_dir.parent)
    num_labels = len(label_info["label_names"])

    model = DistilBertForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()  # Setting to eval mode, this disables dropout layers, uses running averages over batch statistics for normalisation, gradient calculations are disabled
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))

    print(f"Model loaded from {model_dir}")
    print(f"  Device: {device}")
    return model, tokenizer


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    output_dir: Path = DEFAULT_OUTPUT,
    max_samples: int = 1_000,
    max_length: int = 256,
    batch_size: int = 32,
):

    device = (
        next(model.parameters()).device
    )  # safest way to get the device the model is on, works even if model is on GPU

    # Load only the test set directly (skip loading training data)
    import pandas as pd
    from datasets import Dataset

    test_path = DEFAULT_DATA / "test-00000-of-00001.parquet"
    if not test_path.exists():
        from data_utils import download_data

        download_data()

    test_df = pd.read_parquet(test_path)
    if max_samples and max_samples < len(test_df):
        test_df = test_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Tokenize directly
    texts = [f"{t} {c}" for t, c in zip(test_df["title"], test_df["content"])]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["labels"] = test_df["label"].tolist()

    test_dataset = Dataset.from_dict(tokenized)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Test set: {len(test_dataset)} samples")

    # Create dataloader
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_logits = []
    all_labels = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(
            outputs.logits.cpu()
        )  # Moves logits to CPU (if they are on GPU) to allow numpy operations later
        all_labels.append(labels)

    # Concatenating together all the batches of logits and lables inot a single tensor
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    predictions = torch.argmax(
        logits, dim=-1
    ).numpy()  # .numpy() converts the tensor into a numpy array for sklearn metrics
    labels_np = labels.numpy()
    probs = F.softmax(logits, dim=-1).numpy()

    # ── Metrics ────────────────────────────────────────────────
    acc = accuracy_score(labels_np, predictions)
    cm = confusion_matrix(labels_np, predictions)
    report = classification_report(
        labels_np, predictions, target_names=LABEL_NAMES, digits=4
    )
    per_class = precision_recall_fscore_support(
        labels_np, predictions, labels=range(len(LABEL_NAMES))
    )

    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"{'=' * 60}")
    print(report)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(
            precision_recall_fscore_support(labels_np, predictions, average="macro")[2],
            4,
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "precision": [round(p, 4) for p in per_class[0].tolist()],
            "recall": [round(r, 4) for r in per_class[1].tolist()],
            "f1": [round(f, 4) for f in per_class[2].tolist()],
            "support": per_class[3].tolist(),
        },
    }

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ── Plots ──────────────────────────────────────────────────
    plot_confusion_matrix(
        cm, LABEL_NAMES, output_dir / "figures" / "confusion_matrix.png"
    )
    plot_per_class_metrics(
        per_class, LABEL_NAMES, output_dir / "figures" / "per_class_metrics.png"
    )

    return metrics, probs, labels_np


def plot_confusion_matrix(cm, class_names, save_path: Path):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Sentiment Analysis", fontsize=14)
    plt.colorbar(shrink=0.8)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(per_class, class_names, save_path: Path):
    """Plot per-class precision, recall, F1 as a grouped bar chart."""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, per_class[0], width, label="Precision")
    bars2 = ax.bar(x, per_class[1], width, label="Recall")
    bars3 = ax.bar(x + width, per_class[2], width, label="F1-Score")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Precision, Recall, and F1-Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-class metrics saved to {save_path}")


def visualize_attention(
    model,
    tokenizer,
    output_dir: Path = DEFAULT_OUTPUT,
    num_examples: int = 5,
):
    """Visualise attention weights on example reviews."""
    device = next(model.parameters()).device

    # Example reviews across sentiment classes
    example_reviews = [
        (
            "positive",
            "This product is amazing! I absolutely love it, works perfectly and exceeded all my expectations.",
        ),
        (
            "positive",
            "Great quality and fast shipping. Would definitely recommend to anyone looking for a good buy.",
        ),
        (
            "positive",
            "It's okay for the price. Nothing special but it gets the job done. Average product overall.",
        ),
        (
            "negative",
            "The item arrived on time and works as described. Not great but not terrible either.",
        ),
        (
            "negative",
            "Terrible product, completely broken on arrival. Waste of money, do not buy this.",
        ),
        (
            "negative",
            "Very disappointed with the quality. Stopped working after just two days. Poor customer service.",
        ),
    ]

    save_dir = output_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    for sentiment, text in example_reviews[:num_examples]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Get attention from the last layer, averaged over all heads
        # DistilBERT has 6 layers, each with 12 attention heads
        # Shape: (batch, heads, seq_len, seq_len)
        attentions = outputs.attentions[-1]  # last layer
        avg_attention = attentions.mean(dim=1).squeeze(0)  # (seq_len, seq_len)

        # Aggregate attention from [CLS] token to all other tokens
        cls_attention = avg_attention[0, 1:].cpu().numpy()  # skip [CLS]→[CLS]

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = tokens[1:]  # skip [CLS]

        # Filter out [SEP] tokens (attention sinks that dominate the scale)
        keep = [i for i, t in enumerate(tokens) if t != "[SEP]"]
        tokens = [tokens[i] for i in keep]
        cls_attention = cls_attention[keep]

        # Get prediction
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        pred_label = LABEL_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

        # Normalise attention scores (now relative to content tokens only)
        cls_attention = (cls_attention - cls_attention.min()) / (
            cls_attention.max() - cls_attention.min() + 1e-8
        )

        # Plot
        fig, ax = plt.subplots(figsize=(14, 4))
        tokens_display = [t.replace("Ġ", "") for t in tokens]

        colors = plt.cm.Blues(cls_attention)
        bars = ax.bar(
            range(len(tokens_display)),
            cls_attention,
            color=colors,
            edgecolor="gray",
            linewidth=0.5,
        )

        ax.set_xticks(range(len(tokens_display)))
        ax.set_xticklabels(tokens_display, rotation=60, ha="right", fontsize=9)
        ax.set_ylabel("Attention Weight", fontsize=11)
        ax.set_title(
            f"Attention Visualisation — True: {sentiment} | Pred: {pred_label} ({confidence:.2f})",
            fontsize=13,
        )
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        safe_name = sentiment.replace("/", "_")
        path = save_dir / f"attention_{safe_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Attention plot saved to {path}")


def confusion_matrix_to_json(model, tokenizer, output_dir: Path = DEFAULT_OUTPUT):
    """Alias for evaluate_model that also returns the confusion matrix."""
    metrics, probs, labels = evaluate_model(model, tokenizer, output_dir)
    return metrics["confusion_matrix"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=1_000)
    parser.add_argument(
        "--visualize-attention",
        action="store_true",
        help="Generate attention visualisation plots",
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    if args.visualize_attention:
        print("\n--- Visualising attention weights ---")
        visualize_attention(model, tokenizer, args.output_dir)

    print("\n--- Running evaluation ---")
    evaluate_model(model, tokenizer, args.output_dir, max_samples=args.max_samples)
