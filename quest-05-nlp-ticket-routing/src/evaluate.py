"""Evaluation and visualisation for the ticket routing models.

Evaluates TF-IDF, DistilBERT, and optionally DeepSeek zero-shot
on the Banking77 test set. Generates comparison reports and plots.
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from data_utils import (
    load_banking77,
    load_label_info,
    _stratified_sample_df,
)
from data_utils import DEFAULT_OUTPUT

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MODEL = DEFAULT_OUTPUT / "model"
DEFAULT_TFIDF_DIR = DEFAULT_OUTPUT / "tfidf_model"


def load_tfidf_model(model_dir: Path = DEFAULT_TFIDF_DIR):
    """Load trained TF-IDF vectorizer and classifier."""
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    print(f"TF-IDF model loaded from {model_dir}")
    return vectorizer, clf


def load_distilbert_model(model_dir: Path = DEFAULT_MODEL, device: torch.device = None):
    """Load trained DistilBERT model and tokenizer."""
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
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))

    print(f"DistilBERT model loaded from {model_dir}")
    print(f"  Device: {device}")
    return model, tokenizer


@torch.no_grad()
def evaluate_distilbert(
    model,
    tokenizer,
    test_texts: list,
    test_labels: list,
    label_names: list,
    max_length: int = 128,
    batch_size: int = 32,
):
    """Run DistilBERT on test set, return predictions and probabilities."""
    device = next(model.parameters()).device

    tokenized = tokenizer(
        test_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["labels"] = test_labels
    test_dataset = Dataset.from_dict(tokenized)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_labels = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    predictions = torch.argmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()
    probs = F.softmax(logits, dim=-1).numpy()

    return predictions, labels_np, probs


def evaluate_tfidf(vectorizer, clf, test_texts: list, test_labels: list):
    """Run TF-IDF classifier on test set."""
    X_test = vectorizer.transform(test_texts)
    predictions = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    return predictions, np.array(test_labels), probs


from llm_eval import evaluate_deepseek


def compute_metrics_dict(
    predictions, labels_np, probs, label_names, approach_name: str
):
    """Compute and print evaluation metrics."""
    acc = accuracy_score(labels_np, predictions)
    cm = confusion_matrix(labels_np, predictions)
    report = classification_report(
        labels_np,
        predictions,
        labels=range(len(label_names)),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        labels_np, predictions, labels=range(len(label_names)), zero_division=0
    )

    print(f"\n{'=' * 60}")
    print(f"  {approach_name} — Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"{'=' * 60}")
    print(report[:500])  # Truncated for readability
    print(f"  ... (full report saved to metrics.json)")

    result = {
        "approach": approach_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(
            precision_recall_fscore_support(
                labels_np, predictions, average="macro", zero_division=0
            )[2],
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
    # Save mean confidence if probs provided (e.g. from DeepSeek)
    if probs is not None and len(probs) > 0:
        result["mean_confidence"] = round(float(np.mean(probs)), 4)
    return result


def plot_confusion_matrix(cm, class_names, save_path: Path, title: str = ""):
    """Plot and save a confusion matrix heatmap (truncated to top-N classes)."""
    # If too many classes, show a truncated view (top 20 by support)
    top_n = min(20, len(class_names))
    class_support = cm.sum(axis=1)
    top_indices = np.argsort(class_support)[-top_n:]
    cm_trunc = cm[np.ix_(top_indices, top_indices)]
    class_trunc = [class_names[i] for i in top_indices]

    plt.figure(figsize=(10, 9))
    plt.imshow(cm_trunc, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (top {top_n} classes) — {title}", fontsize=13)
    plt.colorbar(shrink=0.8)

    tick_marks = np.arange(len(class_trunc))
    plt.xticks(tick_marks, class_trunc, rotation=90, fontsize=7)
    plt.yticks(tick_marks, class_trunc, fontsize=7)

    # Annotate
    thresh = cm_trunc.max() / 2.0
    for i in range(cm_trunc.shape[0]):
        for j in range(cm_trunc.shape[1]):
            plt.text(
                j,
                i,
                format(cm_trunc[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_trunc[i, j] > thresh else "black",
                fontsize=6,
            )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=11)
    plt.xlabel("Predicted label", fontsize=11)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def plot_comparison_bar(metrics_list: list, save_path: Path):
    """Plot accuracy comparison bar chart across approaches."""
    approaches = [m["approach"] for m in metrics_list]
    accuracies = [m["accuracy"] for m in metrics_list]
    f1s = [m.get("macro_f1", 0) for m in metrics_list]

    x = np.arange(len(approaches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#3498db")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Ticket Routing", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)

    # Annotate bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison chart saved to {save_path}")


def main(
    output_dir: Path = DEFAULT_OUTPUT,
    max_test_samples: int = 1_500,
    max_length: int = 128,
    batch_size: int = 32,
    deepseek: bool = False,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_only: bool = False,
):
    print("=" * 60)
    print("  Evaluation — Support Ticket Routing")
    print("=" * 60)

    # Load label info
    label_info = load_label_info(output_dir)
    label_names = label_info["label_names"]

    # Load test data directly
    print("\nLoading test set...")
    dataset_raw, _ = load_banking77(max_samples=None)  # load full dataset

    # Stratified subsample the test set so all classes are represented
    test_df = dataset_raw["test"].to_pandas()
    test_df = _stratified_sample_df(
        test_df, max_test_samples, seed=42, label_col="label"
    )
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    print(f"  Evaluating on {len(test_texts)} test samples")

    all_metrics = []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. TF-IDF ──────────────────────────────────────────────
    if not deepseek_only:
        print("\n" + "-" * 40)
        print("Evaluating TF-IDF...")
        print("-" * 40)
        try:
            vectorizer, clf = load_tfidf_model()
            preds_tfidf, labels_tfidf, probs_tfidf = evaluate_tfidf(
                vectorizer, clf, test_texts, test_labels
            )
            m = compute_metrics_dict(
                preds_tfidf, labels_tfidf, probs_tfidf, label_names, "TF-IDF"
            )
            all_metrics.append(m)

            plot_confusion_matrix(
                np.array(m["confusion_matrix"]),
                label_names,
                figures_dir / "confusion_matrix_tfidf.png",
                title="TF-IDF + Logistic Regression",
            )
        except FileNotFoundError:
            print("  TF-IDF model not found. Run `python src/train.py` first.")

    # ── 2. DistilBERT ──────────────────────────────────────────
    if not deepseek_only:
        print("\n" + "-" * 40)
        print("Evaluating DistilBERT...")
        print("-" * 40)
        try:
            model, tokenizer = load_distilbert_model()
            preds_bert, labels_bert, probs_bert = evaluate_distilbert(
                model,
                tokenizer,
                test_texts,
                test_labels,
                label_names,
                max_length=max_length,
                batch_size=batch_size,
            )
            m = compute_metrics_dict(
                preds_bert, labels_bert, probs_bert, label_names, "DistilBERT"
            )
            all_metrics.append(m)

            plot_confusion_matrix(
                np.array(m["confusion_matrix"]),
                label_names,
                figures_dir / "confusion_matrix_distilbert.png",
                title="DistilBERT",
            )
        except FileNotFoundError:
            print("  DistilBERT model not found. Run `python src/train.py` first.")

    # ── 3. DeepSeek (optional) ─────────────────────────────────
    if deepseek:
        print("\n" + "-" * 40)
        print(f"Evaluating DeepSeek ({deepseek_model})...")
        print("-" * 40)

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            try:
                from dotenv import load_dotenv

                env_path = PROJECT.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            except ImportError:
                pass

        if not api_key:
            print("  DEEPSEEK_API_KEY not found — skipping DeepSeek eval")
        else:
            # Use smaller sample for DeepSeek (API cost)
            ds_samples = min(500, max_test_samples)
            ds_texts = test_texts[:ds_samples]
            ds_labels = test_labels[:ds_samples]
            print(f"  Evaluating on {ds_samples} samples (API cost ~$0.01-0.02)")

            preds_ds, labels_ds, confs_ds = evaluate_deepseek(
                ds_texts,
                ds_labels,
                label_names,
                api_key,
                model=deepseek_model,
            )

            # Filter out unmatched (-1)
            valid_mask = preds_ds != -1
            preds_valid = preds_ds[valid_mask]
            labels_valid = labels_ds[valid_mask]
            print(f"  Valid predictions: {len(preds_valid)}/{len(preds_ds)}")

            # ── Save raw per-sample predictions for later analysis ──
            raw_records = []
            for i in range(len(ds_texts)):
                raw_records.append(
                    {
                        "ticket": ds_texts[i],
                        "true_label": int(ds_labels[i]),
                        "true_label_name": label_names[int(ds_labels[i])],
                        "predicted_label": int(preds_ds[i])
                        if preds_ds[i] != -1
                        else None,
                        "predicted_label_name": label_names[int(preds_ds[i])]
                        if preds_ds[i] != -1
                        else None,
                        "confidence": float(confs_ds[i]),
                        "matched": bool(valid_mask[i]),
                    }
                )
            raw_path = (
                output_dir
                / f"deepseek_predictions_{deepseek_model.replace('.', '-')}.json"
            )
            with open(raw_path, "w") as f:
                json.dump(raw_records, f, indent=2)
            print(f"  Raw predictions saved to {raw_path}")

            if len(preds_valid) > 0:
                # Pass confidence scores as probs for the valid subset
                confs_valid = confs_ds[valid_mask]
                m = compute_metrics_dict(
                    preds_valid,
                    labels_valid,
                    confs_valid,
                    label_names,
                    f"DeepSeek ({deepseek_model})",
                )
                # Adjust accuracy to include unmatched as errors
                total = len(preds_ds)
                matched = len(preds_valid)
                m["accuracy"] = round(m["accuracy"] * matched / total, 4)
                m["match_rate"] = round(matched / total, 4)
                all_metrics.append(m)

    # ── Comparison plot ────────────────────────────────────────
    if len(all_metrics) >= 2:
        plot_comparison_bar(all_metrics, figures_dir / "model_comparison.png")

    # ── Save metrics ───────────────────────────────────────────
    combined = {"evaluation": all_metrics}
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(combined)
        combined = existing
    with open(metrics_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Evaluation metrics saved to {metrics_path}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    for m in all_metrics:
        print(
            f"  {m['approach']:20s}  Accuracy: {m['accuracy']:.4f}  Macro F1: {m['macro_f1']:.4f}"
        )
    print(f"\n  Figures saved to {figures_dir}")

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ticket routing models")
    parser.add_argument("--max-samples", type=int, default=1_500)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--deepseek", action="store_true", help="Evaluate DeepSeek zero-shot"
    )
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--deepseek-only",
        action="store_true",
        help="Skip TF-IDF and DistilBERT, evaluate DeepSeek only",
    )
    args = parser.parse_args()

    main(
        max_test_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        deepseek=args.deepseek or args.deepseek_only,
        deepseek_model=args.deepseek_model,
        deepseek_only=args.deepseek_only,
    )
