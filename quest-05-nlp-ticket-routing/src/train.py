"""
Train classifiers for Support Ticket Routing.

Implements three approaches:
1. TF-IDF + Logistic Regression — fast classical baseline
2. DistilBERT fine-tuning — transformer-based classifier
3. (Evaluation only) Zero-shot via DeepSeek API — frontier LLM
"""

import json
import time
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from data_utils import (
    load_banking77,
    get_tokenizer,
    tokenize_dataset,
    build_tfidf_pipeline,
    save_label_info,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"


def train_tfidf(
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    output_dir: Path,
    max_features: int = 5000,
):
    """Train TF-IDF + Logistic Regression baseline."""
    print("\n" + "=" * 60)
    print("Approach 1: TF-IDF + Logistic Regression")
    print("=" * 60)

    vectorizer = build_tfidf_pipeline(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    print(f"  TF-IDF matrix: {X_train.shape}")
    print(f"  Training Logistic Regression (max_iter=500, C=1.0)...")

    clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1)
    start = time.time()
    clf.fit(X_train, train_labels)
    elapsed = time.time() - start

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f}")
    print(f"  Training time:  {elapsed:.1f}s")

    # Save model + vectorizer
    model_dir = output_dir / "tfidf_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print(f"  TF-IDF model saved to {model_dir}")

    return {
        "approach": "tfidf",
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "training_time_s": round(elapsed, 1),
        "max_features": max_features,
        "num_classes": len(np.unique(train_labels)),
    }


def compute_metrics_fn(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def train_distilbert(
    dataset,
    label_names: list,
    output_dir: Path,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    seed: int = 42,
):
    """Fine-tune DistilBERT for ticket routing."""
    print("\n" + "=" * 60)
    print("Approach 2: DistilBERT Fine-tuning")
    print("=" * 60)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Tokenize
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    # Build model
    num_labels = len(label_names)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)},
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Training arguments
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"distilbert-tickets-{int(time.time())}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print(
        f"\n  Training — {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}"
    )
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    print(f"  Training completed in {elapsed / 60:.1f} minutes")

    # Final evaluation on validation set
    final_eval = trainer.evaluate()
    val_accuracy = round(final_eval.get("eval_accuracy", 0), 4)
    val_f1 = round(final_eval.get("eval_macro_f1", 0), 4)
    print(f"  Final validation accuracy: {val_accuracy:.4f}  F1: {val_f1:.4f}")

    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"  Model saved to {model_path}")

    return {
        "approach": "distilbert",
        "train_loss": round(train_result.metrics.get("train_loss", 0), 4),
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1,
        "training_time_min": round(elapsed / 60, 2),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }


def main(
    data_dir: Path = DEFAULT_DATA,
    output_dir: Path = DEFAULT_OUTPUT,
    max_samples: int = 8_000,
    max_length: int = 128,
    batch_size: int = 16,
    seed: int = 42,
    skip_distilbert: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("=" * 60)
    print("  Support Ticket Routing — Training Pipeline")
    print("=" * 60)

    dataset, label_names = load_banking77(max_samples=max_samples, seed=seed)
    save_label_info(output_dir, label_names)

    # Extract plain text for TF-IDF
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    all_metrics = {"label_names": label_names, "num_labels": len(label_names)}

    # ── 1. TF-IDF + Logistic Regression ────────────────────────
    tfidf_metrics = train_tfidf(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        output_dir,
    )
    all_metrics["tfidf"] = tfidf_metrics

    # ── 2. DistilBERT ──────────────────────────────────────────
    if not skip_distilbert:
        bert_metrics = train_distilbert(
            dataset,
            label_names,
            output_dir,
            max_length=max_length,
            batch_size=batch_size,
            seed=seed,
        )
        all_metrics["distilbert"] = bert_metrics
    else:
        print("\n  Skipping DistilBERT (--skip-distilbert flag set)")

    # ── Save combined metrics ──────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(all_metrics)
        all_metrics = existing
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete — Summary")
    print("=" * 60)
    print(f"  TF-IDF validation accuracy: {tfidf_metrics['val_accuracy']:.4f}")
    if not skip_distilbert:
        print(
            f"  DistilBERT training time:   {bert_metrics['training_time_min']:.1f} min"
        )
    print(f"\n  Next step: run `python src/evaluate.py` for full evaluation")

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ticket routing classifiers")
    parser.add_argument(
        "--max-samples", type=int, default=8_000, help="Max training samples"
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-distilbert", action="store_true", help="Skip DistilBERT training"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        skip_distilbert=args.skip_distilbert,
        seed=args.seed,
    )
