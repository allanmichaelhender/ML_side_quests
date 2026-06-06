"""
Train DistilBERT on a 10-class subset of Banking77 with improved hyperparameters.

Selected classes have genuine semantic overlap to test whether DistilBERT's
contextual understanding can outperform keyword-based methods.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT_DIR = PROJECT / "results" / "10class"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── The 10 selected classes ─────────────────────────────────────
SELECTED_LABELS = [
    "card_arrival",
    "card_delivery_estimate",
    "declined_card_payment",
    "declined_transfer",
    "declined_cash_withdrawal",
    "top_up_failed",
    "pending_top_up",
    "top_up_reverted",
    "card_payment_fee_charged",
    "extra_charge_on_statement",
]


def filter_to_10_classes(dataset, label_names):
    """Filter dataset to only the 10 selected classes and remap labels 0-9."""
    selected_ids = {name: i for i, name in enumerate(SELECTED_LABELS)}
    original_ids = [label_names.index(name) for name in SELECTED_LABELS]
    id_map = {orig: new for orig, new in zip(original_ids, range(len(SELECTED_LABELS)))}

    def filter_fn(examples):
        mask = [l in original_ids for l in examples["label"]]
        return {"keep": mask}

    def remap_fn(examples):
        return {"labels": [id_map[l] for l in examples["label"]]}

    filtered = {}
    for split in dataset:
        # Filter
        with_mask = dataset[split].add_column("keep", filter_fn(dataset[split])["keep"])
        filtered_ds = with_mask.filter(lambda x: x["keep"], remove_columns=["keep"])
        # Remap labels
        filtered_ds = filtered_ds.map(remap_fn, remove_columns=["label"])
        # Rename text column for consistency
        filtered[split] = filtered_ds.rename_column("text", "text")
        print(f"  {split}: {len(dataset[split])} → {len(filtered_ds)} samples")

    return DatasetDict(filtered), SELECTED_LABELS


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    print("=" * 60)
    print("  DistilBERT — 10-Class Subset Training")
    print("=" * 60)

    # ── Load full dataset ──────────────────────────────────────
    print("\nLoading Banking77...")
    full_dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names = full_dataset["train"].features["label"].names
    print(
        f"  Full dataset: {len(full_dataset['train'])} train, {len(full_dataset['test'])} test"
    )

    # ── Filter to 10 classes ───────────────────────────────────
    print(f"\nFiltering to {len(SELECTED_LABELS)} selected classes:")
    for i, name in enumerate(SELECTED_LABELS):
        orig_id = label_names.index(name)
        train_count = sum(1 for l in full_dataset["train"]["label"] if l == orig_id)
        test_count = sum(1 for l in full_dataset["test"]["label"] if l == orig_id)
        print(f"  [{i}] {name:40s}  train={train_count:4d}  test={test_count:3d}")

    # Combine all training data into one pool
    train_df = full_dataset["train"].to_pandas()
    orig_ids = [label_names.index(n) for n in SELECTED_LABELS]
    train_df = train_df[train_df["label"].isin(orig_ids)].copy()

    # Map original labels → 0-9
    id_map = {orig: new for orig, new in zip(orig_ids, range(10))}
    train_df["label"] = train_df["label"].map(id_map)

    # Also filter + remap test set
    test_df = full_dataset["test"].to_pandas()
    test_df = test_df[test_df["label"].isin(orig_ids)].copy()
    test_df["label"] = test_df["label"].map(id_map)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    # ── 5-Fold Cross-Validation ────────────────────────────────
    from sklearn.model_selection import StratifiedKFold

    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_fold = {"fold": -1, "val_acc": 0}

    print(f"\n  {N_FOLDS}-Fold Cross-Validation on {len(train_df)} total samples\n")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    # Pre-tokenize test set once
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)
    test_tokenized = test_tokenized.remove_columns(["text"])
    test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        print(f"\n{'=' * 60}")
        print(f"  Fold {fold + 1}/{N_FOLDS}")
        print(f"{'=' * 60}")

        train_part = train_df.iloc[train_idx]
        val_part = train_df.iloc[val_idx]

        train_ds = Dataset.from_pandas(train_part[["text", "label"]])
        val_ds = Dataset.from_pandas(val_part[["text", "label"]])

        # Tokenize
        train_tok = train_ds.map(tokenize_fn, batched=True)
        val_tok = val_ds.map(tokenize_fn, batched=True)
        train_tok = train_tok.remove_columns(["text"])
        val_tok = val_tok.remove_columns(["text"])
        train_tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Build fresh model for each fold
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=10,
            id2label={i: l for i, l in enumerate(SELECTED_LABELS)},
            label2id={l: i for i, l in enumerate(SELECTED_LABELS)},
        )
        model.to(device)

        fold_dir = OUTPUT_DIR / "checkpoints" / f"fold_{fold}"
        training_args = TrainingArguments(
            output_dir=str(fold_dir),
            run_name=f"distilbert-10class-fold{fold}-{int(time.time())}",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=10,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=42 + fold,
            dataloader_num_workers=0,
            fp16=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics_fn,
        )

        fold_start = time.time()
        trainer.train()
        fold_time = time.time() - fold_start

        val_metrics = trainer.evaluate(val_tok)
        test_metrics = trainer.evaluate(test_tokenized)

        val_acc = val_metrics["eval_accuracy"]
        test_acc = test_metrics["eval_accuracy"]
        fold_results.append(
            {
                "fold": fold + 1,
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_metrics["eval_macro_f1"], 4),
                "test_accuracy": round(test_acc, 4),
                "test_macro_f1": round(test_metrics["eval_macro_f1"], 4),
                "training_time_min": round(fold_time / 60, 2),
            }
        )

        print(
            f"  Fold {fold + 1} — val acc: {val_acc:.4f}  test acc: {test_acc:.4f}  [{fold_time / 60:.1f} min]"
        )

        if val_acc > best_fold["val_acc"]:
            best_fold = {"fold": fold + 1, "val_acc": val_acc}
            # Save best model
            best_model_path = OUTPUT_DIR / "model"
            trainer.save_model(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))

    total_elapsed = time.time() - total_start

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Cross-Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Folds:         {N_FOLDS}")
    print(f"  Total time:    {total_elapsed / 60:.1f} minutes")
    print(f"  Best fold:     {best_fold['fold']} (val acc: {best_fold['val_acc']:.4f})")

    val_accs = [r["val_accuracy"] for r in fold_results]
    test_accs = [r["test_accuracy"] for r in fold_results]
    print(f"\n  Val acc  — mean: {np.mean(val_accs):.4f}  std: {np.std(val_accs):.4f}")
    print(f"  Test acc — mean: {np.mean(test_accs):.4f}  std: {np.std(test_accs):.4f}")

    # Evaluate best model on test
    best_model = DistilBertForSequenceClassification.from_pretrained(
        str(OUTPUT_DIR / "model")
    )
    best_model.to(device)
    best_trainer = Trainer(model=best_model, compute_metrics=compute_metrics_fn)
    best_test_metrics = best_trainer.evaluate(test_tokenized)
    test_preds = best_trainer.predict(test_tokenized)
    y_pred = np.argmax(test_preds.predictions, axis=-1)
    y_true = test_preds.label_ids

    print(
        f"\n  Best model on test set — accuracy: {best_test_metrics['eval_accuracy']:.4f}"
    )
    print(f"\n  Classification report (test set):")
    print(classification_report(y_true, y_pred, target_names=SELECTED_LABELS, digits=4))

    # ── Save results ───────────────────────────────────────────
    results = {
        "selected_classes": SELECTED_LABELS,
        "total_samples": len(train_df),
        "test_samples": len(test_dataset),
        "n_folds": N_FOLDS,
        "num_epochs": 10,
        "batch_size": 32,
        "total_training_time_min": round(total_elapsed / 60, 2),
        "per_fold": fold_results,
        "val_accuracy_mean": round(float(np.mean(val_accs)), 4),
        "val_accuracy_std": round(float(np.std(val_accs)), 4),
        "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
        "test_accuracy_std": round(float(np.std(test_accs)), 4),
        "best_fold": best_fold["fold"],
        "best_model_test_accuracy": round(best_test_metrics["eval_accuracy"], 4),
        "best_model_test_macro_f1": round(best_test_metrics["eval_macro_f1"], 4),
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'metrics.json'}")
    print(f"  Best model saved to {OUTPUT_DIR / 'model'}")


if __name__ == "__main__":
    main()
