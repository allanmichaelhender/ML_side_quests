import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from data_utils import (
    load_amazon_reviews,
    get_tokenizer,
    tokenize_dataset,
    save_label_info,
    LABEL_NAMES,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    import sklearn.metrics as metrics

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def train(
    data_dir: Path = DEFAULT_DATA,
    output_dir: Path = DEFAULT_OUTPUT,
    max_samples: int = 10_000,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load & prepare dataset ──────────────────────────────────
    dataset = load_amazon_reviews(data_dir=data_dir, max_samples=max_samples, seed=seed)
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    # ── Save label info ─────────────────────────────────────────
    save_label_info(output_dir)

    # ── Build model ─────────────────────────────────────────────
    num_labels = len(LABEL_NAMES)

    # DistilBertForSequenceClassification is the DistilBERT transformer plus a classification head to classifiy the vectors, we initialise with the pretrained base weights
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},  # mapping for id to labels
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},  # mapping for labels to id
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Training arguments ──────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"distilbert-sentiment-{int(time.time())}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),  # where to save checkpoints
        run_name=run_name,  # name for this training run 
        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",  # save checkpoint after each epoch
        logging_strategy="steps",  # log metrics every N steps
        logging_steps=50,  # log every 50 steps
        learning_rate=learning_rate,  # peak learning rate
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size
        * 2,  # eval batch size (larger = faster eval)
        num_train_epochs=num_epochs,  # number of full passes through training data
        weight_decay=0.01,  # L2 regularization to prevent overfitting
        warmup_ration=0.1,  # LR linearly increases for first N steps
        lr_scheduler_type="linear",  # LR decays linearly after warmup
        optim="adamw_torch",  # optimizer: AdamW with decoupled weight decay
        load_best_model_at_end=True,  # restore the best checkpoint after training
        metric_for_best_model="accuracy",  # use accuracy to pick the best checkpoint
        greater_is_better=True,  # higher accuracy = better model
        save_total_limit=2,  # keep only the 2 most recent checkpoints
        report_to="none",  # don't log to external services
        seed=seed,  # random seed for reproducibility
        dataloader_num_workers=0,  # data loading processes (0 = main process)
        fp16=False,  # don't use half-precision (CPU-safe)
    )

    # ── Trainer ─────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # This monitors the evaluation metric (accuracy) each epoch and stops training early if it stops improving. patience=2 means: if accuracy doesn't improve for 2 consecutive evaluations, stop training.
    )

    # ── Train ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(
        f"Starting fine-tuning — {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}"
    )
    print(f"{'=' * 60}")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    # ── Save model ──────────────────────────────────────────────
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"Model saved to {model_path}")

    # Export to ONNX for efficient inference
    _export_to_onnx(model, tokenizer, output_dir, max_length)

    # ── Save training metrics ───────────────────────────────────
    train_metrics = train_result.metrics
    train_metrics["train_duration_min"] = round(elapsed / 60, 2)
    train_metrics["max_samples"] = max_samples

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(train_metrics)
        train_metrics = existing

    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ── Final evaluation ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Final evaluation on validation set:")
    eval_results = trainer.evaluate()
    print(f"  Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation macro F1: {eval_results['eval_macro_f1']:.4f}")

    return model, tokenizer, eval_results


def _export_to_onnx(model, tokenizer, output_dir: Path, max_length: int = 256):
    """Export trained model to ONNX format for lightweight inference."""
    try:
        from transformers.onnx import export
        from transformers import PipelinesConfig, AutoConfig

        onnx_path = output_dir / "model.onnx"
        print(f"\nExporting to ONNX: {onnx_path}")

        # Use the built-in ONNX export
        from transformers.onnx import export
        from pathlib import Path as P
        import tempfile
        import os

        # Save model and tokenizer to temp dir for ONNX export
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = P(tmp)
            model.save_pretrained(tmp_path)
            tokenizer.save_pretrained(tmp_path)

            from transformers import AutoModelForSequenceClassification

            model_loaded = AutoModelForSequenceClassification.from_pretrained(tmp_path)

            from transformers.onnx import export
            from transformers.onnx import FeaturesManager

            # Use the ONNX export feature
            feature = "sequence-classification"
            model_kind, model_onnx_config = (
                FeaturesManager.check_supported_model_or_raise(
                    model_loaded, feature=feature
                )
            )
            onnx_config = model_onnx_config(model_loaded.config)

            # export
            from transformers.onnx import export as onnx_export

            onnx_inputs, onnx_outputs = onnx_export(
                preprocessor=tokenizer,
                model=model_loaded,
                config=onnx_config,
                opset=14,
                output=onnx_path,
            )
        print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped ({e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for sentiment analysis"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-samples", type=int, default=10_000, help="Samples to use for training"
    )
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seed=args.seed,
    )
