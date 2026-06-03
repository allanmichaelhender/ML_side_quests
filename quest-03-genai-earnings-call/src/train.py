"""
Fine-tune TinyLlama 1.1B with LoRA for energy earnings call analysis.

Optimised for CPU training on Ryzen 5600X — targets 2-3 hours for 500 samples.
"""

import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset, DatasetDict

from data_utils import (
    prepare_dataset,
    format_for_tinyllama,
    ENERGY_COMPANIES,
    FINANCIAL_METRICS,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA_DIR = PROJECT / "data"
DEFAULT_OUTPUT_DIR = PROJECT / "results"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def compute_metrics(eval_pred):
    """Compute perplexity for evaluation.

    Note: eval_pred contains numpy arrays (not torch tensors) in newer transformers.
    """
    logits, labels = eval_pred
    # Convert to torch tensors for loss computation
    logits_t = torch.from_numpy(logits[:, :-1, :])
    labels_t = torch.from_numpy(labels[:, 1:])

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        logits_t.reshape(-1, logits_t.size(-1)),
        labels_t.reshape(-1),
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity, "eval_loss": loss.item()}


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    # ── Dataset ─────────────────────────────────────────────────
    max_samples: int = 500,
    max_length: int = 512,
    val_ratio: float = 0.1,
    # ── LoRA hyperparameters ────────────────────────────────────
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    # ── Training ────────────────────────────────────────────────
    num_epochs: int = 2,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 20,
    weight_decay: float = 0.01,
    logging_steps: int = 5,
    save_steps: int = 50,
    # ── CPU optimisation ────────────────────────────────────────
    num_cpu_threads: int = 12,
    seed: int = 42,
):
    # ── CPU threading ───────────────────────────────────────────
    torch.set_num_threads(num_cpu_threads)
    print(f"[THREAD] CPU threads: {torch.get_num_threads()}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════
    # 1. PREPARE DATASET
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[DATA] Step 1: Prepare dataset")
    print("=" * 60)
    pairs, label_info = prepare_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        max_samples=max_samples,
        seed=seed,
    )

    # ── Load tokeniser ──────────────────────────────────────────
    print(f"\n[NOTE] Loading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Format & tokenize ───────────────────────────────────────
    print("\n🔤 Formatting and tokenizing...")
    formatted_texts = format_for_tinyllama(pairs)
    print(f"   {len(formatted_texts)} formatted examples")

    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    split = tokenized.train_test_split(test_size=val_ratio, seed=seed)
    tokenized_dataset = DatasetDict(train=split["train"], validation=split["test"])
    print(f"   Train: {len(tokenized_dataset['train'])} samples")
    print(f"   Validation: {len(tokenized_dataset['validation'])} samples")

    # ════════════════════════════════════════════════════════════
    # 2. LOAD BASE MODEL & APPLY LoRA
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🦙 Step 2: Load TinyLlama + LoRA")
    print("=" * 60)

    print(f"\nLoading base model: {MODEL_NAME}")
    load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU requires float32
        low_cpu_mem_usage=True,
        use_cache=False,  # Disable KV cache during training
    )
    model.to(device)
    print(f"   Loaded in {time.time() - load_start:.1f}s")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── LoRA configuration (aggressive — r=4, q_proj+v_proj only) ─
    print("\n[CONFIG] Configuring LoRA...")
    target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected: ~1.1M trainable params out of ~1.1B (0.1%)

    # ════════════════════════════════════════════════════════════
    # 3. TRAINING
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[START] Step 3: Training")
    print("=" * 60)

    run_name = f"tinyllama-earnings-lora-r{lora_r}-{int(time.time())}"
    checkpoint_dir = output_dir / "checkpoints"

    # Effective batch size = batch_size * grad_accum_steps = 4
    # With 500 samples, 2 epochs = 1000 steps / 4 = 250 optimizer steps
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        run_name=run_name,
        # ── Eval / save ─────────────────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        # ── Optimizer ───────────────────────────────────────────
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        # ── Speed ───────────────────────────────────────────────
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        # ── Misc ────────────────────────────────────────────────
        report_to="none",
        seed=seed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # ── Estimate time ───────────────────────────────────────────
    num_train = len(tokenized_dataset["train"])
    steps_per_epoch = math.ceil(num_train / (batch_size * grad_accum_steps))
    total_steps = steps_per_epoch * num_epochs
    print(f"\n[DATA] Training stats:")
    print(f"   Train samples: {num_train}")
    print(f"   Validation samples: {len(tokenized_dataset['validation'])}")
    print(f"   Batch size (per device): {batch_size}")
    print(f"   Gradient accumulation: {grad_accum_steps}")
    print(f"   Effective batch size: {batch_size * grad_accum_steps}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Max sequence length: {max_length}")
    print(f"   LoRA rank: r={lora_r}")
    print(f"   Target modules: {target_modules}")
    print(f"   CPU threads: {num_cpu_threads}")
    print(f"\n⏱️  Expected time: ~1-1.5 hours per epoch → ~2-3 hours total")

    # ── Train ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[FIRE] Starting fine-tuning...")
    print(f"{'=' * 60}")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(
        f"\n[OK] Training completed in {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)"
    )

    # ════════════════════════════════════════════════════════════
    # 4. SAVE
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[SAVE] Step 4: Saving model & metrics")
    print("=" * 60)

    # Save LoRA adapter (small — just a few MB)
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"   Model saved to {model_path}")

    # Save training metrics
    train_metrics = train_result.metrics
    train_metrics.update(
        {
            "train_duration_min": round(elapsed / 60, 2),
            "train_duration_hours": round(elapsed / 3600, 2),
            "max_samples": max_samples,
            "max_length": max_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "num_cpu_threads": num_cpu_threads,
            "model_name": MODEL_NAME,
        }
    )

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(train_metrics)
        train_metrics = existing

    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"   Metrics saved to {metrics_path}")

    # ── Final eval ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[CHART] Final evaluation:")
    eval_results = trainer.evaluate()
    print(f"   Perplexity: {eval_results.get('eval_perplexity', 'N/A'):.4f}")
    print(f"   Eval loss:  {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"{'=' * 60}")

    return model, tokenizer, eval_results


if __name__ == "__main__":
    train()
