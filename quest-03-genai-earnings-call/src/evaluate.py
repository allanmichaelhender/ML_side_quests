"""
Evaluate the fine-tuned TinyLlama + LoRA earnings call analyst model.

Generates sample responses and computes perplexity on a test set.
"""

import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "results"
DEFAULT_DATA = PROJECT / "data"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Test prompts ────────────────────────────────────────────────
TEST_PROMPTS = [
    {
        "instruction": "What was Exxon Mobil's revenue for FY2024?",
        "input": "",
    },
    {
        "instruction": "Summarize the key financial results for Chevron for FY2024 based on their SEC filing.",
        "input": "",
    },
    {
        "instruction": "Analyze the financial performance of ConocoPhillips, including revenue, net income, and total assets.",
        "input": "",
    },
    {
        "instruction": "Compare the revenue and net income of Exxon Mobil and Chevron.",
        "input": "",
    },
    {
        "instruction": "What are the key financial metrics I should look at when evaluating an energy company?",
        "input": "",
    },
]


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format prompt in TinyLlama's chat format."""
    if input_text:
        return f"""<|system|>
You are a financial analyst specializing in energy company earnings. Answer questions accurately based on SEC filing data.</s>
<|user|>
{instruction}

{input_text}</s>
<|assistant|>"""
    else:
        return f"""<|system|>
You are a financial analyst specializing in energy company earnings. Answer questions accurately based on SEC filing data.</s>
<|user|>
{instruction}</s>
<|assistant|>"""


def load_model(
    output_dir: Path = DEFAULT_OUTPUT,
    num_cpu_threads: int = 12,
):
    """Load the fine-tuned model and tokeniser."""
    torch.set_num_threads(num_cpu_threads)
    model_path = output_dir / "model"

    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the fine-tuned model."""
    prompt = format_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    if "<|assistant|>" in full_text:
        response = full_text.split("<|assistant|>")[-1].strip()
    else:
        response = full_text[len(prompt) :].strip()

    return response


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate(
    output_dir: Path = DEFAULT_OUTPUT,
    num_cpu_threads: int = 12,
):
    """Run full evaluation: generate test responses and compute metrics."""
    print("=" * 60)
    print("[DATA] Earnings Call Analyst — Evaluation")
    print("=" * 60)

    # Load model
    print("\n[CACHE] Loading model...")
    model, tokenizer = load_model(output_dir, num_cpu_threads)

    # ── Generate test responses ─────────────────────────────────
    print("\n" + "-" * 60)
    print("[CHAT] Test Generations")
    print("-" * 60)

    results = []
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n[NOTE] Test {i}: {test['instruction'][:80]}...")
        start = time.time()

        response = generate_response(
            model,
            tokenizer,
            test["instruction"],
            test["input"],
        )

        elapsed = time.time() - start
        print(f"   ⏱️  {elapsed:.1f}s")
        print(f"   Response: {response[:200]}...")

        results.append(
            {
                "instruction": test["instruction"],
                "response": response,
                "generation_time_s": round(elapsed, 2),
            }
        )

    # ── Compute perplexity on test prompts ──────────────────────
    print(f"\n{'=' * 60}")
    print("[CHART] Computing perplexity...")
    test_texts = [format_prompt(t["instruction"], t["input"]) for t in TEST_PROMPTS]
    perplexity = compute_perplexity(model, tokenizer, test_texts)
    print(f"   Perplexity: {perplexity:.4f}")

    # ── Save results ────────────────────────────────────────────
    eval_results = {
        "perplexity": round(perplexity, 4),
        "num_test_prompts": len(TEST_PROMPTS),
        "generations": results,
    }

    eval_path = output_dir / "eval_results.json"
    # Merge with existing if present
    if eval_path.exists():
        with open(eval_path) as f:
            existing = json.load(f)
        existing.update(eval_results)
        eval_results = existing

    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[OK] Results saved to {eval_path}")

    return eval_results


if __name__ == "__main__":
    evaluate()
