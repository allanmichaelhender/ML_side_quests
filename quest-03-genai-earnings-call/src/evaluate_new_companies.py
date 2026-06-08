"""
Evaluate fine-tuned TinyLlama + LoRA on Marathon Petroleum (MPC),
an energy company NOT seen during training. Uses real SEC 10-K text as context.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "results"

sys.path.insert(0, str(HERE))
from data_utils import FINANCIAL_METRICS, fetch_financial_data, download_filing_texts, _format_currency

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

NEW_COMPANIES = [{"name": "Marathon Petroleum", "ticker": "MPC", "cik": 1510295}]
CACHE_DIR = PROJECT / "data" / "cache"


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return f"""<|system|>
You are a financial analyst specializing in energy company earnings. Answer questions accurately based on SEC filing data.</s>
<|user|>
{instruction}

{input_text}</s>
<|assistant|>"""
    return f"""<|system|>
You are a financial analyst specializing in energy company earnings. Answer questions accurately based on SEC filing data.</s>
<|user|>
{instruction}</s>
<|assistant|>"""


def load_model(output_dir: Path = RESULTS, num_cpu_threads: int = 12):
    torch.set_num_threads(num_cpu_threads)
    model_path = output_dir / "model"
    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full else full[len(prompt):].strip()


def extract_dollar_value(text: str) -> float | None:
    m = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(B|b|M|m|K|k|T|t)?', text)
    if m:
        n = float(m.group(1).replace(",", ""))
        u = (m.group(2) or "").upper()
        return n * {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}.get(u, 1)
    return None


def evaluate():
    print("=" * 60)
    print("Generalization Test: Marathon Petroleum (unseen during training)")
    print("=" * 60)

    # Fetch real SEC filing text + financial data
    print("\nFetching SEC data for MPC...")
    records = fetch_financial_data(companies=NEW_COMPANIES, metrics=FINANCIAL_METRICS, max_fiscal_years=2, cache_dir=CACHE_DIR)
    print(f"Fetched {len(records)} financial records")

    filing_texts = download_filing_texts(companies=NEW_COMPANIES, form_types=("10-K",), max_filings_per_company=1, cache_dir=CACHE_DIR)
    filing_text = filing_texts[0]["text"] if filing_texts else ""
    print(f"Filing text: {len(filing_text):,} chars")

    # Test on FY2025 metrics (latest full year)
    test_cases = []
    for rec in records:
        if rec.get("fy") != 2025:
            continue
        metric = rec["metric"]
        formatted = _format_currency(rec["value"])
        instruction = f"What was Marathon Petroleum's {metric.lower()} for FY2025?"
        test_cases.append({
            "company": "Marathon Petroleum", "ticker": "MPC", "metric": metric,
            "period": "FY2025", "instruction": instruction,
            "filing_context": filing_text,
            "expected_answer": f"For FY2025, Marathon Petroleum (MPC) reported {metric.lower()} of {formatted}.",
            "expected_value": rec["value"],
        })

    print(f"Created {len(test_cases)} test prompts")

    model, tokenizer = load_model()
    results = []
    correct, close, total = 0, 0, len(test_cases)

    for tc in test_cases:
        prompt = format_prompt(tc["instruction"], tc["filing_context"])
        print(f"\n  Testing: MPC - {tc['metric']}")
        start = time.time()
        response = generate_response(model, tokenizer, prompt)
        elapsed = time.time() - start

        extracted = extract_dollar_value(response)
        exp_val = tc["expected_value"]
        is_correct = rel_err = None
        if extracted and exp_val:
            rel_err = abs(extracted - exp_val) / exp_val
            is_correct = rel_err < 0.15
            if is_correct:
                correct += 1
                if rel_err < 0.05:
                    close += 1

        status = "✅" if is_correct else "❌"
        print(f"    Expected: {tc['expected_answer']}")
        print(f"    Got:      {response[:150]}")
        print(f"    Match:    {status} ({elapsed:.1f}s)")

        results.append({
            "company": tc["company"], "ticker": tc["ticker"], "metric": tc["metric"],
            "period": tc["period"], "instruction": tc["instruction"],
            "expected_answer": tc["expected_answer"], "model_response": response,
            "expected_value": exp_val, "extracted_value": extracted,
            "relative_error": round(rel_err, 4) if rel_err else None,
            "is_correct": is_correct, "generation_time_s": round(elapsed, 2),
        })

    accuracy = correct / total if total else 0
    close_acc = close / total if total else 0
    print(f"\n{'=' * 60}")
    print(f"RESULTS on unseen company: Marathon Petroleum (MPC)")
    print(f"  {correct}/{total} within 15%  ({accuracy:.1%})")
    print(f"  {close}/{total} within 5%    ({close_acc:.1%})")
    print(f"{'=' * 60}")

    eval_results = {"generalization_test": {
        "test_company": "Marathon Petroleum (MPC)", "num_test_cases": total,
        "accuracy_within_15pct": round(accuracy, 4), "accuracy_within_5pct": round(close_acc, 4),
        "details": results,
    }}

    eval_path = RESULTS / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            existing = json.load(f)
        existing.update(eval_results)
        eval_results = existing
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to {eval_path}")


if __name__ == "__main__":
    evaluate()
