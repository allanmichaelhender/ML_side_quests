"""
LLM evaluation for Support Ticket Routing.

Evaluates DeepSeek (or any OpenAI-compatible API) zero-shot
on the Banking77 test set using structured outputs (Pydantic + response_format).
"""

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LABEL_INFO_PATH = PROJECT / "results" / "label_info.json"

# Load intent categories once at module level
if LABEL_INFO_PATH.exists():
    with open(LABEL_INFO_PATH) as f:
        _info = json.load(f)
    INTENT_CATEGORIES: list[str] = _info["label_names"]
else:
    # Fallback: load from Hugging Face dataset
    from datasets import load_dataset

    _ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    INTENT_CATEGORIES = _ds.features["label"].names

# Dynamic Literal type — Pydantic v2 validates against this at runtime
IntentCategory = Literal[tuple(INTENT_CATEGORIES)]


class TicketClassification(BaseModel):
    """Structured response from the LLM for ticket routing.

    category must be exactly one of the {len(INTENT_CATEGORIES)} Banking77 intent names.
    """

    category: IntentCategory
    confidence_score: float


def normalize(s: str) -> str:
    """Normalize a string for fuzzy matching: lowercase, underscores/hyphens to spaces."""
    return s.lower().replace("_", " ").replace("-", " ").strip()


def evaluate_deepseek(
    test_texts: list,
    test_labels: list,
    label_names: list,
    api_key: str,
    model: str = "deepseek-v4-flash",
    max_retries: int = 2,
) -> tuple:
    """Evaluate DeepSeek zero-shot on test samples using structured outputs.

    Uses Pydantic + response_format to enforce valid category responses.
    Returns (predictions, true_labels, confidence_scores).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    intents_str = "\n".join(f"  - {name}" for name in label_names)

    system_prompt = (
        "You are a precise ticket routing classifier. "
        "Respond with a JSON object containing 'category' (exactly one of the intent names listed) "
        "and 'confidence_score' (a float between 0 and 1)."
    )
    user_prompt = f"""Classify this ticket into exactly ONE of the following {len(label_names)} categories.

Available intents:
{intents_str}

Rules:
- "category" must be the exact intent name from the list above
- "confidence_score" must be a float between 0 and 1

Ticket: "{{ticket}}"

Return valid JSON only."""

    predictions = []
    confidences = []
    failures = []  # collect a few rejections for inspection
    total = len(test_texts)
    matched = 0

    for i, text in enumerate(test_texts):
        prompt = user_prompt.format(ticket=text)
        success = False

        for attempt in range(max_retries + 1):
            raw = ""
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=100,
                )
                raw = response.choices[0].message.content.strip()

                # Parse with Pydantic (Literal enforces exact category match)
                data = json.loads(raw)
                classification = TicketClassification(**data)
                raw_category = classification.category.strip().lower()
                confidence = max(0.0, min(1.0, classification.confidence_score))

                # Match category against label_names
                raw_norm = normalize(raw_category)
                pred = None
                for idx, name in enumerate(label_names):
                    if normalize(name) == raw_norm:
                        pred = idx
                        break
                if pred is None:
                    for idx, name in enumerate(label_names):
                        name_norm = normalize(name)
                        if raw_norm in name_norm or name_norm in raw_norm:
                            pred = idx
                            break

                if pred is not None:
                    predictions.append(pred)
                    confidences.append(confidence)
                    matched += 1
                    success = True
                    break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                elif len(failures) < 5:
                    # Save first few failures for debugging
                    failures.append(
                        {
                            "ticket": text[:80],
                            "raw_response": raw[:200] if raw else "no response",
                            "error": str(e)[:100],
                        }
                    )

        if not success:
            predictions.append(-1)
            confidences.append(0.0)

        if (i + 1) % 50 == 0:
            print(f"  DeepSeek: {i + 1}/{total} classified ({matched}/{i + 1} matched)")

    print(f"  DeepSeek done: {matched}/{total} successfully matched")
    if failures:
        print(f"\n  Sample rejections ({len(failures)} shown):")
        for f in failures:
            print(f'    Ticket: "{f["ticket"]}..."')
            print(f"    Returned: {f['raw_response']}")
            print(f"    Error: {f['error']}")
            print()
    return np.array(predictions), np.array(test_labels), np.array(confidences)
