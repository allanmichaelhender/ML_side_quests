"""Quick test: what does DeepSeek return for ambiguous tickets?"""
import os, json
from dotenv import load_dotenv
load_dotenv("../.env")
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

# Load the 77 intent names
import sys
sys.path.insert(0, "src")
from llm_eval import INTENT_CATEGORIES

intents_str = "\n".join(f"  - {name}" for name in INTENT_CATEGORIES)

tickets = [
    "I need help with my account",
    "Something is wrong with my card",
    "Can you check my balance please",
    "I want to know about fees",
]

for ticket in tickets:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "Respond with JSON containing 'category' (exactly one intent name from the list) "
                    "and 'confidence_score' (float 0-1)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Classify this ticket into one of {len(INTENT_CATEGORIES)} categories.\n\n"
                    f"Available intents:\n{intents_str}\n\n"
                    f'Ticket: "{ticket}"\n\nReturn valid JSON only.'
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=150,
    )
    raw = resp.choices[0].message.content
    finish = resp.choices[0].finish_reason
    print(f'Ticket: "{ticket}"')
    print(f"  Raw: {repr(raw)}")
    print(f"  Finish: {finish}")
    if raw:
        try:
            data = json.loads(raw)
            valid = data.get("category") in INTENT_CATEGORIES
            print(f"  Parsed: {json.dumps(data)}")
            print(f"  Valid: {valid}")
        except json.JSONDecodeError as e:
            print(f"  JSON error: {e}")
    else:
        print("  EMPTY RESPONSE")
    print()
