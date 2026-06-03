"""
Streamlit app for the Energy Earnings Call Analyst.

Loads the fine-tuned TinyLlama + LoRA model and provides a chat interface
for querying energy company financial data.
"""

import json
import sys
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

RESULTS = HERE / "results"
MODEL_PATH = RESULTS / "model"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Earnings Call Analyst",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the fine-tuned model (cached)."""
    torch.set_num_threads(12)

    if not MODEL_PATH.exists():
        st.warning(
            "⚠️  No fine-tuned model found. Please run `python src/train.py` first. "
            "Using base model for demo purposes."
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        with st.spinner("Loading fine-tuned model..."):
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format in TinyLlama chat style."""
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


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response."""
    formatted = format_prompt(prompt)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in full:
        return full.split("<|assistant|>")[-1].strip()
    return full[len(formatted) :].strip()


# ── Load label info ────────────────────────────────────────────
def load_label_info():
    path = RESULTS / "label_info.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "companies": [
            "Exxon Mobil",
            "Chevron",
            "ConocoPhillips",
            "EOG Resources",
            "Pioneer Natural Resources",
            "Occidental Petroleum",
            "Schlumberger",
            "Baker Hughes",
        ],
        "metrics": [
            "Revenue",
            "Net Income",
            "Total Assets",
            "Total Liabilities",
            "Operating Income",
            "Earnings Per Share",
            "Cash and Equivalents",
            "Long Term Debt",
            "Gross Profit",
        ],
    }


# ── UI ──────────────────────────────────────────────────────────
st.title("📊 Energy Earnings Call Analyst")
st.markdown(
    "Ask questions about energy company financials. The model is fine-tuned on "
    "SEC filing data from major energy companies using TinyLlama + LoRA."
)

label_info = load_label_info()

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Model:** TinyLlama 1.1B + LoRA\n\n"
        "**Data:** SEC XBRL financial data from energy companies\n\n"
        "**Companies tracked:**"
    )
    for c in label_info.get("companies", []):
        st.markdown(f"- {c}")

    st.markdown("**Metrics:**")
    for m in label_info.get("metrics", []):
        st.markdown(f"- {m}")

    st.divider()
    st.caption("ML Side Quest #3 — GenAI")

# Load model
model, tokenizer = load_model()

# ── Example queries ─────────────────────────────────────────────
with st.expander("💡 Example queries", expanded=False):
    examples = [
        "What was Exxon Mobil's revenue for FY2024?",
        "Summarize Chevron's financial performance for FY2024.",
        "Compare the net income of ConocoPhillips and EOG Resources.",
        "What are the total assets of Occidental Petroleum?",
        "Analyze the debt levels of Schlumberger.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

# ── Chat interface ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle pre-filled query
query = st.session_state.get("query", "")
if query:
    st.session_state.pop("query", None)
    # Display as user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = generate(model, tokenizer, query)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if prompt := st.chat_input("Ask about energy company financials..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = generate(model, tokenizer, prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# ── Generation controls ────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max new tokens", 64, 512, 256, 32)
