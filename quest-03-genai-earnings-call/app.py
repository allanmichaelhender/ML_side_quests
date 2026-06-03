"""
Streamlit app for the Energy Earnings Call Analyst — results showcase.

Displays training metrics, dataset overview, and sample instruction-response
pairs from the fine-tuned TinyLlama + LoRA model. No model loading required.
"""

import json
import random
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Earnings Call Analyst",
    page_icon="📊",
    layout="wide",
)


# ── Helpers ────────────────────────────────────────────────────
@st.cache_data
def load_json(filename: str):
    path = RESULTS / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_metrics():
    return load_json("metrics.json")


@st.cache_data
def load_label_info():
    return load_json("label_info.json")


@st.cache_data
def load_instruction_pairs():
    return load_json("instruction_pairs.json")


metrics = load_metrics()
label_info = load_label_info()
pairs = load_instruction_pairs()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Energy Earnings Call Analyst")
    st.markdown("**Quest 03 — GenAI** · Results Dashboard")

    if label_info:
        st.divider()
        st.markdown("**Companies tracked:**")
        for c in label_info.get("companies", []):
            st.markdown(f"- {c}")
        st.markdown(f"**Total records:** {label_info.get('total_records', '—')}")
        st.markdown(f"**Training pairs:** {label_info.get('total_pairs', '—')}")

    if metrics:
        st.divider()
        st.markdown("**Model:** TinyLlama 1.1B + LoRA")
        st.markdown(
            f"**LoRA r={metrics.get('lora_r')}** · alpha={metrics.get('lora_alpha')}"
        )
        st.markdown(f"**Epochs:** {metrics.get('num_epochs')}")

    st.divider()
    st.caption("ML Side Quest #3 — GenAI")
    st.caption("Built with TinyLlama, LoRA, and Streamlit")


# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════

st.title("📊 Energy Earnings Call Analyst")
st.markdown(
    "Fine-tuned **TinyLlama 1.1B** with **LoRA** on SEC filing data from "
    "major energy companies."
)

# ════════════════════════════════════════════════════════════════
# 1. TRAINING SUMMARY
# ════════════════════════════════════════════════════════════════
if metrics:
    st.header("🏋️ Training Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Loss", f"{metrics['train_loss']:.4f}")
    c2.metric("Duration", f"{metrics['train_duration_min']:.0f} min")
    c3.metric("Samples", metrics["max_samples"])
    c4.metric("Epochs", metrics["num_epochs"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batch Size", metrics["batch_size"])
    c2.metric("Grad Accum", metrics["grad_accum_steps"])
    c3.metric("Learning Rate", f"{metrics['learning_rate']:.0e}")
    c4.metric("CPU Threads", metrics["num_cpu_threads"])

    with st.expander("📋 Full training config"):
        st.json(metrics)

# ════════════════════════════════════════════════════════════════
# 2. DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════
if label_info:
    st.header("📚 Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Companies")
        companies_df = {
            "Ticker": label_info.get("tickers", []),
            "Company": label_info.get("companies", []),
        }
        st.dataframe(companies_df, width="stretch", hide_index=True)

    with col2:
        st.subheader("Financial Metrics")
        for m in label_info.get("metrics", []):
            st.markdown(f"- {m}")

# ════════════════════════════════════════════════════════════════
# 3. SAMPLE INSTRUCTION-RESPONSE PAIRS
# ════════════════════════════════════════════════════════════════
if pairs:
    st.header("💬 Sample Instruction-Response Pairs")

    st.markdown(
        "The model was trained on **{:,}** instruction-response pairs "
        "generated from SEC XBRL data.".format(len(pairs))
    )

    # Random sample of 8 pairs
    random.seed(42)
    sample = random.sample(pairs, min(8, len(pairs)))

    for i, pair in enumerate(sample):
        with st.container(border=True):
            tags = f"🏷️ {pair.get('company', '—')} · {pair.get('ticker', '')} · {pair.get('metric', '—')} · {pair.get('period', '—')}"
            st.caption(tags)
            st.markdown(f"**Q:** {pair['instruction']}")
            st.markdown(f"**A:** {pair['response']}")

# ════════════════════════════════════════════════════════════════
# 4. FIGURES (if any)
# ════════════════════════════════════════════════════════════════
fig_dir = RESULTS / "figures"
figures = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []

if figures:
    st.header("📈 Training Figures")
    for fig_path in figures:
        st.image(str(fig_path), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Model weights excluded from deployment. Run `python src/train.py` locally "
    "to train, then use `python src/evaluate.py` for evaluation. "
    "See the [GitHub repo](https://github.com/allan-wojciechowski/ML_side_quests) for details."
)
