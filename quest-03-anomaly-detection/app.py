"""
Streamlit dashboard for Quest 03 — Anomaly Detection.

Loads pre-computed results from batch_evaluate.py — no live inference.
Shows useless-detector comparison, recall vs precision tradeoff plot.
"""

import json
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anomaly Detection — Quest 03",
    page_icon="📊",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "comparison_results.json"

# ── Load saved results ───────────────────────────────────────────────────────

if not RESULTS_PATH.exists():
    st.error(
        f"No pre-computed results found at `{RESULTS_PATH}`. "
        "Run `python batch_evaluate.py` first."
    )
    st.stop()

with open(RESULTS_PATH) as f:
    data = json.load(f)

train_set = data["train_set"]
test_set = data["test_set"]
useless = data["useless_detectors"]
threshold_data = data["threshold_comparison"]

# ── Layout order: AE, then XGBoost, then Hybrid ──────────────────────────────

ALL_KEYS_ORDERED = []
for k in threshold_data:
    if k == "Autoencoder":
        ALL_KEYS_ORDERED.append(k)
        break
for t in [0.2, 0.3, 0.4, 0.5]:
    ALL_KEYS_ORDERED.append(f"XGBoost_{t}")
for t in [0.2, 0.3, 0.4, 0.5]:
    ALL_KEYS_ORDERED.append(f"Hybrid_{t}")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Anomaly Detection")
st.sidebar.markdown("**Quest 03** — Pre-computed results")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")

st.sidebar.markdown("**Training Set**")
st.sidebar.metric("Samples", f"{train_set['n_samples']:,}")
st.sidebar.metric("Anomalies", f"{train_set['n_anomalies']}")
st.sidebar.metric("Rate", f"{train_set['anomaly_rate'] * 100:.3f}%")

st.sidebar.markdown("**Test Set**")
st.sidebar.metric("Samples", f"{test_set['n_samples']:,}")
st.sidebar.metric("Anomalies", f"{test_set['n_anomalies']}")
st.sidebar.metric("Rate", f"{test_set['anomaly_rate'] * 100:.3f}%")

st.sidebar.metric("Features", test_set["n_features"])

# ── Main panel ──────────────────────────────────────────────────────────────

st.title("📊 Anomaly Detection — Method Comparison")
st.markdown(
    "Pre-computed results on the **Credit Card Fraud** dataset (284k transactions). "
    "No live model inference."
)

# ═══════════════════════════════════════════════════════════════════════════
#   SECTION 1 — Useless Detectors
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("❌ Unsupervised Detectors — Not Viable on This Dataset")

useless_df = pd.DataFrame(useless).T[["precision", "recall", "f1"]]
useless_df.index.name = "Detector"
useless_df.columns = ["Precision", "Recall", "F1"]

st.dataframe(useless_df.style.format("{:.4f}"), use_container_width=True)

st.markdown(
    "**Isolation Forest**, **LOF**, and **DBSCAN** all perform poorly. We therefore focused on two options, NN Autoencoder and XGBoost."
)

st.markdown(
    "We tested XGBoost with 4 threshold values for positive classification. We will also consider using XGBoost in concert with the Autoencoder, simply classifiying a row as positive whenever either model returns a positive classification."
)


# ═══════════════════════════════════════════════════════════════════════════
#   SECTION 2 — Recall vs Precision Plot (9 options)
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("🎯 Recall vs Precision — Threshold Tradeoff")

st.markdown(
    "**Precision** = of all flagged anomalies, how many were real?\n\n"
    "**Recall** = of all real anomalies, how many did we catch?\n\n"
    "$\\text{Precision} = \\frac{TP}{TP + FP}$ "
    "$\\text{Recall} = \\frac{TP}{TP + FN}$"
)

plot_points = []
for key in ALL_KEYS_ORDERED:
    m = threshold_data[key]
    plot_points.append(
        {
            "key": key,
            "label": m["label"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "threshold": m["threshold"],
            "group": (
                "Autoencoder"
                if key == "Autoencoder"
                else "XGBoost"
                if key.startswith("XGBoost")
                else "Hybrid"
            ),
        }
    )

df_plot = pd.DataFrame(plot_points)

color_map = {"Autoencoder": "#E24A33", "XGBoost": "#348ABD", "Hybrid": "#8EBA42"}
marker_map = {"Autoencoder": "s", "XGBoost": "o", "Hybrid": "^"}
size_map = {"Autoencoder": 180, "XGBoost": 120, "Hybrid": 120}

fig, ax = plt.subplots(figsize=(10, 7))

for group in ["Autoencoder", "XGBoost", "Hybrid"]:
    subset = df_plot[df_plot["group"] == group]
    ax.scatter(
        subset["recall"],
        subset["precision"],
        s=size_map[group],
        c=color_map[group],
        marker=marker_map[group],
        label=group,
        edgecolors="white",
        linewidth=0.8,
        zorder=3,
    )
    for _, row in subset.iterrows():
        short = row["label"].replace("XGBoost", "XGB").replace("Autoencoder", "AE")
        ax.annotate(
            short,
            (row["recall"], row["precision"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            alpha=0.8,
        )

# F1 contour lines
recall_vals = np.linspace(0, 1, 200)
precision_vals = np.linspace(0, 1, 200)
R, P_grid = np.meshgrid(recall_vals, precision_vals)
F = 2 * P_grid * R / (P_grid + R + 1e-10)
contours = ax.contour(
    R, P_grid, F, levels=[0.7, 0.8, 0.85], colors="grey", alpha=0.3, linewidths=0.8
)
ax.clabel(contours, inline=True, fontsize=8, fmt="F1=%.2f")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_xlim(0.55, 0.95)
ax.set_ylim(0.55, 0.95)
ax.grid(True, alpha=0.3)
ax.legend(title="Method", fontsize=10)
ax.set_title("Recall vs Precision — Threshold Sweep", fontsize=14)

st.pyplot(fig)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
#   SECTION 3 — Full metrics table (all 9 options)
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("📈 Full Comparison Table")

metrics_rows = []
for key in ALL_KEYS_ORDERED:
    m = threshold_data[key]
    metrics_rows.append(
        {
            "Method": m["label"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
        }
    )

full_df = pd.DataFrame(metrics_rows)


st.dataframe(
    full_df.style.format(
        {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ═══════════════════════════════════════════════════════════════════════════
#   SECTION 4 — Key takeaways
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("💡 Key Takeaways")

best_entry = max(threshold_data.values(), key=lambda x: x["f1"])
best_label = best_entry["label"]

st.markdown(
    f"- **Best F1**: **{best_label}** — F1 = **{best_entry['f1']:.4f}** "
    f"(Precision={best_entry['precision']:.4f}, Recall={best_entry['recall']:.4f})\n"
    f"- If **recall is the priority**, XGBoost at t=0.2 catches **86.9% of frauds** "
    f"while keeping precision at 66.2% — 2 in 3 flagged transactions are genuine fraud.\n"
    f"- The Hybrid OR-gate boosts recall slightly (87.9% at t=0.2) but precision drops "
    f"to 58.8%, the Autoencoder adding noise.\n"
    f"- Relative to these results, all 3 unsupervised methods (IF, LOF, DBSCAN) are non-viable — "
    f"supervised learning is mandatory for this problem."
)

st.markdown("---")
st.markdown(
    "<small>Quest 03 — Anomaly Detection | "
    "Methods: Isolation Forest, LOF, One-Class SVM, Autoencoder (PyTorch), DBSCAN</small>",
    unsafe_allow_html=True,
)
