"""
Streamlit dashboard for Quest 08 — Anomaly Detection.

Explore the Credit Card Fraud dataset interactively, compare all 5
detection methods, tune thresholds, and inspect results.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_utils import (
    download_credit_card_fraud,
    load_credit_card_fraud,
    generate_synthetic_data,
    load_nab_dataset,
    train_val_test_split,
    DEFAULT_DATA,
    DEFAULT_RESULTS,
)
from src.detectors import DETECTOR_NAMES, fit_detector, score_detector
from src.threshold import find_best_threshold, apply_threshold, evaluate_threshold
from src.visualize import (
    plot_score_distributions,
    plot_comparison_bar,
    plot_projection,
    plot_roc_curves,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anomaly Detection — Quest 08",
    page_icon="📊",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "autoencoder_model"

SAMPLE_FRAC_OPTIONS = {
    "Full (284k)": None,
    "50% (142k)": 0.5,
    "20% (57k)": 0.2,
    "10% (28k)": 0.1,
    "5% (14k)": 0.05,
}

# ── Session state ────────────────────────────────────────────────────────────

if "X_train" not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.all_scores = None
    st.session_state.results = None
    st.session_state.data_loaded = False
    st.session_state.eval_run = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Anomaly Detection")
st.sidebar.markdown("**Quest 08** — 5 methods comparison")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")

data_source = st.sidebar.selectbox(
    "Data source",
    ["Credit Card Fraud (Kaggle)", "Synthetic", "NAB (time series)"],
    index=0,
)

if data_source == "Credit Card Fraud (Kaggle)":
    sample_label = st.sidebar.selectbox(
        "Subsample",
        list(SAMPLE_FRAC_OPTIONS.keys()),
        index=3,  # default: 10%
    )
    sample_frac = SAMPLE_FRAC_OPTIONS[sample_label]
elif data_source == "Synthetic":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        syn_samples = st.number_input("Samples", min_value=500, value=5000, step=500)
    with col2:
        syn_features = st.number_input("Features", min_value=2, value=8, step=1)
    syn_contamination = st.sidebar.slider("Anomaly ratio", 0.01, 0.30, 0.05, step=0.01)
else:  # NAB
    nab_dataset = st.sidebar.selectbox(
        "NAB dataset",
        [
            "machine_temp",
            "ec2_cpu",
            "ec2_network",
            "rds_cpu",
            "nyc_taxi",
            "ambient_temp",
        ],
        index=0,
    )

load_data_btn = st.sidebar.button("🔄 Load Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Detectors")

detectors_to_run = []
for name in DETECTOR_NAMES:
    if st.sidebar.checkbox(name, value=True):
        detectors_to_run.append(name)

run_eval_btn = st.sidebar.button(
    "▶ Run Evaluation", use_container_width=True, type="primary"
)

# ── Main panel ──────────────────────────────────────────────────────────────

st.title("📊 Anomaly Detection — Method Comparison")
st.markdown(
    "Compare **Isolation Forest**, **LOF**, **One-Class SVM**, "
    "**Autoencoder (PyTorch)**, and **DBSCAN** on real-world data."
)


# ── Load data ────────────────────────────────────────────────────────────────


def load_data():
    with st.spinner("Loading data..."):
        if data_source == "Credit Card Fraud (Kaggle)":
            # Try to download if not cached
            csv_path = RESULTS_DIR.parent / "data" / "kaggle" / "creditcard.csv"
            if not csv_path.exists():
                try:
                    download_credit_card_fraud()
                except Exception as e:
                    st.warning(
                        f"Kaggle download failed: {e}. Falling back to synthetic data."
                    )
                    X, y = generate_synthetic_data(n_samples=5000)
                    return X, y, "synthetic_fallback"
            X, y = load_credit_card_fraud(path=csv_path, sample_frac=sample_frac)
            return X, y, "credit_card_fraud"

        elif data_source == "Synthetic":
            X, y = generate_synthetic_data(
                n_samples=syn_samples,
                n_features=syn_features,
                contamination=syn_contamination,
            )
            return X, y, "synthetic"

        else:  # NAB
            X, y = load_nab_dataset(nab_dataset)
            return X, y, f"nab_{nab_dataset}"


if load_data_btn:
    X, y, data_name = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.data_name = data_name
    st.session_state.data_loaded = True
    st.session_state.eval_run = False
    st.session_state.all_scores = None
    st.session_state.results = None
    st.rerun()


# ── Data overview ────────────────────────────────────────────────────────────

if st.session_state.data_loaded:
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    data_name = st.session_state.data_name

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total samples", len(X_train) + len(X_test))
    col2.metric("Features", X_train.shape[1])
    col3.metric("Test anomalies", int(y_test.sum()))
    col4.metric("Anomaly rate", f"{y_test.mean() * 100:.2f}%")

    # Show sample data
    with st.expander("📋 Sample data preview", expanded=False):
        df_sample = pd.DataFrame(
            np.vstack([X_train[:5], X_test[:5]]),
            columns=[f"f{i}" for i in range(X_train.shape[1])],
        )
        df_sample["split"] = ["train"] * 5 + ["test"] * 5
        st.dataframe(df_sample, use_container_width=True)

    # Class balance
    st.subheader("Class Balance")
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Normal", "Anomaly"]
    counts = [int((y_test == 0).sum()), int((y_test == 1).sum())]
    colors = ["#0072B2", "#D55E00"]
    bars = ax.bar(labels, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Count")
    ax.set_title(f"Test Set — {data_name}")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)
    plt.close(fig)


# ── Run evaluation ───────────────────────────────────────────────────────────

if run_eval_btn and st.session_state.data_loaded:
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    if not detectors_to_run:
        st.error("Select at least one detector.")
        st.stop()

    # Use only normal data for autoencoder
    X_train_normal = X_train[y_train == 0] if len(y_train) > 0 else X_train

    progress_bar = st.progress(0, text="Running detectors...")
    status_text = st.empty()

    results = {}
    all_scores = {}

    for i, name in enumerate(detectors_to_run):
        status_text.text(f"Fitting {name}...")
        fit_X = X_train_normal if name == "Autoencoder" else X_train
        model_dir = MODEL_DIR if name == "Autoencoder" else None

        model, fit_time = fit_detector(name, fit_X, model_dir=model_dir)

        status_text.text(f"Scoring {name}...")
        scores = score_detector(name, model, X_test)
        all_scores[name] = scores

        threshold, f1, metrics = find_best_threshold(y_test, scores)

        results[name] = {
            "fit_time_s": round(fit_time, 3),
            "score_time_s": 0.0,  # computed inline
            "threshold": round(metrics["threshold"], 6),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "n_test": len(X_test),
            "n_anomalies_test": int(y_test.sum()),
            "anomaly_rate_test": round(float(y_test.mean()), 4),
        }
        progress_bar.progress((i + 1) / len(detectors_to_run))

    progress_bar.empty()
    status_text.text("Done!")

    st.session_state.results = results
    st.session_state.all_scores = all_scores
    st.session_state.eval_run = True
    st.rerun()


# ── Results display ──────────────────────────────────────────────────────────

if st.session_state.eval_run:
    results = st.session_state.results
    all_scores = st.session_state.all_scores
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # ── Metrics table ────────────────────────────────────────────────────────
    st.subheader("📈 Performance Comparison")

    metrics_df = pd.DataFrame(results).T
    display_cols = ["f1", "precision", "recall", "threshold", "fit_time_s"]
    display_names = {
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "threshold": "Threshold",
        "fit_time_s": "Fit Time (s)",
    }
    metrics_df = metrics_df[display_cols].rename(columns=display_names)
    metrics_df.index.name = "Detector"

    # Highlight best F1
    def highlight_best(val):
        if isinstance(val, (int, float)) and val == metrics_df["F1"].max():
            return "background-color: #d4edda"
        return ""

    st.dataframe(
        metrics_df.style.applymap(highlight_best),
        use_container_width=True,
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Score Distributions", "Comparison", "Projection", "ROC Curves"]
    )

    with tab1:
        with st.spinner("Generating score distributions..."):
            fig = plot_score_distributions(all_scores, y_test)
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        with st.spinner("Generating comparison chart..."):
            fig = plot_comparison_bar(results)
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        method = st.radio("Projection method", ["umap", "tsne"], horizontal=True)
        with st.spinner(f"Running {method.upper()}..."):
            fig = plot_projection(X_test, y_test, method=method)
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        with st.spinner("Generating ROC curves..."):
            fig = plot_roc_curves(all_scores, y_test)
            st.pyplot(fig)
            plt.close(fig)

    # ── Interactive threshold explorer ──────────────────────────────────────
    st.subheader("🎛️ Interactive Threshold Explorer")

    selected_detector = st.selectbox("Select detector", list(all_scores.keys()))
    scores = all_scores[selected_detector]

    # Show score distribution with threshold line
    thresh_range = np.percentile(scores, [1, 99])
    user_threshold = st.slider(
        "Threshold",
        min_value=float(thresh_range[0]),
        max_value=float(thresh_range[1]),
        value=float(np.percentile(scores, 95)),
    )

    # Evaluate at this threshold
    user_metrics = evaluate_threshold(y_test, scores, user_threshold)
    preds = apply_threshold(scores, user_threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1", f"{user_metrics['f1']:.4f}")
    col2.metric("Precision", f"{user_metrics['precision']:.4f}")
    col3.metric("Recall", f"{user_metrics['recall']:.4f}")
    col4.metric("Anomalies detected", f"{int(preds.sum())} / {int(y_test.sum())}")

    # Confusion matrix-like visualization
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())

    cm_fig, cm_ax = plt.subplots(figsize=(4, 3))
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    cm_ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    cm_ax.set_xticks([0, 1])
    cm_ax.set_yticks([0, 1])
    cm_ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    cm_ax.set_yticklabels(["True Normal", "True Anomaly"])
    for (row, col), val in np.ndenumerate(cm_matrix):
        cm_ax.text(col, row, str(val), ha="center", va="center", fontsize=14)
    cm_ax.set_title("Confusion Matrix")
    st.pyplot(cm_fig)
    plt.close(cm_fig)

    # ── Download results ────────────────────────────────────────────────────
    st.subheader("💾 Export Results")

    col1, col2 = st.columns(2)
    metrics_json = pd.DataFrame(results).to_json(orient="index", indent=2)
    col1.download_button(
        "📥 Download metrics (JSON)",
        data=metrics_json,
        file_name="anomaly_metrics.json",
        mime="application/json",
    )

    scores_df = pd.DataFrame(all_scores)
    scores_df["true_label"] = y_test
    col2.download_button(
        "📥 Download scores (CSV)",
        data=scores_df.to_csv(index=False),
        file_name="anomaly_scores.csv",
        mime="text/csv",
    )

else:
    st.info(
        "👈 Load a dataset from the sidebar, then click "
        "**▶ Run Evaluation** to compare detectors."
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small>Quest 08 — Anomaly Detection | "
    "Methods: Isolation Forest, LOF, One-Class SVM, Autoencoder (PyTorch), DBSCAN</small>",
    unsafe_allow_html=True,
)
