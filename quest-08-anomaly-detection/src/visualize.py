"""
Visualization utilities for Quest 08 — Anomaly Detection.

Produces:
  - Score distribution histograms (per detector, split by normal vs anomaly)
  - Comparison bar charts (F1, precision, recall, latency across methods)
  - t-SNE / UMAP projection with anomaly overlay
  - ROC curves
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.size"] = 11

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES_DIR = PROJECT / "results" / "figures"

# Colour palette for detectors
DETECTOR_COLORS = {
    "Isolation Forest": "#0072B2",
    "LOF": "#D55E00",
    "One-Class SVM": "#CC79A7",
    "Autoencoder": "#009E73",
    "DBSCAN": "#F0E442",
}
DEFAULT_COLOR = "#999999"


def _get_color(name: str) -> str:
    return DETECTOR_COLORS.get(name, DEFAULT_COLOR)


# =========================================================================
#  Score Distribution
# =========================================================================


def plot_score_distributions(
    all_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_bins: int = 80,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot histogram of anomaly scores for normal vs anomalous samples.

    One subplot per detector. Normal scores in blue, anomaly scores in red.

    Args:
        all_scores: dict[name -> (n_samples,) scores]
        y_true: Ground-truth labels.
        n_bins: Number of histogram bins.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    n_detectors = len(all_scores)
    n_cols = min(3, n_detectors)
    n_rows = int(np.ceil(n_detectors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    is_anom = y_true == 1

    for idx, (name, scores) in enumerate(all_scores.items()):
        ax = axes[idx]
        color = _get_color(name)

        # Normal scores
        if is_anom.sum() < len(y_true):
            ax.hist(
                scores[~is_anom],
                bins=n_bins,
                alpha=0.6,
                color="#0072B2",
                label=f"Normal ({len(scores[~is_anom])})",
                density=True,
            )
        # Anomaly scores
        if is_anom.sum() > 0:
            ax.hist(
                scores[is_anom],
                bins=n_bins,
                alpha=0.6,
                color="#D55E00",
                label=f"Anomaly ({len(scores[is_anom])})",
                density=True,
            )

        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_detectors, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Anomaly Score Distributions by Detector", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
#  Comparison Bar Chart
# =========================================================================


def plot_comparison_bar(
    results: Dict[str, dict],
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Grouped bar chart comparing F1, precision, recall, and latency.

    Args:
        results: dict[name -> {f1, precision, recall, fit_time_s, score_time_s}]
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: F1, Precision, Recall
    metrics = ["f1", "precision", "recall"]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[n].get(metric, 0) for n in names]
        bars = ax1.bar(
            x + i * width, values, width, label=metric.capitalize(), color=color
        )
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("Detection Performance")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Fit time + Score time (log scale for readability)
    fit_times = [results[n].get("fit_time_s", 0) for n in names]
    score_times = [results[n].get("score_time_s", 0) for n in names]

    x2 = np.arange(len(names))
    ax2.bar(x2 - width / 2, fit_times, width, label="Fit time (s)", color="#0072B2")
    ax2.bar(x2 + width / 2, score_times, width, label="Score time (s)", color="#D55E00")

    for i, (ft, st) in enumerate(zip(fit_times, score_times)):
        ax2.text(
            i - width / 2,
            ft + max(fit_times) * 0.02,
            f"{ft:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax2.text(
            i + width / 2,
            st + max(fit_times) * 0.02,
            f"{st:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Latency Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Anomaly Detector Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
#  t-SNE / UMAP Projection
# =========================================================================


def plot_projection(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """2D projection of data with anomaly overlay.

    Args:
        X: (n_samples, n_features) data.
        y: Ground-truth labels.
        method: 'umap' (default) or 'tsne'.
        n_neighbors: UMAP n_neighbors (ignored for t-SNE).
        min_dist: UMAP min_dist (ignored for t-SNE).
        random_state: Random seed.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    # Subsample if too large (UMAP struggles above ~100k)
    if len(X) > 30000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), 30000, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                n_jobs=1,
            )
            embedding = reducer.fit_transform(X_sub)
        except ImportError:
            logger.warning("umap-learn not installed — falling back to t-SNE")
            method = "tsne"
        except Exception as e:
            logger.warning(f"UMAP failed ({e}) — falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        # Subsample further for t-SNE (it's O(n²))
        if len(X_sub) > 10000:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_sub), 10000, replace=False)
            X_sub = X_sub[idx]
            y_sub = y_sub[idx]
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        embedding = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=figsize)

    is_anom = y_sub == 1
    ax.scatter(
        embedding[~is_anom, 0],
        embedding[~is_anom, 1],
        c="#0072B2",
        alpha=0.4,
        s=5,
        label=f"Normal ({len(embedding[~is_anom])})",
        rasterized=True,
    )
    if is_anom.sum() > 0:
        ax.scatter(
            embedding[is_anom, 0],
            embedding[is_anom, 1],
            c="#D55E00",
            alpha=0.8,
            s=20,
            edgecolors="black",
            linewidth=0.3,
            label=f"Anomaly ({len(embedding[is_anom])})",
        )

    ax.set_title(
        f"2D Projection ({method.upper()}) — Ground Truth Labels", fontweight="bold"
    )
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
#  ROC Curves
# =========================================================================


def plot_roc_curves(
    all_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot ROC curves for all detectors.

    Args:
        all_scores: dict[name -> (n_samples,) scores]
        y_true: Ground-truth labels.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    for name, scores in all_scores.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        color = _get_color(name)
        ax.plot(
            fpr, tpr, color=color, linewidth=1.5, label=f"{name} (AUC={roc_auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
#  Convenience: generate all plots
# =========================================================================


def generate_all_plots(
    results: Dict[str, dict],
    all_scores: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_name: str = "dataset",
    save_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Run all visualizations and save to disk.

    Args:
        results: Per-detector metrics.
        all_scores: Per-detector anomaly scores.
        X_test: Test features.
        y_test: Test labels.
        data_name: Label for filenames.
        save_dir: Directory to save figures (default: FIGURES_DIR).

    Returns:
        dict: plot_name -> file_path
    """
    save_dir = save_dir or FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    plots = {
        "score_distributions": lambda: plot_score_distributions(all_scores, y_test),
        "comparison": lambda: plot_comparison_bar(results),
        "projection": lambda: plot_projection(X_test, y_test),
        "roc_curves": lambda: plot_roc_curves(all_scores, y_test),
    }

    for name, plot_fn in plots.items():
        try:
            fig = plot_fn()
            path = save_dir / f"{data_name}_{name}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved[name] = path
            logger.info(f"Saved {path}")
        except Exception as e:
            logger.warning(f"Failed to generate '{name}' plot: {e}")

    return saved
