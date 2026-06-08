"""
Full evaluation pipeline for Quest 08 — Anomaly Detection.

Runs all 5 detection methods, tunes thresholds, computes metrics,
generates visualizations, and saves results.

Usage:
    python src/evaluate.py                          # Kaggle Credit Card Fraud
    python src/evaluate.py --data synthetic         # synthetic data
    python src/evaluate.py --data nab --nab-dataset machine_temp  # NAB
    python src/evaluate.py --sample 0.1             # 10% subsample (quick test)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib

matplotlib.use("Agg")

from data_utils import (
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    load_credit_card_fraud,
    generate_synthetic_data,
    load_nab_dataset,
    train_val_test_split,
    save_metrics,
)
from detectors import DETECTOR_NAMES, fit_detector, score_detector
from threshold import find_best_threshold, evaluate_threshold

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "autoencoder_model"


def run_evaluation(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    data_name: str = "credit_card",
    scale: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run all detectors and collect metrics.

    Args:
        X_train, X_val, X_test: Data splits.
        y_train, y_val, y_test: Ground-truth labels.
        data_name: Label for the dataset (for metrics).
        scale: If True, fit StandardScaler on X_train and transform all splits.
        verbose: Print progress.

    Returns:
        metrics: Nested dict with per-detector results.
    """
    # Optional scaling — critical for Credit Card Fraud (Amount, Time unscaled)
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if len(X_val) > 0:
            X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        if verbose:
            print("  Applied StandardScaler to features")

    # Use only normal data for training autoencoder
    X_train_normal = X_train[y_train == 0] if len(y_train) > 0 else X_train

    results = {}
    all_scores: Dict[str, np.ndarray] = {}
    overall_start = time.time()

    for name in DETECTOR_NAMES:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")

        # Use normal-only data for autoencoder training
        fit_X = X_train_normal if name == "Autoencoder" else X_train

        # Fit
        model_dir = MODEL_DIR if name == "Autoencoder" else None
        model, fit_time = fit_detector(
            name, fit_X, model_dir=model_dir, verbose=verbose
        )

        # Score test set
        score_start = time.time()
        scores = score_detector(name, model, X_test)
        score_time = time.time() - score_start
        all_scores[name] = scores

        # Threshold optimization
        if len(np.unique(y_test)) >= 2:
            best_thresh, best_f1, best_metrics = find_best_threshold(y_test, scores)
        else:
            # Unsupervised: use 95th percentile
            from threshold import find_threshold_percentile, evaluate_threshold

            best_thresh = find_threshold_percentile(scores, percentile=95.0)
            best_metrics = evaluate_threshold(y_test, scores, best_thresh)
            if verbose:
                print(
                    f"    ⚠ Only one class in test set — using 95th percentile threshold"
                )

        results[name] = {
            "fit_time_s": round(fit_time, 3),
            "score_time_s": round(score_time, 4),
            "threshold": round(best_metrics["threshold"], 6),
            "precision": round(best_metrics["precision"], 4),
            "recall": round(best_metrics["recall"], 4),
            "f1": round(best_metrics["f1"], 4),
            "n_test": len(X_test),
            "n_anomalies_test": int(y_test.sum()),
            "anomaly_rate_test": round(float(y_test.mean()), 4),
        }

        if verbose:
            print(
                f"    F1: {best_metrics['f1']:.4f}  |  "
                f"Precision: {best_metrics['precision']:.4f}  |  "
                f"Recall: {best_metrics['recall']:.4f}"
            )
            print(
                f"    Threshold: {best_metrics['threshold']:.4f}  |  "
                f"Fit: {fit_time:.2f}s  |  Score: {score_time:.3f}s"
            )

    total_time = time.time() - overall_start

    # Compile summary
    summary = {
        "dataset": data_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "total_time_s": round(total_time, 2),
        "per_detector": results,
        # Best overall
        "best_f1": max((v["f1"], k) for k, v in results.items()),
    }

    # Generate visualizations
    try:
        _generate_plots(results, all_scores, X_test, y_test, data_name)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()

    return summary


def _generate_plots(
    results: Dict[str, dict],
    all_scores: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_name: str,
):
    """Generate evaluation plots."""
    from visualize import (
        plot_score_distributions,
        plot_comparison_bar,
        plot_projection,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Score distributions
    try:
        if all_scores:
            fig = plot_score_distributions(all_scores, y_test)
            fig.savefig(
                FIGURES_DIR / f"{data_name}_score_distributions.png",
                bbox_inches="tight",
            )
            print(f"  Saved score distributions plot")
    except Exception as e:
        logger.warning(f"Score distribution plot failed: {e}")

    # Comparison bar chart
    try:
        if results:
            fig = plot_comparison_bar(results)
            fig.savefig(
                FIGURES_DIR / f"{data_name}_comparison.png", bbox_inches="tight"
            )
            print(f"  Saved comparison bar chart")
    except Exception as e:
        logger.warning(f"Comparison bar chart failed: {e}")

    # t-SNE/UMAP projection
    try:
        fig = plot_projection(X_test, y_test)
        fig.savefig(FIGURES_DIR / f"{data_name}_projection.png", bbox_inches="tight")
        print(f"  Saved projection plot")
    except Exception as e:
        logger.warning(f"Projection plot failed: {e}")


# =========================================================================
#  CLI
# =========================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Evaluation Pipeline"
    )
    parser.add_argument(
        "--data",
        choices=["credit_card", "synthetic", "nab"],
        default="credit_card",
        help="Dataset to use (default: credit_card)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Subsample fraction for quick testing (e.g. 0.1 = 10%)",
    )
    parser.add_argument(
        "--nab-dataset",
        type=str,
        default="machine_temp",
        help=f"NAB dataset name (default: machine_temp). "
        f"Options: ec2_cpu, ec2_network, rds_cpu, machine_temp, nyc_taxi, ambient_temp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=5000,
        help="Number of synthetic samples (default: 5000)",
    )
    parser.add_argument(
        "--synthetic-features",
        type=int,
        default=8,
        help="Number of synthetic features (default: 8)",
    )
    parser.add_argument(
        "--synthetic-contamination",
        type=float,
        default=0.05,
        help="Anomaly ratio for synthetic data (default: 0.05)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler preprocessing (not recommended for credit_card)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 60)
    print("  Quest 08 — Anomaly Detection Evaluation")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.data}")

    if args.data == "credit_card":
        X, y = load_credit_card_fraud(sample_frac=args.sample)
        data_name = f"credit_card_{args.sample or 'full'}"

    elif args.data == "synthetic":
        X, y = generate_synthetic_data(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            contamination=args.synthetic_contamination,
            random_state=args.seed,
        )
        data_name = f"synthetic_{args.synthetic_samples}"

    elif args.data == "nab":
        X, y = load_nab_dataset(args.nab_dataset)
        data_name = f"nab_{args.nab_dataset}"

    else:
        raise ValueError(f"Unknown dataset: {args.data}")

    print(
        f"  Shape: {X.shape}  |  Anomalies: {y.sum()} / {len(y)} ({y.mean() * 100:.2f}%)"
    )

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_frac=0.6, val_frac=0.2, random_state=args.seed
    )
    if len(X_val) == 0:
        # Fallback for very small datasets
        X_train, X_test, y_train, y_test = train_val_test_split(
            X, y, train_frac=0.8, val_frac=0.0, random_state=args.seed
        )[:4]
        X_val, y_val = np.array([]), np.array([])

    # ── Run evaluation ──────────────────────────────────────────────────────
    metrics = run_evaluation(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        data_name=data_name,
        scale=not args.no_scale,
        verbose=True,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    save_metrics(metrics)

    print(f"\n{'=' * 60}")
    print(f"  Results Summary")
    print(f"{'=' * 60}")
    print(f"  Dataset: {data_name}")
    print(f"  Best F1: {metrics['best_f1'][0]:.4f} ({metrics['best_f1'][1]})")
    print(f"\n  Per-detector comparison:")
    for name in DETECTOR_NAMES:
        m = metrics["per_detector"][name]
        print(
            f"    {name:20s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  Fit={m['fit_time_s']:.2f}s  "
            f"Score={m['score_time_s']:.3f}s"
        )

    print(f"\n  Total time: {metrics['total_time_s']:.2f}s")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
