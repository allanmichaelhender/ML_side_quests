"""
Full evaluation pipeline for Quest 08 — Anomaly Detection.

Loads pre-trained XGBoost and Autoencoder from disk, trains IF and DBSCAN
on the fly (fast, no labels needed), and evaluates Hybrid as a true binary
OR gate.

Usage:
    python src/evaluate.py                          # Kaggle Credit Card Fraud
    python src/evaluate.py --sample 0.1             # 10% subsample
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
    train_val_test_split,
    save_metrics,
)
from detectors import DETECTOR_NAMES, fit_detector, score_detector
from threshold import find_best_threshold, apply_threshold, evaluate_threshold

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
AE_MODEL_DIR = RESULTS_DIR / "autoencoder_model"
XGB_MODEL_DIR = RESULTS_DIR / "model"

# Detectors that retrain every run (fast, unsupervised)
RETRAIN_DETECTORS = {"Isolation Forest", "LOF", "DBSCAN"}


def _load_or_skip_saved(name: str, X_train, y_train, X_val, y_val, verbose):
    """Load pre-trained model or return None if not available."""
    from xgboost_detector import XGBoostDetector
    from autoencoder import Autoencoder

    if name == "XGBoost":
        if not (XGB_MODEL_DIR / "model.json").exists():
            if verbose:
                print(f"    No saved model found — training XGBoost...")
            return None
        model = XGBoostDetector.load(XGB_MODEL_DIR)
        if verbose:
            print(f"    Loaded saved model from {XGB_MODEL_DIR}")
        return model

    if name == "Autoencoder":
        model_path = AE_MODEL_DIR / "autoencoder.pt"
        if not model_path.exists():
            if verbose:
                print(f"    No saved model found — training Autoencoder...")
            return None
        import torch

        model = Autoencoder(input_dim=X_train.shape[1])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        if verbose:
            print(f"    Loaded saved model from {model_path}")
        return model

    return None


def _evaluate_hybrid_or(
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
    xgb_threshold: Optional[float] = None,
) -> Tuple[dict, np.ndarray]:
    """True binary OR gate: score both models, find individual thresholds,
    OR the binary predictions, compute metrics.

    Args:
        xgb_threshold: Optional manual threshold for XGBoost.
                       If None, uses F1-optimized threshold.

    Returns:
        (metrics_dict, combined_scores_for_plotting)
    """
    from xgboost_detector import XGBoostDetector
    from autoencoder import compute_anomaly_scores, Autoencoder
    import torch

    start = time.time()

    # Load both models
    xgb_model = XGBoostDetector.load(XGB_MODEL_DIR)
    ae_model = Autoencoder(input_dim=X_test.shape[1])
    ae_model.load_state_dict(
        torch.load(
            AE_MODEL_DIR / "autoencoder.pt", map_location="cpu", weights_only=True
        )
    )

    # Score with both
    xgb_scores = xgb_model.score_samples(X_test)
    ae_scores = compute_anomaly_scores(ae_model, X_test)

    # Find individual optimal thresholds
    if xgb_threshold is not None:
        xgb_m = evaluate_threshold(y_test, xgb_scores, xgb_threshold)
    else:
        _, _, xgb_m = find_best_threshold(y_test, xgb_scores)
    _, _, ae_m = find_best_threshold(y_test, ae_scores)

    # Apply thresholds → binary → OR gate
    xgb_preds = apply_threshold(xgb_scores, xgb_m["threshold"])
    ae_preds = apply_threshold(ae_scores, ae_m["threshold"])
    hybrid_preds = np.maximum(xgb_preds, ae_preds)  # element-wise OR

    # Compute combined metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    hybrid_metrics = {
        "threshold": max(xgb_m["threshold"], ae_m["threshold"]),
        "precision": round(
            float(precision_score(y_test, hybrid_preds, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, hybrid_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, hybrid_preds, zero_division=0)), 4),
    }

    # Combined score for visualization (max of both raw scores)
    combined_scores = np.maximum(xgb_scores, ae_scores).astype(np.float32)

    elapsed = time.time() - start

    if verbose:
        print(
            f"    XGBoost threshold: {xgb_m['threshold']:.4f}  (F1={xgb_m['f1']:.4f})"
        )
        print(
            f"    Autoencoder threshold: {ae_m['threshold']:.4f}  (F1={ae_m['f1']:.4f})"
        )
        print(
            f"    Hybrid OR — F1: {hybrid_metrics['f1']:.4f}  |  "
            f"Precision: {hybrid_metrics['precision']:.4f}  |  "
            f"Recall: {hybrid_metrics['recall']:.4f}"
        )
        print(f"    Time: {elapsed:.2f}s")

    result = {
        "fit_time_s": 0.0,
        "score_time_s": round(elapsed, 4),
        "threshold": hybrid_metrics["threshold"],
        "precision": hybrid_metrics["precision"],
        "recall": hybrid_metrics["recall"],
        "f1": hybrid_metrics["f1"],
        "n_test": len(X_test),
        "n_anomalies_test": int(y_test.sum()),
        "anomaly_rate_test": round(float(y_test.mean()), 4),
    }

    return result, combined_scores


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
    xgb_threshold: Optional[float] = None,
) -> Dict[str, dict]:
    """Run all detectors and collect metrics.

    - IF, LOF, DBSCAN: retrained every run (fast, unsupervised).
    - XGBoost, Autoencoder: loaded from saved models (must run train_models.py first).
    - Hybrid (XGB+AE): true binary OR of both detectors' predictions.

    Args:
        X_train, X_val, X_test: Data splits.
        y_train, y_val, y_test: Ground-truth labels.
        data_name: Label for the dataset.
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

    X_train_normal = X_train[y_train == 0] if len(y_train) > 0 else X_train

    results = {}
    all_scores: Dict[str, np.ndarray] = {}
    overall_start = time.time()

    for name in DETECTOR_NAMES:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")

        # ── Hybrid: special true OR gate ────────────────────────────────────
        if name == "Hybrid (XGB+AE)":
            result, combined_scores = _evaluate_hybrid_or(
                X_test, y_test, verbose=verbose, xgb_threshold=xgb_threshold
            )
            results[name] = result
            all_scores[name] = combined_scores
            continue

        # ── XGBoost / Autoencoder: load saved model ────────────────────────
        loaded_model = _load_or_skip_saved(
            name, X_train, y_train, X_val, y_val, verbose
        )

        if loaded_model is not None:
            model = loaded_model
            fit_time = 0.0
        else:
            # ── Retrain from scratch ────────────────────────────────────────
            fit_X = X_train_normal if name == "Autoencoder" else X_train
            needs_supervision = name == "XGBoost"
            fit_y = y_train if needs_supervision else None
            fit_X_val = X_val if needs_supervision else None
            fit_y_val = y_val if needs_supervision else None
            model_dir = AE_MODEL_DIR if name == "Autoencoder" else None

            model, fit_time = fit_detector(
                name,
                fit_X,
                y_train=fit_y,
                X_val=fit_X_val,
                y_val=fit_y_val,
                model_dir=model_dir,
                verbose=verbose,
            )

        # ── Score test set ──────────────────────────────────────────────────
        score_start = time.time()
        scores = score_detector(name, model, X_test)
        score_time = time.time() - score_start
        all_scores[name] = scores

        # ── Threshold optimization ──────────────────────────────────────────
        if name == "XGBoost" and xgb_threshold is not None:
            best_metrics = evaluate_threshold(y_test, scores, xgb_threshold)
            if verbose:
                print(f"    ⚡ Using manual threshold: {xgb_threshold:.4f}")
        elif len(np.unique(y_test)) >= 2:
            best_thresh, best_f1, best_metrics = find_best_threshold(y_test, scores)
        else:
            from threshold import find_threshold_percentile

            best_thresh = find_threshold_percentile(scores, percentile=95.0)
            best_metrics = evaluate_threshold(y_test, scores, best_thresh)
            if verbose:
                print(f"    ⚠ Only one class — using 95th percentile threshold")

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

    summary = {
        "dataset": data_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "total_time_s": round(total_time, 2),
        "per_detector": results,
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
        "--sample",
        type=float,
        default=None,
        help="Subsample fraction for quick testing (e.g. 0.1 = 10%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler preprocessing (not recommended for credit_card)",
    )
    parser.add_argument(
        "--xgb-threshold",
        type=float,
        default=None,
        help="Override XGBoost threshold (default: F1-optimized)",
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
    print(f"\nLoading Credit Card Fraud dataset")

    X, y = load_credit_card_fraud(sample_frac=args.sample)
    data_name = f"credit_card_{args.sample or 'full'}"

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
        xgb_threshold=args.xgb_threshold,
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
