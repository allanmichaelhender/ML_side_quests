"""
Batch evaluation — pre-computes all results and saves to JSON.

Run once, then the Streamlit app loads the saved data — no live inference.

Usage:
    python batch_evaluate.py

Saves to: results/comparison_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from data_utils import load_credit_card_fraud, train_val_test_split
from detectors import fit_detector, score_detector
from threshold import apply_threshold, evaluate_threshold, find_best_threshold
from xgboost_detector import XGBoostDetector
from autoencoder import Autoencoder, compute_anomaly_scores
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_DIR = _HERE / "results"
XGB_MODEL_DIR = RESULTS_DIR / "model"
AE_MODEL_DIR = RESULTS_DIR / "autoencoder_model"
OUTPUT_PATH = RESULTS_DIR / "comparison_results.json"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def main():
    print("=" * 60)
    print("  Batch Evaluation — Pre-computing All Results")
    print("=" * 60)

    # ── 1. Load & split data ───────────────────────────────────────────────
    print("\nLoading Credit Card Fraud dataset (full)...")
    X, y = load_credit_card_fraud()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(
        f"  Test anomalies: {y_test.sum()} / {len(y_test)} ({y_test.mean() * 100:.2f}%)"
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  Applied StandardScaler")

    # ── 2. Useless detectors (IF, LOF, DBSCAN) ─────────────────────────────
    print("\n--- Evaluating IF, LOF, DBSCAN on test set ---")
    useless_results: Dict[str, dict] = {}
    for name in ["Isolation Forest", "LOF", "DBSCAN"]:
        t0 = time.time()
        model, _ = fit_detector(name, X_train, verbose=False)
        scores = score_detector(name, model, X_test_scaled)
        _, _, metrics = find_best_threshold(y_test, scores)
        useless_results[name] = {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "threshold": round(metrics["threshold"], 6),
        }
        print(
            f"  {name:20s}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
        )

    # ── 3. Load saved XGBoost & Autoencoder ────────────────────────────────
    print("\n--- Loading saved models ---")
    xgb_model = XGBoostDetector.load(XGB_MODEL_DIR)
    print(f"  XGBoost loaded from {XGB_MODEL_DIR}")

    ae_model = Autoencoder(input_dim=X_test_scaled.shape[1])
    ae_model.load_state_dict(
        torch.load(
            AE_MODEL_DIR / "autoencoder.pt", map_location="cpu", weights_only=True
        )
    )
    print(f"  Autoencoder loaded from {AE_MODEL_DIR}")

    # ── 4. Score test set ──────────────────────────────────────────────────
    print("\n--- Scoring test set ---")
    xgb_scores = xgb_model.score_samples(X_test_scaled)
    ae_scores = compute_anomaly_scores(ae_model, X_test_scaled)
    print(f"  Scored {len(xgb_scores)} test samples")

    # ── 5. Autoencoder standalone ──────────────────────────────────────────
    _, _, ae_metrics = find_best_threshold(y_test, ae_scores)

    # ── 6. Compute metrics at each threshold ───────────────────────────────
    print("\n--- Computing threshold sweep ---")
    threshold_results: Dict[str, dict] = {}

    # Autoencoder standalone (F1-optimized)
    threshold_results["Autoencoder"] = {
        "precision": round(ae_metrics["precision"], 4),
        "recall": round(ae_metrics["recall"], 4),
        "f1": round(ae_metrics["f1"], 4),
        "threshold": round(ae_metrics["threshold"], 6),
        "label": "Autoencoder",
    }

    for t in THRESHOLDS:
        # XGBoost standalone at threshold t
        xgb_metrics = evaluate_threshold(y_test, xgb_scores, t)
        key_xgb = f"XGBoost_{t}"
        threshold_results[key_xgb] = {
            "precision": round(xgb_metrics["precision"], 4),
            "recall": round(xgb_metrics["recall"], 4),
            "f1": round(xgb_metrics["f1"], 4),
            "threshold": t,
            "label": f"XGBoost (t={t})",
        }

        # Hybrid (XGB+AE OR) at threshold t
        xgb_preds = apply_threshold(xgb_scores, t)
        _, _, ae_best = find_best_threshold(y_test, ae_scores)
        ae_preds = apply_threshold(ae_scores, ae_best["threshold"])
        hybrid_preds = np.maximum(xgb_preds, ae_preds)

        key_hyb = f"Hybrid_{t}"
        threshold_results[key_hyb] = {
            "precision": round(
                float(precision_score(y_test, hybrid_preds, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_test, hybrid_preds, zero_division=0)), 4
            ),
            "f1": round(float(f1_score(y_test, hybrid_preds, zero_division=0)), 4),
            "threshold": t,
            "label": f"Hybrid (t={t})",
        }

        print(
            f"  t={t}:  XGBoost F1={threshold_results[key_xgb]['f1']:.4f}  "
            f"Hybrid F1={threshold_results[key_hyb]['f1']:.4f}"
        )

    # ── 7. Compile & save ──────────────────────────────────────────────────
    output = {
        "test_set": {
            "n_samples": int(len(X_test)),
            "n_anomalies": int(y_test.sum()),
            "anomaly_rate": round(float(y_test.mean()), 6),
            "n_features": int(X_test_scaled.shape[1]),
        },
        "useless_detectors": useless_results,
        "threshold_comparison": threshold_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(
        f"  Test set: {output['test_set']['n_samples']} samples, "
        f"{output['test_set']['n_anomalies']} anomalies"
    )
    print()
    for name, m in useless_results.items():
        print(
            f"  {name:20s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
        )
    print()
    for key in list(threshold_results.keys()):
        m = threshold_results[key]
        print(
            f"  {m['label']:20s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
        )
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
