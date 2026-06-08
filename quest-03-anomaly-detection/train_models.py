"""
One-time training script for XGBoost and Autoencoder.

Saves trained models to disk so evaluation runs can load them
instead of retraining every time.

Usage:
    python train_models.py                          # full dataset
    python train_models.py --sample 0.1             # 10% subsample (quick test)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

from src.data_utils import (
    load_credit_card_fraud,
    train_val_test_split,
)
from src.detectors import fit_detector
from src.xgboost_detector import XGBoostDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RESULTS_DIR = _HERE / "results"
MODEL_DIR = RESULTS_DIR / "autoencoder_model"
XGB_MODEL_DIR = RESULTS_DIR / "model"


def main():
    parser = argparse.ArgumentParser(
        description="Train and save XGBoost + Autoencoder models"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Subsample fraction (e.g. 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Training XGBoost + Autoencoder")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print(
        f"\nLoading Credit Card Fraud dataset"
        + (f" ({args.sample * 100:.0f}% subsample)" if args.sample else "")
    )
    X, y = load_credit_card_fraud(sample_frac=args.sample)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_frac=0.6, val_frac=0.2, random_state=args.seed
    )

    # Scale features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(
        f"  Anomaly ratio: train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}"
    )

    # ── Train XGBoost ─────────────────────────────────────────────────────
    print("\n--- Training XGBoost ---")
    t0 = time.perf_counter()
    xgb_model = XGBoostDetector(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=args.seed,
        n_jobs=-1,
    )
    eval_set = (X_val, y_val) if len(X_val) > 0 else None
    xgb_model.fit(X_train, y_train, eval_set=eval_set)
    print(f"  Fitted in {xgb_model.fit_time_:.2f}s")
    xgb_model.save(XGB_MODEL_DIR)
    print(f"  Saved to {XGB_MODEL_DIR}")

    # Quick validation
    xgb_scores = xgb_model.score_samples(X_test)
    from sklearn.metrics import roc_auc_score

    print(f"  Test ROC-AUC: {roc_auc_score(y_test, xgb_scores):.4f}")

    # ── Train Autoencoder ──────────────────────────────────────────────────
    print("\n--- Training Autoencoder ---")
    X_train_normal = X_train[y_train == 0] if len(y_train) > 0 else X_train
    X_val_normal = X_val[y_val == 0] if len(X_val) > 0 and len(y_val) > 0 else None
    t0 = time.perf_counter()
    from src.autoencoder import train_autoencoder, compute_anomaly_scores

    ae_model, _ = train_autoencoder(
        X_train_normal,
        X_val=X_val_normal,
        input_dim=X_train.shape[1],
        bottleneck_dim=8,
        epochs=50,
        batch_size=256,
        lr=1e-3,
        model_dir=MODEL_DIR,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fitted in {elapsed:.1f}s")
    print(f"  Saved to {MODEL_DIR}")

    # Quick validation
    ae_scores = compute_anomaly_scores(ae_model, X_test)
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, ae_scores):.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  XGBoost:   {XGB_MODEL_DIR}")
    print(f"  Autoencoder: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
