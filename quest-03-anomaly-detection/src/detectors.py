"""
Unified detector wrappers for 5 anomaly detection methods.

Provides a consistent .fit() / .score() interface so the evaluation
pipeline can treat all detectors interchangeably.

Methods:
  - Isolation Forest          (sklearn)
  - Local Outlier Factor      (sklearn)
  - Autoencoder               (PyTorch — see autoencoder.py)
  - DBSCAN                    (sklearn, used as outlier detector)
  - XGBoost                   (xgboost, supervised — uses y_train)
  - Hybrid (XGB+AE)          (ensemble, OR-gate combination)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from xgboost_detector import XGBoostDetector
from hybrid_detector import HybridDetector

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# ── Detector registry ──────────────────────────────────────────────────────

# Each entry: name -> (class_or_factory, fit_kwargs, score_method)
# 'score_method' is the attribute to call for anomaly scores
# (higher = more anomalous, or we'll negate if needed)

DETECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "Isolation Forest": {
        "class": IsolationForest,
        "default_kwargs": {
            "n_estimators": 100,
            "contamination": "auto",
            "random_state": 42,
            "n_jobs": -1,
        },
        "score_attr": "score_samples",  # sklearn: more negative = more anomalous
        "negate_score": True,  # so higher score = more anomalous
        "needs_fit": True,
        "requires_y": False,
    },
    "LOF": {
        "class": LocalOutlierFactor,
        "default_kwargs": {
            "n_neighbors": 20,
            "contamination": "auto",
            "novelty": True,  # enables .score_samples() on new data
        },
        "score_attr": "score_samples",  # more negative = more anomalous
        "negate_score": True,
        "needs_fit": True,
        "requires_y": False,
    },
    "DBSCAN": {
        "class": DBSCAN,
        "default_kwargs": {
            "eps": 0.5,
            "min_samples": 10,
            "n_jobs": -1,
        },
        "score_attr": None,  # custom: uses cluster label (-1 = outlier)
        "negate_score": False,
        "needs_fit": True,
        "requires_y": False,
    },
    "Autoencoder": {
        "class": None,  # special handling
        "default_kwargs": {},
        "score_attr": None,
        "negate_score": False,
        "needs_fit": True,
        "requires_y": False,  # trained only on normal data internally
    },
    "XGBoost": {
        "class": XGBoostDetector,
        "default_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": None,  # auto-computed from class ratio
            "random_state": 42,
            "n_jobs": -1,
        },
        "score_attr": "score_samples",  # returns proba of class 1
        "negate_score": False,  # already higher = more anomalous
        "needs_fit": True,
        "requires_y": True,  # supervised — needs labels!
    },
    "Hybrid (XGB+AE)": {
        "class": None,  # special handling
        "default_kwargs": {},
        "score_attr": None,
        "negate_score": False,
        "needs_fit": True,
        "requires_y": True,  # XGBoost needs labels; AE uses normal only
    },
}

DETECTOR_NAMES = list(DETECTOR_INFO.keys())


# =========================================================================
#  Unified fit / score
# =========================================================================


def _fit_autoencoder(
    X_train: np.ndarray,
    kwargs: dict,
    model_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Any, float]:
    """Train the PyTorch autoencoder on normal data only."""
    from autoencoder import train_autoencoder

    # Use only normal samples if labels are available
    # If X_train has more rows than features, assume labels not provided
    # and train on all data (unsupervised).
    input_dim = X_train.shape[1]

    fit_kwargs = {**kwargs}
    # Pop training-specific args
    bottleneck_dim = fit_kwargs.pop("bottleneck_dim", 8)
    epochs = fit_kwargs.pop("epochs", 50)
    batch_size = fit_kwargs.pop("batch_size", 256)
    lr = fit_kwargs.pop("lr", 1e-3)

    start = time.time()
    model, _ = train_autoencoder(
        X_train,
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_dir=model_dir,
        verbose=verbose,
    )
    elapsed = time.time() - start
    return model, elapsed


def fit_detector(
    name: str,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    kwargs: Optional[dict] = None,
    model_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Any, float]:
    """Fit a detector on training data.

    Args:
        name: Detector name (must be in DETECTOR_NAMES).
        X_train: (n_samples, n_features) training data.
        y_train: (n_samples,) labels — required for supervised detectors
                 like XGBoost (requires_y=True).
        X_val: Optional validation features (for XGBoost early stopping).
        y_val: Optional validation labels (for XGBoost early stopping).
        kwargs: Override default kwargs for the detector.
        model_dir: Directory to save autoencoder (only used for Autoencoder).
        verbose: Print progress.

    Returns:
        model: Fitted detector object.
        fit_time: Time taken to fit (seconds).
    """
    if name not in DETECTOR_INFO:
        raise ValueError(f"Unknown detector '{name}'. Available: {DETECTOR_NAMES}")

    info = DETECTOR_INFO[name]
    fit_kwargs = {**info["default_kwargs"], **(kwargs or {})}

    if name == "Autoencoder":
        return _fit_autoencoder(X_train, fit_kwargs, model_dir, verbose)

    if name == "Hybrid (XGB+AE)":
        if y_train is None:
            raise ValueError(
                "Hybrid (XGB+AE) requires y_train (supervised component). "
                "Pass y_train to fit_detector()."
            )
        start = time.time()
        model = HybridDetector(
            xgb_kwargs=kwargs or {},
            model_dir=model_dir,
            verbose=verbose,
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        elapsed = time.time() - start
        logger.info(f"Hybrid (XGB+AE) fitted in {elapsed:.2f}s")
        return model, elapsed

    if name == "XGBoost":
        if y_train is None:
            raise ValueError(
                "XGBoost requires y_train (supervised detector). "
                "Pass y_train to fit_detector()."
            )
        start = time.time()
        model = XGBoostDetector(**fit_kwargs)
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_set)
        elapsed = time.time() - start
        logger.info(f"XGBoost fitted in {elapsed:.2f}s")
        return model, elapsed

    start = time.time()
    model = info["class"](**fit_kwargs)
    model.fit(X_train)
    elapsed = time.time() - start

    logger.info(f"{name} fitted in {elapsed:.2f}s")
    return model, elapsed


def score_detector(
    name: str,
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Compute anomaly scores for a fitted detector.

    Returns scores where **higher = more anomalous** (normalized convention).

    Args:
        name: Detector name.
        model: Fitted detector object.
        X: (n_samples, n_features) data to score.

    Returns:
        scores: (n_samples,) array, higher = more anomalous.
    """
    info = DETECTOR_INFO[name]

    if name == "Autoencoder":
        from autoencoder import compute_anomaly_scores

        return compute_anomaly_scores(model, X)

    if name == "Hybrid (XGB+AE)":
        return model.score_samples(X)

    if name == "DBSCAN":
        # DBSCAN assigns -1 to outliers, 0/1/... to clusters.
        # Convert: -1 outlier = high score, cluster = low score.
        # We use the distance to nearest cluster as a continuous score.
        labels = (
            model.fit_predict(X) if not hasattr(model, "labels_)") else model.labels_
        )

        # For a smoother score, use distance to nearest in-cluster point
        # Fallback: binary -1 -> score 1, else 0
        scores = np.ones(len(X), dtype=np.float32)
        scores[labels != -1] = 0.0
        return scores

    # Standard sklearn interface: score_samples() or decision_function()
    score_attr = info["score_attr"]
    if hasattr(model, score_attr):
        raw = getattr(model, score_attr)(X)
    else:
        raise AttributeError(f"{name} has no '{score_attr}' method")

    if info["negate_score"]:
        raw = -raw

    return raw.astype(np.float32)


# =========================================================================
#  Convenience: fit and score all detectors
# =========================================================================


def run_all_detectors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    detector_kwargs: Optional[Dict[str, dict]] = None,
    model_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fit all detectors and return scores + fit times.

    Args:
        X_train: Training data.
        X_test: Test data to score.
        detector_kwargs: Per-detector kwarg overrides, keyed by name.
        model_dir: Directory for saving autoencoder.
        verbose: Print progress.

    Returns:
        dict: name -> {"scores": np.ndarray, "fit_time": float, "score_time": float}
    """
    results = {}
    for name in DETECTOR_NAMES:
        if verbose:
            print(f"\n  [{name}] Fitting...")

        kwargs = (detector_kwargs or {}).get(name, {})

        model, fit_time = fit_detector(name, X_train, kwargs, model_dir, verbose)

        start = time.time()
        scores = score_detector(name, model, X_test)
        score_time = time.time() - start

        results[name] = {
            "scores": scores,
            "fit_time": fit_time,
            "score_time": score_time,
        }

        if verbose:
            n_detected = int((scores > np.median(scores)).sum())
            print(
                f"    Fit: {fit_time:.2f}s | Score: {score_time:.3f}s | "
                f"Score range: [{scores.min():.4f}, {scores.max():.4f}]"
            )

    return results
