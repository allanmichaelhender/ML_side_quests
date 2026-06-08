"""
Automated threshold optimization for anomaly detection.

Given anomaly scores and ground-truth labels, finds the threshold
that maximizes F1 score. Also supports unsupervised percentile-based
thresholding when labels are unavailable.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

logger = logging.getLogger(__name__)


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 1000,
) -> Tuple[float, float, Dict[str, float]]:
    """Find the score threshold that maximizes F1 score.

    Sweeps percentile thresholds across the score distribution,
    computes F1 at each point, and returns the best.

    Args:
        y_true: Ground-truth binary labels (0=normal, 1=anomaly).
        scores: Anomaly scores (higher = more anomalous).
        n_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        best_threshold: The score threshold giving max F1.
        best_f1: The best F1 score achieved.
        best_metrics: Dict with 'precision', 'recall', 'f1', 'threshold'.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — using median threshold")
        med = float(np.median(scores))
        pred = (scores > med).astype(int)
        f1 = f1_score(y_true, pred) if len(np.unique(y_true)) == 2 else 0.0
        return med, f1, {"precision": 0.0, "recall": 0.0, "f1": f1, "threshold": med}

    # Use precision-recall curve to evaluate all thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    # F1 at each threshold
    # Note: precision_recall_curve returns n+1 thresholds, last prec/recall are
    # for "all positive" (threshold = -inf). We skip that for thresholding.
    f1_scores = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        if precisions[i] + recalls[i] > 0:
            f1_scores[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_metrics = {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
    }

    return best_threshold, best_f1, best_metrics


def find_threshold_percentile(
    scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Unsupervised threshold: use a percentile of the score distribution.

    Useful when labels are not available. Assumes the top `percentile`
    of scores are anomalies.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        percentile: Percentile threshold (e.g. 95 = top 5% are anomalies).

    Returns:
        threshold: Score value at the given percentile.
    """
    return float(np.percentile(scores, percentile))


def apply_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert anomaly scores to binary predictions.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        threshold: Cutoff value. Scores > threshold are anomalies.

    Returns:
        predictions: Binary array (1 = anomaly, 0 = normal).
    """
    return (scores > threshold).astype(np.int64)


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate a specific threshold and return metrics.

    Args:
        y_true: Ground-truth labels.
        scores: Anomaly scores.
        threshold: Score cutoff.

    Returns:
        metrics: Dict with 'precision', 'recall', 'f1', 'threshold'.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    preds = apply_threshold(scores, threshold)

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
