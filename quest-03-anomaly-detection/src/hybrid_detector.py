"""
Hybrid XGBoost + Autoencoder detector.

Trains both models independently, then combines scores using an
OR gate — if either model flags a sample as anomalous, the hybrid
gives a high score.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from autoencoder import Autoencoder, train_autoencoder, compute_anomaly_scores
from xgboost_detector import XGBoostDetector

logger = logging.getLogger(__name__)


class HybridDetector:
    """Hybrid detector that OR-combines XGBoost and Autoencoder scores.

    During scoring, scores from both detectors are normalized to [0, 1]
    and the element-wise maximum is returned — so if *either* model
    flags a sample as anomalous, the hybrid score is high.
    """

    def __init__(
        self,
        xgb_kwargs: Optional[dict] = None,
        ae_kwargs: Optional[dict] = None,
        model_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.xgb_kwargs = xgb_kwargs or {}
        self.ae_kwargs = ae_kwargs or {}
        self.model_dir = model_dir
        self.verbose = verbose

        self.xgb_model: Optional[XGBoostDetector] = None
        self.ae_model: Optional[Autoencoder] = None
        self.ae_score_min: float = 0.0
        self.ae_score_max: float = 1.0
        self.input_dim: Optional[int] = None
        self.fit_time_: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HybridDetector":
        """Train both XGBoost and Autoencoder.

        XGBoost trains on all data with labels.
        Autoencoder trains on normal (y=0) samples only.
        Also records AE score statistics for normalization.

        Args:
            X: Training features.
            y: Binary labels (0=normal, 1=anomaly).
            X_val: Optional validation features.
            y_val: Optional validation labels.
        """
        import time

        start = time.perf_counter()
        self.input_dim = X.shape[1]

        # --- Train XGBoost ---
        if self.verbose:
            print("  [Hybrid] Training XGBoost...")
        self.xgb_model = XGBoostDetector(**self.xgb_kwargs)
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.xgb_model.fit(X, y, eval_set=eval_set)

        # --- Train Autoencoder on normal data only ---
        if self.verbose:
            print("  [Hybrid] Training Autoencoder...")
        X_normal = X[y == 0] if len(y) > 0 else X

        bottleneck_dim = self.ae_kwargs.pop("bottleneck_dim", 8)
        epochs = self.ae_kwargs.pop("epochs", 50)
        batch_size = self.ae_kwargs.pop("batch_size", 256)
        lr = self.ae_kwargs.pop("lr", 1e-3)

        self.ae_model, _ = train_autoencoder(
            X_normal,
            X_val=(
                X_val[y_val == 0] if X_val is not None and y_val is not None else None
            ),
            input_dim=self.input_dim,
            bottleneck_dim=bottleneck_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_dir=self.model_dir,
            verbose=self.verbose,
        )

        # --- Compute AE score stats on training normal data for normalization ---
        ae_scores_train = compute_anomaly_scores(self.ae_model, X_normal)
        self.ae_score_min = float(ae_scores_train.min())
        self.ae_score_max = float(ae_scores_train.max())
        if self.ae_score_max - self.ae_score_min < 1e-8:
            self.ae_score_max = self.ae_score_min + 1.0

        self.fit_time_ = time.perf_counter() - start
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score = max(normalized XGBoost score, normalized AE score).

        Normalizes the AE reconstruction error to [0, 1] using the
        min/max observed on the training normal data, then takes the
        element-wise max with the XGBoost probability.

        Args:
            X: Features to score.

        Returns:
            (n_samples,) array where higher = more anomalous (range ~[0, 1]).
        """
        if self.xgb_model is None or self.ae_model is None:
            raise RuntimeError("HybridDetector not fitted yet. Call fit() first.")

        # XGBoost score: probability of class 1 (already in [0, 1])
        xgb_scores = self.xgb_model.score_samples(X)

        # AE score: raw reconstruction error
        ae_scores_raw = compute_anomaly_scores(self.ae_model, X)

        # Normalize AE scores to [0, 1]
        ae_scores_norm = (ae_scores_raw - self.ae_score_min) / (
            self.ae_score_max - self.ae_score_min
        )
        ae_scores_norm = np.clip(ae_scores_norm, 0.0, 1.0)

        # OR gate: take the max of both normalized scores
        combined = np.maximum(xgb_scores, ae_scores_norm).astype(np.float32)
        return combined
