"""
XGBoost wrapper for anomaly detection.

Uses XGBoost as a supervised binary classifier (fraud vs normal),
then treats the predicted probability of the "anomaly" class as
the anomaly score.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Type alias to allow both raw arrays and dataframes with category types
DataType = Union[np.ndarray, pd.DataFrame]


class XGBoostDetector:
    """Wrapper around XGBoost that exposes a sklearn-like fit/score interface.

    Trains a binary classifier on (X_train, y_train) and uses the
    probability of class 1 as the anomaly score.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = 0,
        early_stopping_rounds: int = 15,
        enable_categorical: bool = True,
        **kwargs,
    ):
        if xgb is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_categorical = enable_categorical
        self.kwargs = kwargs

        self._model = None
        self.fit_time_: Optional[float] = None

    def fit(
        self,
        X: DataType,
        y: np.ndarray,
        eval_set: Optional[Tuple[DataType, np.ndarray]] = None,
    ) -> "XGBoostDetector":
        """Train XGBoost on labelled data.

        Args:
            X: Training features (numpy array or pandas DataFrame).
            y: Binary labels (0=normal, 1=anomaly).
            eval_set: Optional validation (X_val, y_val) tuple for early
                      stopping. If None, falls back to using the training
                      data itself.

        Returns:
            self
        """
        start_time = time.perf_counter()

        # Auto-compute scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            n_normal = int((y == 0).sum())
            n_anom = int((y == 1).sum())
            self.scale_pos_weight = n_normal / max(n_anom, 1)

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            early_stopping_rounds=self.early_stopping_rounds,
            enable_categorical=self.enable_categorical,
            eval_metric="aucpr",
            **self.kwargs,
        )

        # Early stopping requires an evaluation set
        if eval_set is None:
            eval_set_list = [(X, y)]
        else:
            eval_set_list = [eval_set]

        self._model.fit(X, y, eval_set=eval_set_list, verbose=False)

        self.fit_time_ = time.perf_counter() - start_time
        return self

    def save(self, path: Path) -> "XGBoostDetector":
        """Save the trained XGBoost model to disk.

        Args:
            path: Directory path to save the model files.
        """
        if self._model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "model.json"))
        # Save init params so we can reconstruct the wrapper
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "early_stopping_rounds": self.early_stopping_rounds,
            "enable_categorical": self.enable_categorical,
        }
        with open(path / "params.json", "w") as f:
            json.dump(params, f)
        logger.info(f"XGBoost model saved to {path}")
        return self

    @classmethod
    def load(cls, path: Path) -> "XGBoostDetector":
        """Load a trained XGBoost model from disk.

        Args:
            path: Directory path containing the saved model files.

        Returns:
            Loaded XGBoostDetector instance.
        """
        with open(path / "params.json") as f:
            params = json.load(f)
        detector = cls(**params)
        detector._model = xgb.XGBClassifier()
        detector._model.load_model(str(path / "model.json"))
        logger.info(f"XGBoost model loaded from {path}")
        return detector

    def predict_proba(self, X: DataType) -> np.ndarray:
        """Get predicted probabilities.

        Args:
            X: Features to predict.

        Returns:
            Array of shape (n_samples, 2): [prob_normal, prob_anomaly]
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self._model.predict_proba(X)

    def score_samples(self, X: DataType) -> np.ndarray:
        """Anomaly scores — probability of the anomaly class.

        Args:
            X: Features to score.

        Returns:
            (n_samples,) array where higher = more anomalous.
        """
        proba = self.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def predict(self, X: DataType, threshold: float = 0.5) -> np.ndarray:
        """Convert probabilities to hard binary decisions.

        Args:
            X: Features to predict.
            threshold: Probability threshold to flag as fraud.

        Returns:
            (n_samples,) binary array (0 or 1).
        """
        scores = self.score_samples(X)
        return (scores >= threshold).astype(np.int32)
