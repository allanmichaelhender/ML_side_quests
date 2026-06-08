"""
Data utilities for Quest 08 — Anomaly Detection.

Primary dataset: Credit Card Fraud (Kaggle).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_RESULTS = PROJECT / "results"

# ── Kaggle dataset info ────────────────────────────────────────────────────

CREDIT_CARD_KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
CREDIT_CARD_FILE = "creditcard.csv"
CREDIT_CARD_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
CREDIT_CARD_TARGET = "Class"

# ── Paths ────────────────────────────────────────────────────────────────────

KAGGLE_CACHE_DIR = DEFAULT_DATA / "kaggle"


# =========================================================================
#  Credit Card Fraud (Kaggle)
# =========================================================================


def download_credit_card_fraud(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download Credit Card Fraud dataset from Kaggle via kagglehub.

    Loads Kaggle credentials from the project .env file (KAGGLE_USERNAME,
    KAGGLE_KEY) before calling kagglehub.

    Returns the path to the downloaded CSV.
    """
    # Load Kaggle credentials from .env
    _load_kaggle_env()

    import kagglehub

    cache_dir = cache_dir or KAGGLE_CACHE_DIR
    csv_path = cache_dir / CREDIT_CARD_FILE

    if csv_path.exists() and not force_download:
        logger.info(f"Using cached dataset: {csv_path}")
        return csv_path

    logger.info(f"Downloading {CREDIT_CARD_KAGGLE_DATASET} from Kaggle...")
    path = kagglehub.dataset_download(CREDIT_CARD_KAGGLE_DATASET)
    # kagglehub downloads to a versioned cache path; copy to our cache dir
    source = Path(path) / CREDIT_CARD_FILE
    cache_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(source, csv_path)
    logger.info(f"Saved to {csv_path}")
    return csv_path


def _load_kaggle_env():
    """Load Kaggle credentials from .env into environment variables."""
    from dotenv import load_dotenv

    # Try project root .env first, then grandparent (repo root)
    for env_path in [
        PROJECT / ".env",
        PROJECT.parent / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded env from {env_path}")
            break

    # Ensure kagglehub can find credentials
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        logger.warning(
            "KAGGLE_USERNAME and/or KAGGLE_KEY not found in environment. "
            "kagglehub may fall back to ~/.kaggle/kaggle.json."
        )


def load_credit_card_fraud(
    path: Optional[Path] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Credit Card Fraud dataset.

    Args:
        path: Path to creditcard.csv. If None, attempts to find/download.
        sample_frac: If set, take a stratified random fraction (useful for
                     quick iterations on the full 284k dataset).
        random_state: Random seed for subsampling.

    Returns:
        X: (n_samples, 30) feature array
        y: (n_samples,) binary labels (0=normal, 1=fraud)
    """
    if path is None:
        path = KAGGLE_CACHE_DIR / CREDIT_CARD_FILE
        if not path.exists():
            path = download_credit_card_fraud()

    logger.info(f"Loading Credit Card Fraud from {path}")
    df = pd.read_csv(path)

    if sample_frac is not None:
        # Stratified subsample to preserve class balance
        normal = df[df[CREDIT_CARD_TARGET] == 0]
        fraud = df[df[CREDIT_CARD_TARGET] == 1]
        n_normal = int(len(normal) * sample_frac)
        n_fraud = int(len(fraud) * sample_frac)
        # Ensure at least 1 fraud sample
        n_fraud = max(n_fraud, 1)
        normal = normal.sample(n=n_normal, random_state=random_state)
        fraud = fraud.sample(n=n_fraud, random_state=random_state)
        df = pd.concat([normal, fraud], ignore_index=True)
        logger.info(f"Subsampled to {len(df)} rows ({n_fraud} fraud)")

    X = df[CREDIT_CARD_FEATURES].values.astype(np.float32)
    y = df[CREDIT_CARD_TARGET].values.astype(np.int64)

    n_fraud = int(y.sum())
    logger.info(
        f"Loaded {len(X)} samples, {n_fraud} fraud ({n_fraud / len(X) * 100:.3f}%)"
    )
    return X, y


# =========================================================================
#  Shared utilities
# =========================================================================


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split preserving anomaly ratio."""
    from sklearn.model_selection import train_test_split

    test_frac = 1.0 - train_frac - val_frac
    if test_frac < 0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - train_frac), random_state=random_state, stratify=y
    )

    relative_val = (
        val_frac / (val_frac + test_frac) if (val_frac + test_frac) > 0 else 0
    )
    if relative_val > 0 and relative_val < 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1.0 - relative_val),
            random_state=random_state,
            stratify=y_temp,
        )
    else:
        X_val, y_val = np.array([]), np.array([])
        X_test, y_test = X_temp, y_temp

    logger.info(
        f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test "
        f"(anomaly ratios: {y_train.mean():.3f}, {y_val.mean() if len(y_val) else 0:.3f}, "
        f"{y_test.mean():.3f})"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_metrics(metrics: dict, path: Optional[Path] = None) -> Path:
    """Save evaluation metrics to JSON."""
    path = path or (DEFAULT_RESULTS / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {path}")
    return path


def load_metrics(path: Optional[Path] = None) -> dict:
    """Load evaluation metrics from JSON."""
    path = path or (DEFAULT_RESULTS / "metrics.json")
    with open(path) as f:
        return json.load(f)
