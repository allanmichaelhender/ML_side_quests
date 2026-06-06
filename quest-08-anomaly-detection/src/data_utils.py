"""
Data utilities for Quest 08 — Anomaly Detection.

Primary dataset: Credit Card Fraud (Kaggle).
Fallback: Synthetic multivariate normal with injected outliers.
Also provides hooks for Numenta Anomaly Benchmark (NAB) time series data.
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
SYNTHETIC_FILE = DEFAULT_DATA / "synthetic_data.npz"


# =========================================================================
#  Credit Card Fraud (Kaggle)
# =========================================================================


def download_credit_card_fraud(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download Credit Card Fraud dataset from Kaggle via kagglehub.

    Returns the path to the downloaded CSV.
    """
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
#  Synthetic Data (fallback / quick-test)
# =========================================================================


def generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 8,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multivariate normal data with injected outliers.

    Outliers are placed farther from the mean in random directions.

    Args:
        n_samples: Total number of samples.
        n_features: Dimensionality.
        contamination: Fraction of samples that are outliers.
        random_state: Random seed.

    Returns:
        X: (n_samples, n_features) feature array
        y: (n_samples,) binary labels (0=normal, 1=anomaly)
    """
    rng = np.random.default_rng(random_state)
    n_normal = n_samples - int(n_samples * contamination)
    n_anom = n_samples - n_normal

    # Normal data: multivariate normal with random covariance
    mean = rng.uniform(-1, 1, size=n_features)
    cov = rng.uniform(0.5, 2.0, size=(n_features, n_features))
    cov = cov @ cov.T + np.eye(n_features) * 0.1  # make PSD

    X_normal = rng.multivariate_normal(mean, cov, size=n_normal)

    # Anomalies: shifted from mean by 3-5 sigma in random directions
    directions = rng.standard_normal((n_anom, n_features))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(3.0, 6.0, size=(n_anom, 1))
    X_anom = mean + directions * magnitudes * np.sqrt(np.diag(cov))

    X = np.vstack([X_normal, X_anom]).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    y[n_normal:] = 1

    # Shuffle
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    logger.info(
        f"Generated synthetic data: {n_samples} samples, "
        f"{n_features} features, {n_anom} anomalies ({contamination * 100:.1f}%)"
    )
    return X, y


def save_synthetic_data(
    X: np.ndarray,
    y: np.ndarray,
    path: Optional[Path] = None,
) -> Path:
    """Save synthetic data to NPZ file."""
    path = path or SYNTHETIC_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    logger.info(f"Saved synthetic data to {path}")
    return path


def load_synthetic_data(path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load synthetic data from NPZ file."""
    path = path or SYNTHETIC_FILE
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data not found: {path}")
    data = np.load(path)
    logger.info(f"Loaded synthetic data from {path}")
    return data["X"], data["y"]


# =========================================================================
#  Numenta Anomaly Benchmark (NAB) — optional
# =========================================================================

NAB_REPO = "https://raw.githubusercontent.com/numenta/NAB/master/data"
NAB_DATASETS = {
    "ec2_cpu": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "ec2_network": "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "rds_cpu": "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "machine_temp": "realKnownCause/machine_temperature_system_failure.csv",
    "nyc_taxi": "realKnownCause/nyc_taxi.csv",
    "ambient_temp": "realKnownCause/ambient_temperature_system_failure.csv",
}


def download_nab_dataset(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download a NAB dataset CSV by name.

    Args:
        dataset_name: Key from NAB_DATASETS (e.g. 'ec2_cpu', 'machine_temp').
        cache_dir: Directory to cache the CSV.

    Returns:
        Path to the downloaded CSV.

    Raises:
        ValueError: If dataset_name not in NAB_DATASETS.
    """
    import requests

    if dataset_name not in NAB_DATASETS:
        raise ValueError(
            f"Unknown NAB dataset '{dataset_name}'. "
            f"Available: {list(NAB_DATASETS.keys())}"
        )

    cache_dir = cache_dir or (DEFAULT_DATA / "nab")
    csv_path = cache_dir / f"{dataset_name}.csv"

    if csv_path.exists():
        logger.info(f"Using cached NAB dataset: {csv_path}")
        return csv_path

    url = f"{NAB_REPO}/{NAB_DATASETS[dataset_name]}"
    logger.info(f"Downloading NAB dataset '{dataset_name}' from {url}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(csv_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"Saved to {csv_path}")
    return csv_path


def load_nab_dataset(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    value_column: str = "value",
    timestamp_column: str = "timestamp",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NAB dataset as (X, y) for anomaly detection.

    NAB datasets include an 'anomaly' label column (0/1).
    Returns a 1D feature array reshaped to (n_samples, 1) for sklearn compat.

    Args:
        dataset_name: Key from NAB_DATASETS.
        cache_dir: Directory for caching.
        value_column: Column name for the time series values.
        timestamp_column: Column name for timestamps.

    Returns:
        X: (n_samples, 1) feature array
        y: (n_samples,) binary labels
    """
    import pandas as pd

    path = download_nab_dataset(dataset_name, cache_dir)
    df = pd.read_csv(path)

    if "anomaly_label" not in df.columns:
        # Some NAB CSVs don't have labels embedded — warn
        logger.warning(f"No 'anomaly_label' column in {dataset_name}; using zeros")
        df["anomaly_label"] = 0

    X = df[value_column].values.astype(np.float32).reshape(-1, 1)
    y = df["anomaly_label"].values.astype(np.int64)

    n_anom = int(y.sum())
    logger.info(
        f"Loaded NAB '{dataset_name}': {len(X)} samples, "
        f"{n_anom} anomalies ({n_anom / len(X) * 100:.2f}%)"
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
