"""Data loading, preprocessing, and tokenization for sentiment analysis."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"

# Binary sentiment: amazon_polarity uses 0 = negative, 1 = positive
LABEL_NAMES = ["negative", "positive"]

# Parquet files on HuggingFace Hub
HF_BASE = "https://huggingface.co/datasets/amazon_polarity/resolve/main/amazon_polarity"
PARQUET_FILES = {
    "train": [f"train-0000{i}-of-00004.parquet" for i in range(4)],
    "test": ["test-00000-of-00001.parquet"],
}


def download_data(data_dir: Path = DEFAULT_DATA):
    """Download Parquet files from HuggingFace Hub to local cache."""
    data_dir.mkdir(parents=True, exist_ok=True)

    local_files = {}
    for split, filenames in PARQUET_FILES.items():
        split_files = []
        for fname in filenames:
            local_path = data_dir / fname
            if not local_path.exists():
                url = f"{HF_BASE}/{fname}"
                print(f"Downloading {url}...")
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(
                    f"  Saved to {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)"
                )
            else:
                print(f"  {local_path.name} already exists, skipping")
            split_files.append(str(local_path))
        local_files[split] = split_files
    return local_files


def load_amazon_reviews(
    data_dir: Path = DEFAULT_DATA,
    max_samples: int = 10_000,
    seed: int = 42,
    download_if_missing: bool = True,
) -> DatasetDict:
    """Load Amazon Polarity dataset from local Parquet files.

    Returns a DatasetDict with 'train', 'validation' (subset of train),
    and 'test' splits.  Labels are 0 = negative, 1 = positive.
    If max_samples is set, the training set is stratified-sampled.
    """
    # Download parquet files if needed
    if not list(data_dir.glob("*.parquet")):
        if download_if_missing:
            download_data(data_dir)
        else:
            print(f"ERROR: No Parquet files found in {data_dir}")
            print(
                "Run `python -c 'from data_utils import download_data; download_data()'` first."
            )
            sys.exit(1)

    print("Loading Amazon Polarity dataset from local Parquet files...")
    # Only read the first parquet file (900k rows) — more than enough for our needs
    train_files = sorted(data_dir.glob("train-*.parquet"))
    train_df = pd.read_parquet(train_files[0])
    test_df = pd.read_parquet(data_dir / "test-00000-of-00001.parquet")

    print(f"  Train: {len(train_df)} samples (from {train_files[0].name})")
    print(f"  Test:  {len(test_df)} samples")

    # Build train/validation splits
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_df))
    val_size = min(10_000, len(train_df) // 10)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # Stratified subsample training set
    train_df_subset = _stratified_sample_df(train_df_part, max_samples, seed)

    # Also subsample validation to a reasonable size for quick eval
    val_max = min(len(val_df), 5_000)
    val_df = _stratified_sample_df(val_df, val_max, seed)

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df_subset[["label", "title", "content"]])
    val_ds = Dataset.from_pandas(val_df[["label", "title", "content"]])
    test_ds = Dataset.from_pandas(test_df[["label", "title", "content"]])

    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    # Print class distribution
    for split_name in dataset.keys():
        labels = dataset[split_name]["label"]
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        print(
            f"  {split_name}: {total} samples -> "
            f"neg={counts[0]} ({counts[0] / total * 100:.1f}%), "
            f"pos={counts[1]} ({counts[1] / total * 100:.1f}%)"
        )

    return dataset


def _stratified_sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified subsample a DataFrame to n rows while keeping class balance."""
    rng = np.random.default_rng(seed)
    samples_per_class = n // df["label"].nunique()
    result = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        chosen = class_df.sample(
            n=min(samples_per_class, len(class_df)), random_state=rng.integers(0, 2**31)
        )
        result.append(chosen)
    sampled = pd.concat(result, ignore_index=True)
    return sampled.sample(frac=1, random_state=rng.integers(0, 2**31)).reset_index(
        drop=True
    )


def get_tokenizer():
    """Load DistilBERT tokenizer."""
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 256,
) -> DatasetDict:
    """Tokenize the dataset with the given tokenizer."""

    def tokenize_fn(examples):
        # Combine title and content for richer context
        texts = [f"{t} {c}" for t, c in zip(examples["title"], examples["content"])]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print(f"Tokenizing with max_length={max_length}...")
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["title", "content"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def save_label_info(output_dir: Path):
    """Save label names as JSON for use in evaluation and demo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {"label_names": LABEL_NAMES}
    path = output_dir / "label_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Label info saved to {path}")
    return path


def load_label_info(output_dir: Path = None) -> dict:
    """Load label info from JSON."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT
    path = output_dir / "label_info.json"
    with open(path) as f:
        return json.load(f)
