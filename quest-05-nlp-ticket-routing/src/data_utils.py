"""
Data utilities for Support Ticket Routing.

Loads the Banking77 dataset (77 intents for banking customer queries),
prepares train/val/test splits, builds TF-IDF vectors, and tokenizes
for DistilBERT.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"


def load_banking77(
    max_samples: int = 10_000,
    seed: int = 42,
) -> DatasetDict:
    """Load Banking77 dataset from Hugging Face.

    Returns a DatasetDict with 'train', 'validation', and 'test' splits.
    Labels are integers 0-76 mapped to intent categories.
    """
    print("Loading Banking77 dataset from Hugging Face...")
    dataset = load_dataset("PolyAI/banking77")

    # Get label names
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    print(f"  {num_labels} intent categories loaded")
    print(f"  Full train set: {len(dataset['train'])} samples")
    print(f"  Full test set:  {len(dataset['test'])} samples")

    # Stratified subsample training set
    rng = np.random.default_rng(seed)
    train_df = dataset["train"].to_pandas()

    if max_samples is not None and max_samples < len(train_df):
        train_df = _stratified_sample_df(train_df, max_samples, seed, label_col="label")
        print(f"  Subsampled train to {len(train_df)} samples (stratified)")

    # Split train into train/validation (90/10)
    indices = rng.permutation(len(train_df))
    val_size = int(len(train_df) * 0.1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_part = train_df.iloc[val_idx].reset_index(drop=True)

    # Convert test set directly
    test_df = dataset["test"].to_pandas()

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_part[["text", "label"]])
    val_ds = Dataset.from_pandas(val_part[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    result = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    # Print class distribution
    for split_name in result.keys():
        labels = result[split_name]["label"]
        total = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        print(
            f"  {split_name}: {total} samples, {len(unique)} classes"
            f"  ({counts.min()}-{counts.max()} per class)"
        )

    return result, label_names


def _stratified_sample_df(
    df: pd.DataFrame, n: int, seed: int = 42, label_col: str = "label"
) -> pd.DataFrame:
    """Stratified subsample a DataFrame to n rows while keeping class balance."""
    rng = np.random.default_rng(seed)
    samples_per_class = max(1, n // df[label_col].nunique())
    result = []
    for label in df[label_col].unique():
        class_df = df[df[label_col] == label]
        chosen = class_df.sample(
            n=min(samples_per_class, len(class_df)),
            random_state=int(rng.integers(0, 2**31)),
        )
        result.append(chosen)
    sampled = pd.concat(result, ignore_index=True)
    return sampled.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(
        drop=True
    )


def build_tfidf_pipeline(max_features: int = 5000, ngram_range=(1, 2)):
    """Build a TF-IDF vectorizer configured for ticket text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True,
    )


def get_tokenizer():
    """Load DistilBERT tokenizer."""
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize the dataset with the given tokenizer."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print(f"Tokenizing with max_length={max_length}...")
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def save_label_info(output_dir: Path, label_names: list):
    """Save label names as JSON for use in evaluation and demo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "label_names": label_names,
        "num_labels": len(label_names),
    }
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
