"""Shared utilities for dataset handling."""

import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


def find_class_root(raw_dir: Path) -> Path:
    """Find the directory containing actual class folders within the raw extract.

    The Kaggle zip has a nested structure (PlantVillage/PlantVillage/<classes>).
    This walks candidates to find the right level.
    """
    candidates = [raw_dir] + sorted(raw_dir.iterdir())
    for c in candidates:
        if not c.is_dir():
            continue
        subdirs = [d for d in c.iterdir() if d.is_dir()]
        if len(subdirs) >= 3 and not any(d.name == raw_dir.name for d in subdirs):
            return c
    print(f"ERROR: Could not find class folders under {raw_dir}")
    sys.exit(1)


class TransformedSubset(Subset):
    """A Subset that applies a transform to items from the underlying dataset.

    Useful when using random_split with datasets that return PIL images
    (e.g. ImageFolder without a transform), so each split can have its
    own transform pipeline.
    """

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y
