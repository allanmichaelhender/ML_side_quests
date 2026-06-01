"""
Download and prepare the PlantVillage dataset.

Downloads from Kaggle via the official Kaggle API, organises into
train/val splits, and creates a small sample subset for quick testing.

Requires Kaggle API credentials. Provide them via one of:
  A) Environment variables: KAGGLE_USERNAME + KAGGLE_KEY (.env file)
  B) ~/.kaggle/kaggle.json file (download from Kaggle settings)
"""

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"


def download_plantvillage(data_dir: Path) -> Path:
    """Download PlantVillage dataset from Kaggle. Returns the extracted path."""
    dest = data_dir / "plantvillage_raw"
    zip_path = data_dir / "plantdisease.zip"

    if dest.exists():
        print(f"Removing existing {dest}")
        shutil.rmtree(dest)

    has_env_vars = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    has_kaggle_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_env_vars and not has_kaggle_file:
        print("ERROR: Kaggle API credentials not found.")
        print()
        print("  Option A — Environment variables (recommended with Docker):")
        print("    Create a .env file in the project root:")
        print("      KAGGLE_USERNAME=your_username")
        print("      KAGGLE_KEY=your_api_key")
        print()
        print("  Option B — Credentials file:")
        print("    Place kaggle.json at ~/.kaggle/kaggle.json")
        print(
            "    Download from: https://www.kaggle.com/settings -> API -> Create New Token"
        )
        print()
        sys.exit(1)

    print("Downloading PlantVillage dataset from Kaggle...")
    print("  Dataset: emmarex/plantdisease")
    if has_env_vars:
        print("  Auth: environment variables")
    else:
        print("  Auth: ~/.kaggle/kaggle.json")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("emmarex/plantdisease", path=str(data_dir), unzip=False)
    print(f"Downloaded to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    print(f"Extracted to {dest}")
    return dest


def organise_splits(
    raw_dir: Path, data_dir: Path, val_split: float = 0.2, seed: int = 42
):
    """Find class folders, split into train/val using relative symlinks."""
    random.seed(seed)

    candidates = [raw_dir] + sorted(raw_dir.iterdir())
    class_root = None
    for c in candidates:
        if not c.is_dir():
            continue
        subdirs = [d for d in c.iterdir() if d.is_dir()]
        if len(subdirs) >= 3 and not any(d.name == raw_dir.name for d in subdirs):
            class_root = c
            break
    if class_root is None:
        print(f"ERROR: Could not find class folders under {raw_dir}")
        sys.exit(1)

    # Exclude nested folders with the same name (duplicate)
    class_dirs = sorted(
        d for d in class_root.iterdir() if d.is_dir() and d.name != class_root.name
    )
    print(f"Found {len(class_dirs)} classes under {class_root}")

    for split_name in ("train", "val"):
        dst = data_dir / split_name
        if dst.exists():
            try:
                shutil.rmtree(dst)
            except OSError:
                import subprocess

                subprocess.run(["rm", "-rf", str(dst)], check=True)
        dst.mkdir(parents=True)

    for class_dir in class_dirs:
        images = sorted(class_dir.iterdir())
        random.shuffle(images)
        split_idx = max(1, int(len(images) * (1 - val_split)))

        for img in images[:split_idx]:
            dest_dir = data_dir / "train" / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            rel = os.path.relpath(img, dest_dir)
            link = dest_dir / img.name
            if not link.exists():
                link.symlink_to(rel)

        for img in images[split_idx:]:
            dest_dir = data_dir / "val" / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            rel = os.path.relpath(img, dest_dir)
            link = dest_dir / img.name
            if not link.exists():
                link.symlink_to(rel)

    for split_name in ("train", "val"):
        split_dir = data_dir / split_name
        n_classes = len(list(split_dir.iterdir()))
        n_images = sum(
            len(list(p.iterdir())) for p in split_dir.iterdir() if p.is_dir()
        )
        print(f"  {split_name}: {n_classes} classes, {n_images} images")


def create_sample(data_dir: Path, samples_per_class: int = 3):
    """Create a tiny sample subset from the train/val splits."""
    sample_dir = data_dir / "sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    for split in ("train", "val"):
        src_split = data_dir / split
        dst_split = sample_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)

        for class_dir in sorted(src_split.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(class_dir.iterdir())[:samples_per_class]
            if not images:
                continue
            dst_class = dst_split / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)
            for img in images:
                rel = os.path.relpath(img, dst_class)
                link = dst_class / img.name
                if not link.exists():
                    link.symlink_to(rel)

        n_classes = len(list(dst_split.iterdir()))
        n_images = sum(
            len(list(p.iterdir())) for p in dst_split.iterdir() if p.is_dir()
        )
        print(f"  Sample {split}: {n_classes} classes, {n_images} images")


def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    if args.sample_only:
        if not (data_dir / "train").exists():
            print("Train/val splits not found. Run without --sample-only first.")
            sys.exit(1)
        create_sample(data_dir)
        return

    raw_dir = download_plantvillage(data_dir)
    organise_splits(raw_dir, data_dir)
    create_sample(data_dir)
    print("\nDone! Dataset ready at:", data_dir)


if __name__ == "__main__":
    main()
