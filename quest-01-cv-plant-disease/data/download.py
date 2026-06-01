"""
Download and prepare the PlantVillage dataset.

Downloads from Kaggle via the official Kaggle API, organises into
train/val structure, and creates a small sample subset for quick testing.

Requires Kaggle API credentials. Provide them via one of:
  A) Environment variables: KAGGLE_USERNAME + KAGGLE_KEY (.env file)
  B) ~/.kaggle/kaggle.json file (download from Kaggle settings)

See: https://www.kaggle.com/docs/api#getting-started-installation-&-authentication
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_plantvillage(data_dir: Path) -> Path:
    """Download PlantVillage dataset from Kaggle. Returns the extracted path."""
    dest = data_dir / "plantvillage_raw"
    zip_path = data_dir / "plantdisease.zip"

    if dest.exists():
        print(f"Removing existing {dest}")
        shutil.rmtree(dest)

    # Check credentials before attempting
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

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()  # remove zip after extraction
    print(f"Extracted to {dest}")
    return dest


def organise_splits(
    raw_dir: Path, data_dir: Path, val_split: float = 0.2, seed: int = 42
):
    """
    Find class folders under raw_dir, split into train/val, and copy
    to data_dir/train and data_dir/val.

    The Kaggle "emmarex/plantdisease" dataset has no pre-made splits —
    all class folders sit flat under a subdirectory.
    """
    import random

    random.seed(seed)

    # Find the directory that actually contains class folders
    # Could be raw_dir or raw_dir/PlantVillage
    candidates = [raw_dir] + sorted(raw_dir.iterdir())
    class_root = None
    for c in candidates:
        if not c.is_dir():
            continue
        subdirs = [d for d in c.iterdir() if d.is_dir()]
        # Skip if it looks like a nested folder with the same name pattern
        if len(subdirs) >= 3 and not any(d.name == raw_dir.name for d in subdirs):
            class_root = c
            break
    if class_root is None:
        print(f"ERROR: Could not find class folders under {raw_dir}")
        sys.exit(1)

    # Exclude the nested "PlantVillage" folder — it's a duplicate of the parent
    class_dirs = sorted(
        d for d in class_root.iterdir() if d.is_dir() and d.name != class_root.name
    )
    print(f"Found {len(class_dirs)} classes under {class_root}")

    for split_name in ("train", "val"):
        dst = data_dir / split_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    for class_dir in class_dirs:
        images = sorted(class_dir.iterdir())
        random.shuffle(images)
        split_idx = max(1, int(len(images) * (1 - val_split)))

        # Use relative symlinks (instant) instead of copying
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


def create_sample(raw_dir: Path, sample_dir: Path, samples_per_class: int = 3) -> None:
    """Create a tiny sample subset for quick testing."""
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    for split in ("train", "val"):
        src_split = raw_dir / split
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Destination directory (default: same as this script)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only recreate the sample subset from existing data",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    sample_dir = data_dir / "sample"

    if args.sample_only:
        train_dir = data_dir / "train"
        if not train_dir.exists():
            print("Train/val splits not found. Run without --sample-only first.")
            sys.exit(1)
        create_sample(data_dir, sample_dir)
        return

    # Full download
    raw_dir = download_plantvillage(data_dir)
    organise_splits(raw_dir, data_dir)
    create_sample(data_dir, sample_dir)
    print("\nDone! Dataset ready at:", data_dir)


if __name__ == "__main__":
    main()
