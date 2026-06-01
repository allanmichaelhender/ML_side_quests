"""
Download and prepare the PlantVillage dataset.

Downloads from Kaggle via kagglehub, organises into train/val structure,
and creates a small sample subset for quick testing.
"""

import argparse
import shutil
import sys
from pathlib import Path

import kagglehub


def download_plantvillage(data_dir: Path) -> Path:
    """Download PlantVillage dataset from Kaggle. Returns the extracted path."""
    print("Downloading PlantVillage dataset from Kaggle...")
    path = kagglehub.dataset_download("emmarex/plantdisease")
    src = Path(path)
    print(f"Downloaded to {src}")

    # Copy to our data directory
    dest = data_dir / "plantvillage_raw"
    if dest.exists():
        print(f"Removing existing {dest}")
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"Copied to {dest}")
    return dest


def organise_splits(raw_dir: Path, data_dir: Path) -> None:
    """Copy train/val splits into a clean structure."""
    for split in ("train", "val"):
        src_split = raw_dir / split
        dst_split = data_dir / split
        if dst_split.exists():
            shutil.rmtree(dst_split)
        shutil.copytree(src_split, dst_split)
        n_classes = len(list(dst_split.iterdir()))
        n_images = sum(
            len(list(p.iterdir())) for p in dst_split.iterdir() if p.is_dir()
        )
        print(f"  {split}: {n_classes} classes, {n_images} images")


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
                shutil.copy2(img, dst_class / img.name)

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
        raw_dir = data_dir / "plantvillage_raw"
        if not raw_dir.exists():
            print("Raw dataset not found. Run without --sample-only first.")
            sys.exit(1)
        create_sample(raw_dir, sample_dir)
        return

    # Full download
    raw_dir = download_plantvillage(data_dir)
    organise_splits(raw_dir, data_dir)
    create_sample(raw_dir, sample_dir)
    print("\nDone! Dataset ready at:", data_dir)


if __name__ == "__main__":
    main()
