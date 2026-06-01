"""
Download the PlantVillage dataset from Kaggle.

Downloads from Kaggle via the official Kaggle API and extracts to
data/plantvillage_raw/. Training scripts use torchvision's ImageFolder +
random_split on the raw class folders — no pre-splitting needed here.

Requires Kaggle API credentials. Provide them via one of:
  A) Environment variables: KAGGLE_USERNAME + KAGGLE_KEY (.env file)
  B) ~/.kaggle/kaggle.json file (download from Kaggle settings)
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# Defining the paths
HERE = Path(__file__).resolve().parent  # src/
PROJECT = HERE.parent  # quest-01-cv-plant-disease/
DEFAULT_DATA = PROJECT / "data"  # quest-01-cv-plant-disease/data/


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


def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    download_plantvillage(data_dir)
    print("\nDone! Dataset ready at:", data_dir)


if __name__ == "__main__":
    main()
