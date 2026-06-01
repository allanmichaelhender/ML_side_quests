"""
Train a MobileNetV2 classifier on the PlantVillage dataset.

Phase 1 — Transfer learning: backbone frozen, train classification head only.
Phase 2 — Fine-tuning: unfreeze last 2 convolutional blocks, train end-to-end.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"


# ── Data ───────────────────────────────────────────────────────────────────
def get_transforms():
    """Return train and validation transform pipelines."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, val_transform


def load_data(data_dir: Path, batch_size: int):
    """Load train/val datasets and return dataloaders + class names."""
    train_transform, val_transform = get_transforms()

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: Train/val directories not found in {data_dir}")
        print("Run `python data/download.py` first.")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset.classes


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    """Build MobileNetV2 with a fresh classification head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2):
    """Unfreeze the last `n_blocks` convolutional blocks for fine-tuning."""
    # MobileNetV2 features are in model.features as a list of layers
    blocks = list(model.features)
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    print(
        f"Unfroze last {n_blocks} blocks ({sum(p.numel() for p in blocks[-n_blocks:] if p.requires_grad):,} params trainable)"
    )


# ── Training ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train PlantVillage classifier")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--transfer-epochs",
        type=int,
        default=2,
        help="Phase 1 epochs (backbone frozen)",
    )
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=3, help="Phase 2 epochs (end-to-end)"
    )
    parser.add_argument("--lr-transfer", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use sample data (45 images/class) for quick testing",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    if args.fast:
        data_dir = data_dir / "sample"
        print("⚠️  FAST MODE: using sample data (45 images/class)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = load_data(data_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"{'=' * 60}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"{'=' * 60}")
    print("PHASE 1 — Transfer Learning (backbone frozen)")
    print(f"{'=' * 60}")

    # Only classifier params are trainable
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr_transfer)

    for epoch in range(1, args.transfer_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"  Epoch {epoch:2d}/{args.transfer_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"{elapsed:.1f}s"
        )

    # ── Fine-tune ──────────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("PHASE 2 — Fine-tuning (unfroze last 2 conv blocks)")
    print(f"{'=' * 60}")

    unfreeze_last_blocks(model, n_blocks=2)

    # Now all trainable params (backbone blocks + classifier)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_finetune,
    )

    for epoch in range(1, args.fine_tune_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"  Epoch {epoch:2d}/{args.fine_tune_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"{elapsed:.1f}s"
        )

    # ── Save ───────────────────────────────────────────────────────────────
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc,
            "args": vars(args),
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    # Also save class names as JSON for reuse in other scripts
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print(f"Final validation accuracy: {val_acc:.2f}%")
    print("Done!")


if __name__ == "__main__":
    main()
