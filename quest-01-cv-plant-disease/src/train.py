import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from data_utils import find_class_root, TransformedSubset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"


# Image Preprocessing, returns two transform pipelines, one for training and one for validation
def get_transforms():

    # Default ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training pipeline
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # This is the size our model - MobileNetV2 expects
            # Data augmentation steps to prevent overfitting
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Transforming each image to a tensor
            transforms.ToTensor(),
            # Normalising to better aid the model's learning process
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Transforming valuation set into expected format + mirroring normalization
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def load_data(
    data_dir: Path,
    batch_size: int,
    seed: int = 42,
):
    train_transform, val_transform = get_transforms()

    raw_dir = data_dir / "plantvillage_raw"

    # Logging error and exiting if no data present
    if not raw_dir.exists():
        print(f"ERROR: Raw data not found in {raw_dir}")
        print("Run `python src/download_data.py` first.")
        sys.exit(1)

    # Helper function to find to root dir of all the image classes
    class_root = find_class_root(raw_dir)

    # The image folder method scans the directory, labels each sub directory as a distinct class (assigning integer indicies), and stores everything under its associated class in memory
    full_dataset = datasets.ImageFolder(class_root)

    # Torch random split requires exact lengths, we calculate them here
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * 0.70)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - val_size

    # Generating our train / validation / held-out test split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Save test indices so test.py can load them for truly unseen evaluation
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    test_indices_path = output_dir / "test_indices.json"
    with open(test_indices_path, "w") as f:
        json.dump(test_subset.indices, f)
    print(f"Test indices ({len(test_subset)} samples) saved to {test_indices_path}")

    # Creating our datasets (no images are actually loaded yet, these store the indicies and the instructions)
    train_dataset = TransformedSubset(
        full_dataset, train_subset.indices, train_transform
    )
    val_dataset = TransformedSubset(full_dataset, val_subset.indices, val_transform)

    # Dataloader automates batching, shuffline and parallel loading of the data
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Classes ({len(full_dataset.classes)}): {full_dataset.classes}")
    print(
        f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}  |  Test samples: {test_size}"
    )
    # Return the two data loaders and the classes (as string names from the folder names)
    return train_loader, val_loader, full_dataset.classes


# Creating the pytorch model
def build_model(num_classes: int):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )  # Loading the default pretrained weights for the model

    # Freezing all params, since we redefine the classifier head those are fresh weights and thus uneffected, we do this to require only training on the head itself
    for param in model.parameters():
        param.requires_grad = False

    input_features = model.classifier[
        1
    ].in_features  # Extracting the default model head input features
    model.classifier = nn.Sequential(
        nn.Dropout(
            0.2
        ),  # Zeroing out 20% of the features every time to prevent overfitting
        nn.Linear(
            input_features, num_classes
        ),  # Mapping to 15 features, hightest score is model prediction
    )
    return model


# This function unfreezes the last two blocks in the model (before the classification head) to allow fine tuning of the model
def unfreeze_last_blocks(model: models.MobileNet_V2, n_blocks: int = 2):
    blocks = list(model.features)

    # Unfreezing the last n blocks ready for training
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True  # Enabling gradient calulations on the backward pass and allowing updates during training
    print(f"Unfroze last {n_blocks} blocks")


# Function for performing one epoch of training
def train_one_epoch(model, loader, lr, device):
    model.train()  # Setting the model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()  # Setting our loss function
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )  # Setting our optimiser, filtering by only those which require gradients

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Clearing the gradients from the previous step, by default pytorch accumulates them
        outputs = model(inputs)  # Forward pass. calculating the model's predicitions
        loss = criterion(outputs, labels)
        loss.backward()  # Backwards step to calculate gradients
        optimizer.step()  # Takes the gradients and updates the weights accordingly

        # Calculating runnning statistic per batch
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    SEED = 42
    BATCH_SIZE = 32
    TRANSFER_EPOCHS = 2
    FINE_TUNE_EPOCHS = 3
    LR_TRANSFER = 1e-3
    LR_FINETUNE = 1e-4

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = DEFAULT_DATA
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    train_loader, val_loader, class_names = load_data(data_dir, BATCH_SIZE, seed=SEED)
    num_classes = len(class_names)
    print(f"{'=' * 60}")

    model = build_model(num_classes)
    model = model.to(device)

    print(f"{'=' * 60}")
    print("PHASE 1 — Transfer Learning (backbone frozen)")
    print(f"{'=' * 60}")

    for epoch in range(1, TRANSFER_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, LR_TRANSFER, device
        )
        val_loss, val_acc = validate(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"  Epoch {epoch:2d}/{TRANSFER_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | {elapsed:.1f}s"
        )

    print(f"{'=' * 60}")
    print("PHASE 2 — Fine-tuning")
    print(f"{'=' * 60}")

    unfreeze_last_blocks(model, n_blocks=2)

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, LR_FINETUNE, device
        )
        val_loss, val_acc = validate(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"  Epoch {epoch:2d}/{FINE_TUNE_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | {elapsed:.1f}s"
        )

    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc,
            "config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "transfer_epochs": TRANSFER_EPOCHS,
                "fine_tune_epochs": FINE_TUNE_EPOCHS,
                "lr_transfer": LR_TRANSFER,
                "lr_finetune": LR_FINETUNE,
            },
        },
        model_path,
    )
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print(f"\nModel saved to {model_path}")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print("Done!")


if __name__ == "__main__":
    main()
