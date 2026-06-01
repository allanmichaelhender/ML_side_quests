import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from data_utils import find_class_root, TransformedSubset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model.pt"

# Default ImageNet Normalisation values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# Loading in model from the checkpoint/save defined at the end of train.py
def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = len(checkpoint["class_names"])
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    return model, class_names


@torch.no_grad()
def compute_metrics(model, val_loader, class_names, device):
    all_preds, all_labels = [], []
    model.eval()

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    # Overall fraction of correct predictions
    acc = accuracy_score(all_labels, all_preds)

    # N×N matrix: rows = true class, cols = predicted class; diagonal = correct
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class precision, recall, f1, support as a formatted string
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )

    # Raw per-class metrics: precision, recall, f1, support (each is a list)
    #   precision = TP / (TP + FP)  — how many predicted for this class were correct
    #   recall    = TP / (TP + FN)  — how many actual instances of this class were found
    #   f1        = harmonic mean of precision and recall
    #   support   = number of actual instances of this class in the validation set
    per_class = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names))
    )

    print(f"\n{'=' * 60}")
    print(f"Validation Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"{'=' * 60}\n{report}")

    return {
        "accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "precision": [round(p, 4) for p in per_class[0].tolist()],
            "recall": [round(r, 4) for r in per_class[1].tolist()],
            "f1": [round(f, 4) for f in per_class[2].tolist()],
            "support": per_class[3].tolist(),
        },
    }, cm


def plot_confusion_matrix(cm, class_names, save_path: Path):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# We use GradCAM to visualise which parts of the image the model focuses on for its predictions.
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        logits = self.model(x.unsqueeze(0))
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam).squeeze(0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def generate_gradcam(
    model, val_loader, class_names, device, output_dir: Path, num_samples: int = 8
):
    cam_dir = output_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    samples_processed = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            if samples_processed >= num_samples:
                return
            x = inputs[i]
            true_label = labels[i].item()
            heatmap = gradcam.generate(x)

            img = x.cpu().numpy().transpose(1, 2, 0)
            img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = np.clip(img, 0, 1)

            h, w = img.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = np.clip(0.5 * img + 0.5 * (heatmap_colored / 255.0), 0, 1)

            with torch.no_grad():
                pred = model(x.unsqueeze(0)).argmax(dim=1).item()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=8)
            axes[0].axis("off")
            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=8)
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {class_names[pred]}", fontsize=8)
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(
                cam_dir / f"gradcam_sample_{samples_processed:03d}.png", dpi=150
            )
            plt.close()
            samples_processed += 1
    print(f"Grad-CAM visualisations saved to {cam_dir}/")


# Export to ONNX (Open Neural Network Exchange) — a cross-platform model format.
# This decouples deployment from PyTorch: the Streamlit app (app.py) loads
# model.onnx via lightweight ONNX Runtime instead of needing PyTorch.
# class_names.json is saved alongside so downstream consumers can map output
# indices (0, 1, 2, ...) back to human-readable disease names.
def export_onnx(model, class_names, output_dir: Path):
    onnx_path = output_dir / "model.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"ONNX model exported to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test PlantVillage model on held-out test set"
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=8,
        help="Number of Grad-CAM visualisations to generate (0 to skip)",
    )
    args = parser.parse_args()

    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_dir = DEFAULT_DATA / "plantvillage_raw"
    if not raw_dir.exists():
        print(f"ERROR: Raw data not found in {raw_dir}")
        print("Run `python src/download_data.py` first.")
        sys.exit(1)

    class_root = find_class_root(raw_dir)

    # Load the held-out test indices saved during training
    test_indices_path = output_dir / "test_indices.json"
    if not test_indices_path.exists():
        print(f"ERROR: Test indices not found at {test_indices_path}")
        print("Re-run `python src/train.py` to generate them.")
        sys.exit(1)

    with open(test_indices_path) as f:
        test_indices = json.load(f)

    # Build the test dataset from those indices (truly unseen — never used during training)
    full_dataset = datasets.ImageFolder(class_root)
    test_dataset = TransformedSubset(full_dataset, test_indices, get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples (held-out): {len(test_dataset)}")

    model, class_names = load_checkpoint(DEFAULT_MODEL, device)
    metrics, cm = compute_metrics(model, test_loader, class_names, device)

    metrics["confusion_matrix"] = cm.tolist()
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        cm, class_names, output_dir / "figures" / "confusion_matrix.png"
    )

    if args.gradcam_samples > 0:
        generate_gradcam(
            model,
            test_loader,
            class_names,
            device,
            output_dir,
            num_samples=args.gradcam_samples,
        )
    export_onnx(model, class_names, output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
