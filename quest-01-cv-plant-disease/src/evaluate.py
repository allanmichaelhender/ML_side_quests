"""
Evaluate the trained model: accuracy, per-class metrics, confusion matrix,
Grad-CAM visualisations, and ONNX export.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import cv2
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model.pt"


# ── Helpers ────────────────────────────────────────────────────────────────
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


def load_checkpoint(model_path: Path, device: torch.device):
    """Load model checkpoint and return model + class_names."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = len(checkpoint["class_names"])
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    return model, class_names


# ── Metrics ────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(model, val_loader, class_names, device):
    """Compute accuracy, per-class precision/recall/F1, confusion matrix."""
    all_preds = []
    all_labels = []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    per_class = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names))
    )

    print(f"\n{'=' * 60}")
    print(f"Validation Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"{'=' * 60}")
    print("\nPer-Class Metrics:")
    print(report)

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
    """Plot and save a confusion matrix figure."""
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ── Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    """Compute Grad-CAM heatmap for a target layer."""

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
        """Generate a Grad-CAM heatmap."""
        # Forward
        logits = self.model(x.unsqueeze(0))
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1)  # [1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def generate_gradcam(
    model, val_loader, class_names, device, output_dir: Path, num_samples: int = 8
):
    """Generate Grad-CAM overlays for a few validation samples."""
    cam_dir = output_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # Use the last convolutional layer of MobileNetV2 features
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

            # Denormalize image for display
            img = x.cpu().numpy().transpose(1, 2, 0)
            img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = np.clip(img, 0, 1)

            # Resize heatmap to image size
            h, w = img.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))

            # Overlay
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = 0.5 * img + 0.5 * (heatmap_colored / 255.0)
            overlay = np.clip(overlay, 0, 1)

            # Get predicted class
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                pred = logits.argmax(dim=1).item()
            pred_name = class_names[pred]
            true_name = class_names[true_label]

            # Plot side-by-side
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=8)
            axes[0].axis("off")

            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=8)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {pred_name}", fontsize=8)
            axes[2].axis("off")

            plt.tight_layout()
            save_path = cam_dir / f"gradcam_sample_{samples_processed:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"Grad-CAM visualisations saved to {cam_dir}/")


# ── ONNX Export ────────────────────────────────────────────────────────────
def export_onnx(model, class_names, output_dir: Path):
    """Export the model to ONNX format."""
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
    print(f"ONNX model exported to {onnx_path}")

    # Also save a label mapping for ONNX inference
    label_path = output_dir / "class_names.json"
    with open(label_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {label_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate PlantVillage model")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=8,
        help="Number of Grad-CAM samples to generate (0 to skip)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export to ONNX after evaluation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    val_dir = Path(args.data_dir) / "val"
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Validation samples: {len(val_dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model, class_names = load_checkpoint(Path(args.model_path), device)

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics, cm = compute_metrics(model, val_loader, class_names, device)

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    # Convert cm to list for JSON serialisation
    metrics["confusion_matrix"] = cm.tolist()
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Confusion matrix plot
    plot_confusion_matrix(
        cm, class_names, output_dir / "figures" / "confusion_matrix.png"
    )

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    if args.gradcam_samples > 0:
        generate_gradcam(
            model,
            val_loader,
            class_names,
            device,
            output_dir,
            num_samples=args.gradcam_samples,
        )

    # ── ONNX ───────────────────────────────────────────────────────────────
    if args.export_onnx:
        export_onnx(model, class_names, output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
