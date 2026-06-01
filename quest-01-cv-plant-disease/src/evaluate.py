"""
Evaluate the trained model: accuracy, per-class metrics, confusion matrix,
Grad-CAM visualisations, and ONNX export.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_OUTPUT = PROJECT / "results"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model.pt"

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
    parser = argparse.ArgumentParser(description="Evaluate PlantVillage model")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_dir = Path(args.data_dir) / "val"
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Validation samples: {len(val_dataset)}")

    model, class_names = load_checkpoint(Path(args.model_path), device)
    metrics, cm = compute_metrics(model, val_loader, class_names, device)

    metrics["confusion_matrix"] = cm.tolist()
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        cm, class_names, output_dir / "figures" / "confusion_matrix.png"
    )

    if args.gradcam_samples > 0:
        generate_gradcam(
            model,
            val_loader,
            class_names,
            device,
            output_dir,
            num_samples=args.gradcam_samples,
        )
    if args.export_onnx:
        export_onnx(model, class_names, output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
