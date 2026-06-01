"""
Streamlit app for Plant Disease Classification.

Tabs:
  - Results Dashboard — view training/evaluation findings
  - Live Inference   — upload a leaf image for real-time prediction
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
MODEL_PATH = RESULTS_DIR / "model.onnx"
MODEL_PT_PATH = RESULTS_DIR / "model.pt"
LABELS_PATH = RESULTS_DIR / "class_names.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CM_PATH = RESULTS_DIR / "figures" / "confusion_matrix.png"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
SAMPLE_DIR = HERE / "data" / "sample"

# ── Transforms ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Disease Classification")
st.markdown(
    "MobileNetV2 trained on the PlantVillage dataset — **38 classes** across 14 crop species."
)


# ── Helpers ────────────────────────────────────────────────────────────────


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_onnx_model():
    if not MODEL_PATH.exists():
        return None
    return ort.InferenceSession(str(MODEL_PATH))


@st.cache_data
def load_class_names():
    if not LABELS_PATH.exists():
        return None
    with open(LABELS_PATH) as f:
        return json.load(f)


def parse_classification_report(report_text: str):
    """Parse sklearn's classification_report into a list of dicts."""
    lines = report_text.strip().split("\n")
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0][0].isalpha():
            continue
        # Format: class_name ... precision recall f1-score support
        support = parts[-1]
        f1 = parts[-2]
        recall = parts[-3]
        precision = parts[-4]
        name = " ".join(parts[:-4])
        try:
            rows.append(
                {
                    "class": name,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(support),
                }
            )
        except ValueError:
            continue
    return rows


# ── Inference function ─────────────────────────────────────────────────────


def predict(
    image: Image.Image, session: ort.InferenceSession, class_names: list[str]
) -> tuple[list[dict], np.ndarray]:
    """Run ONNX inference. Returns (predictions, probabilities)."""
    img_tensor = TRANSFORM(image).unsqueeze(0).numpy().astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img_tensor})[0]

    # Softmax
    exp = np.exp(outputs - outputs.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    top_indices = np.argsort(probs[0])[::-1]

    results = []
    for idx in top_indices:
        results.append(
            {
                "class": class_names[idx],
                "confidence": float(probs[0, idx]),
            }
        )
    return results, probs[0]


# ── Grad-CAM ────────────────────────────────────────────────────────────────


class GradCAM:
    """Generate class activation maps via gradient backpropagation."""

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


@st.cache_resource
def load_pytorch_model():
    """Load the PyTorch model checkpoint for Grad-CAM generation."""
    checkpoint = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=True)
    num_classes = len(checkpoint["class_names"])
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_gradcam_overlay(
    pil_image: Image.Image, model, class_names: list[str], top_idx: int = 0
) -> plt.Figure:
    """Generate a Grad-CAM overlay figure for a PIL image."""
    device = next(model.parameters()).device
    input_tensor = TRANSFORM(pil_image).to(device)

    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
        pred_idx = logits.argmax(dim=1).item()

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx=pred_idx)

    # Prepare original image for display
    img = input_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)

    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.5 * img + 0.5 * (heatmap_colored / 255.0), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=9)
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay — Pred: {class_names[pred_idx].replace('_', ' ')}", fontsize=9
    )
    axes[2].axis("off")
    plt.tight_layout()
    return fig


# ── Load data once ─────────────────────────────────────────────────────────
metrics = load_metrics()
class_names = load_class_names()
session = load_onnx_model()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_results, tab_inference = st.tabs(["📊 Results Dashboard", "🔬 Live Inference"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab_results:
    if metrics is None:
        st.warning(
            "No results found. Run `python src/train.py` then `python src/test.py` first."
        )
    else:
        # ── Top-level metrics ──────────────────────────────────────────────
        st.subheader("Overall Performance")

        acc = metrics["accuracy"]
        report_lines = parse_classification_report(metrics["classification_report"])

        if report_lines:
            macro_f1 = np.mean([r["f1"] for r in report_lines])
            weighted_f1 = np.average(
                [r["f1"] for r in report_lines],
                weights=[r["support"] for r in report_lines],
            )
        else:
            macro_f1 = weighted_f1 = 0.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Validation Accuracy", f"{acc * 100:.2f}%")
        with col2:
            st.metric("Macro Avg F1", f"{macro_f1:.4f}")
        with col3:
            st.metric("Weighted Avg F1", f"{weighted_f1:.4f}")
        with col4:
            n_classes = len(class_names) if class_names else "?"
            st.metric("Number of Classes", str(n_classes))

        # ── Confusion Matrix ───────────────────────────────────────────────
        st.subheader("Confusion Matrix")
        if CM_PATH.exists():
            st.image(str(CM_PATH), width="stretch")
        else:
            st.info("Confusion matrix image not found.")

        # ── Per-Class Performance ──────────────────────────────────────────
        st.subheader("Per-Class Performance")

        if report_lines:
            report_sorted = sorted(report_lines, key=lambda r: r["f1"])

            fig, ax = plt.subplots(figsize=(12, max(6, len(report_sorted) * 0.35)))
            classes_display = [
                r["class"].replace("_", " ").title() for r in report_sorted
            ]
            f1_scores = [r["f1"] for r in report_sorted]
            colors = [
                "#e74c3c" if f < 0.92 else "#f39c12" if f < 0.95 else "#27ae60"
                for f in f1_scores
            ]

            bars = ax.barh(
                range(len(classes_display)), f1_scores, color=colors, edgecolor="white"
            )
            ax.set_yticks(range(len(classes_display)))
            ax.set_yticklabels(classes_display, fontsize=8)
            ax.set_xlabel("F1 Score", fontsize=10)
            ax.set_title("Per-Class F1 Score (sorted ascending)", fontsize=12)
            ax.set_xlim(0, 1.05)
            ax.axvline(
                0.9606,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label="Overall Accuracy (96.06%)",
            )
            ax.legend(fontsize=8, loc="lower right")

            for bar, f1 in zip(bars, f1_scores):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{f1:.3f}",
                    va="center",
                    fontsize=7,
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Expandable detail table
            with st.expander("📋 View full per-class table"):
                rows_data = []
                for r in sorted(report_lines, key=lambda x: x["class"]):
                    rows_data.append(
                        {
                            "Class": r["class"].replace("_", " ").title(),
                            "Precision": f"{r['precision']:.4f}",
                            "Recall": f"{r['recall']:.4f}",
                            "F1 Score": f"{r['f1']:.4f}",
                            "Support": r["support"],
                        }
                    )
                st.dataframe(rows_data, width="stretch", hide_index=True)
        else:
            st.info("Could not parse classification report.")

        # ── Misclassification Insights ─────────────────────────────────────
        st.subheader("🔍 Confusion Analysis")
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            n = len(cm)
            off_diag = []
            for i in range(n):
                for j in range(n):
                    if i != j and cm[i, j] > 0:
                        off_diag.append((i, j, int(cm[i, j])))
            off_diag.sort(key=lambda x: x[2], reverse=True)

            if class_names and len(class_names) == n:
                st.markdown("**Top misclassifications (true → predicted):**")
                top_confusions = off_diag[:9]
                cols = st.columns(3)
                for idx, (true_i, pred_j, count) in enumerate(top_confusions):
                    true_name = class_names[true_i].replace("_", " ").title()
                    pred_name = class_names[pred_j].replace("_", " ").title()
                    with cols[idx % 3]:
                        st.metric(
                            label=f"{true_name}  →  {pred_name}",
                            value=f"{count} samples",
                        )
            else:
                st.info("Class names not available for confusion analysis.")
        else:
            st.info("Confusion matrix data not available.")

        # ── Grad-CAM Gallery ───────────────────────────────────────────────
        st.subheader("🔥 Grad-CAM Visualisations")
        gradcam_images = (
            sorted(GRADCAM_DIR.glob("*.png")) if GRADCAM_DIR.exists() else []
        )
        if gradcam_images:
            st.markdown(
                "Heatmap overlays showing which leaf regions activated the model "
                "for each prediction."
            )
            for i in range(0, len(gradcam_images), 3):
                row_imgs = gradcam_images[i : i + 3]
                cols = st.columns(3)
                for col, img_path in zip(cols, row_imgs):
                    with col:
                        st.image(str(img_path), width="stretch")
        else:
            st.info(
                "No Grad-CAM images found. Run "
                "`python src/test.py --gradcam-samples 8`."
            )

        # ── Training setup recap ───────────────────────────────────────────
        with st.expander("⚙️ Training Configuration"):
            st.markdown("""
| Phase | Description | LR | Epochs |
|---|---|---|---|
| 1 — Transfer Learning | Backbone frozen; train classifier head | 1e-3 | 2 |
| 2 — Fine-tuning | Unfreeze last 2 conv blocks | 1e-4 | 3 |

**Augmentation:** Random flips, ±15° rotation, brightness/contrast jitter  
**Batch size:** 32 | **Optimizer:** Adam | **Loss:** Cross-entropy
""")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

with tab_inference:

    if session is None:
        st.error(
            f"ONNX model not found at `{MODEL_PATH}`. "
            "Run `python src/test.py` first, or check the Results tab for findings."
        )
    else:
        # ── Two-column layout ───────────────────────────────────────────
        left_col, right_col = st.columns([1, 2])

        with left_col:
            # ── Image source: upload or sample ──────────────────────────
            uploaded_file = st.file_uploader(
                "Upload a leaf image — supports pepper, potato, and tomato.",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="inference_uploader",
            )

            selected_sample = None
            sample_images = (
                sorted(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
            )

            if sample_images:
                with st.expander(
                    "📁 Or try a sample image", expanded=uploaded_file is None
                ):
                    cols = st.columns(2)
                    for idx, img_path in enumerate(sample_images):
                        col = cols[idx % 2]
                        label = img_path.stem.replace("_", " ").title()
                        with col:
                            if st.button(
                                label,
                                key=f"sample_{idx}",
                                use_container_width=True,
                            ):
                                selected_sample = img_path

        with right_col:
            # ── Determine image to classify ─────────────────────────────
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                source_label = "Uploaded Leaf"
            elif selected_sample is not None:
                image = Image.open(selected_sample).convert("RGB")
                source_label = (
                    f"Sample: {selected_sample.stem.replace('_', ' ').title()}"
                )

            if image is not None:
                img_col, pred_col = st.columns([1, 1])

                with img_col:
                    st.image(image, caption=source_label, width=400)

                with pred_col:
                    with st.spinner("Analysing leaf..."):
                        results, probs = predict(image, session, class_names)

                    st.subheader("Top Predictions")
                    for i, r in enumerate(results[:3], 1):
                        label = r["class"].replace("_", " ").title()
                        confidence = r["confidence"]
                        row = st.columns([2, 3])
                        row[0].markdown(f"**{i}. {label}**")
                        row[1].progress(confidence, text=f"{confidence:.1%}")

                # Grad-CAM for the uploaded image
                st.divider()
                st.subheader("🔍 Model Focus (Grad-CAM)")

                pt_model = load_pytorch_model()
                gradcam_fig = generate_gradcam_overlay(image, pt_model, class_names)
                st.pyplot(gradcam_fig)

# ── Footer ─────────────────────────────────────────────────────────────────
footer_acc = f"{metrics['accuracy'] * 100:.2f}%" if metrics else ""
st.divider()
st.caption(
    "Built with MobileNetV2 + ONNX Runtime | PlantVillage Dataset"
)
