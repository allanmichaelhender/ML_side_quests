"""
Streamlit app for Plant Disease Classification.

Tabs:
  - Results Dashboard — view training/evaluation findings
  - Live Inference   — upload a leaf image for real-time prediction
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
MODEL_PATH = RESULTS_DIR / "model.onnx"
LABELS_PATH = RESULTS_DIR / "class_names.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CM_PATH = RESULTS_DIR / "figures" / "confusion_matrix.png"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

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
            "No results found. Run `python src/train.py` then `python src/evaluate.py` first."
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
            st.image(str(CM_PATH), width='stretch')
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
                st.dataframe(rows_data, width='stretch', hide_index=True)
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
                        st.image(str(img_path), width='stretch')
        else:
            st.info(
                "No Grad-CAM images found. Run "
                "`python src/evaluate.py --gradcam-samples 8`."
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
    st.markdown(
        "Upload a leaf image to identify diseases and healthy conditions "
        "across **38 classes** covering 14 crop species."
    )

    if session is None:
        st.error(
            f"ONNX model not found at `{MODEL_PATH}`. "
            "Run `python src/evaluate.py` first, or check the Results tab for findings."
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="inference_uploader",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Leaf", width='stretch')

            with col2:
                with st.spinner("Analysing leaf..."):
                    results, probs = predict(image, session, class_names)

                st.subheader("Top Predictions")
                for i, r in enumerate(results[:5], 1):
                    label = r["class"].replace("_", " ").title()
                    confidence = r["confidence"]
                    st.markdown(f"**{i}. {label}**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                    st.caption("")

            # Grad-CAM reference
            st.divider()
            st.subheader("🔍 Model Focus (Grad-CAM)")

            gradcam_images = (
                sorted(GRADCAM_DIR.glob("*.png")) if GRADCAM_DIR.exists() else []
            )
            if gradcam_images:
                st.markdown(
                    "Reference Grad-CAM heatmaps from evaluation "
                    "(val-set samples, not the uploaded image):"
                )
                for i in range(0, min(len(gradcam_images), 3), 3):
                    row_imgs = gradcam_images[i : i + 3]
                    cols = st.columns(3)
                    for col, img_path in zip(cols, row_imgs):
                        with col:
                            st.image(str(img_path), width='stretch')
            else:
                st.info(
                    "No Grad-CAM images found. Run "
                    "`python src/evaluate.py --gradcam-samples 8`."
                )

        else:
            st.info("👆 Upload a leaf image to get started.")
            st.markdown("""
**Supported crops:** apple, blueberry, cherry, corn, grape, orange, peach,
pepper, potato, raspberry, soybean, squash, strawberry, tomato

**Example diseases:** Apple scab, Black rot, Cedar apple rust, Early blight,
Late blight, Powdery mildew, Leaf mold, Septoria leaf spot, Spider mites,
Target spot, Yellow leaf curl virus, and more.
""")

# ── Footer ─────────────────────────────────────────────────────────────────
footer_acc = f"{metrics['accuracy'] * 100:.2f}%" if metrics else ""
st.divider()
st.caption(
    f"Built with MobileNetV2 + ONNX Runtime | PlantVillage Dataset | "
    f"Accuracy: {footer_acc}"
    if metrics
    else "Built with MobileNetV2 + ONNX Runtime | PlantVillage Dataset"
)


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
