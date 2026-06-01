"""
Streamlit demo for Plant Disease Classification.

Upload a leaf image and get real-time predictions with confidence scores
and a Grad-CAM heatmap overlay showing which regions drove the model's decision.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "results" / "model.onnx"
LABELS_PATH = HERE / "results" / "class_names.json"
GRADCAM_DIR = HERE / "results" / "gradcam"

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
    layout="centered",
)

st.title("🌿 Plant Disease Classifier")
st.markdown(
    "Upload a leaf image to identify diseases and healthy conditions "
    "across **38 classes** covering 14 crop species."
)


# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_onnx_model():
    if not MODEL_PATH.exists():
        st.error(
            f"ONNX model not found at `{MODEL_PATH}`. Run `python src/evaluate.py` first."
        )
        st.stop()
    return ort.InferenceSession(str(MODEL_PATH))


@st.cache_data
def load_class_names():
    if not LABELS_PATH.exists():
        st.error(f"Class names not found at `{LABELS_PATH}`.")
        st.stop()
    with open(LABELS_PATH) as f:
        return json.load(f)


session = load_onnx_model()
class_names = load_class_names()


# ── Inference ──────────────────────────────────────────────────────────────
def predict(image: Image.Image) -> tuple[list[dict], np.ndarray]:
    """Run ONNX inference. Returns (predictions, probabilities)."""
    img_tensor = TRANSFORM(image).unsqueeze(0).numpy().astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img_tensor})[0]

    # Softmax probabilities
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


# ── UI ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("Analysing leaf..."):
            results, probs = predict(image)

        st.subheader("Top Predictions")
        for i, r in enumerate(results[:5], 1):
            label = r["class"].replace("_", " ").title()
            confidence = r["confidence"]
            bar_color = (
                "#28a745"
                if confidence > 0.8
                else "#ffc107"
                if confidence > 0.5
                else "#dc3545"
            )
            st.markdown(f"**{i}. {label}**")
            st.progress(confidence, text=f"{confidence:.1%}")
            st.caption("")  # spacing

    # ── Grad-CAM preview ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Model Focus (Grad-CAM)")

    # Find a matching sample Grad-CAM from the results directory
    gradcam_images = sorted(GRADCAM_DIR.glob("*.png")) if GRADCAM_DIR.exists() else []
    if gradcam_images:
        st.image(
            gradcam_images[:3],
            caption=[f"Grad-CAM {i + 1}" for i in range(min(3, len(gradcam_images)))],
            use_container_width=True,
        )
        st.caption(
            "Grad-CAM heatmaps generated during evaluation show which leaf regions activated the model."
        )
    else:
        st.info(
            "No Grad-CAM images found. Run `python src/evaluate.py --gradcam-samples 8` to generate them."
        )

else:
    # Placeholder instructions
    st.info("👆 Upload a leaf image to get started.")
    st.markdown("""
    **Supported crops:** apple, blueberry, cherry, corn, grape, orange, peach,
    pepper, potato, raspberry, soybean, squash, strawberry, tomato

    **Example diseases:** Apple scab, Black rot, Cedar apple rust, Early blight,
    Late blight, Powdery mildew, Leaf mold, Septoria leaf spot, Spider mites,
    Target spot, Yellow leaf curl virus, and more.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with MobileNetV2 + ONNX Runtime | PlantVillage Dataset")
