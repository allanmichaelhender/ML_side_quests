"""
Inference on a single leaf image using the trained ONNX model.

Usage:
    python src/predict.py path/to/leaf.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MODEL = PROJECT / "results" / "model.onnx"
DEFAULT_LABELS = PROJECT / "results" / "class_names.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_labels(path: Path) -> list[str]:
    with open(path) as f:
        return json.load(f)


def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img)
    # Add batch dimension and convert to numpy
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def predict(
    image_path: Path,
    onnx_path: Path = DEFAULT_MODEL,
    labels_path: Path = DEFAULT_LABELS,
    top_k: int = 3,
):
    """Run inference and return top-k predictions."""
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run `python src/evaluate.py` first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Class names not found at {labels_path}.")

    class_names = load_labels(labels_path)
    input_data = preprocess(image_path)

    # ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: input_data})[0]  # [1, 38]
    probabilities = 1.0 / (1.0 + np.exp(-outputs))  # sigmoid → probs
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    top_indices = np.argsort(probabilities[0])[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {
                "class": class_names[idx],
                "class_id": int(idx),
                "confidence": float(probabilities[0, idx]),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease from image")
    parser.add_argument("image_path", type=str, help="Path to leaf image")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    results = predict(image_path, Path(args.model), Path(args.labels), top_k=args.top_k)

    print(f"\nPredictions for {image_path.name}:")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class']:40s}  {r['confidence']:.2%}")


if __name__ == "__main__":
    main()
