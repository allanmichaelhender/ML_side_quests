# Quest 1 — Computer Vision: Plant Disease Classification

Classify plant leaf images into 38 disease/healthy categories using transfer learning with MobileNetV2.

## Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) via Kaggle
- **Size**: 54,000+ labelled leaf images
- **Classes**: 38 classes covering 14 crop species (tomato, potato, corn, grape, apple, etc.)
- **Split**: Pre-split into train/validation sets

## Approach

| Phase                 | Description                                                | Learning Rate |
| --------------------- | ---------------------------------------------------------- | ------------- |
| 1 — Transfer Learning | MobileNetV2 backbone frozen; train new classification head | 1e-3          |
| 2 — Fine-tuning       | Unfreeze last 2 conv blocks; train end-to-end              | 1e-4          |

- **Architecture**: MobileNetV2 pretrained on ImageNet
- **Choice rationale**: Depthwise separable convolutions make it CPU-friendly
- **Augmentation**: Random flips, ±15° rotation, brightness/contrast jitter

## Results

_Results to be populated after training._

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | —     |
| Macro F1            | —     |

## Usage

### Option A — Docker (recommended)

```bash
# Build and run training
docker build -t quest-01-cv .
docker run --rm -v "$(pwd)/results:/app/results" quest-01-cv

# Download data first, then train
docker run --rm -v "$(pwd)/data:/app/data" quest-01-cv python data/download.py
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" quest-01-cv

# Launch Streamlit demo
docker run --rm -p 8501:8501 -v "$(pwd)/results:/app/results" quest-01-cv \
  streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Or use docker compose from the root
cd ..
docker compose up quest-01-cv
```

### Option B — Local venv

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python data/download.py
python src/train.py
python src/evaluate.py --gradcam-samples 8
python src/predict.py path/to/leaf.jpg
streamlit run app.py
```

## Files

```
quest-01-cv-plant-disease/
├── Dockerfile
├── .dockerignore
├── README.md
├── requirements.txt
├── app.py                  # Streamlit demo
├── data/
│   ├── download.py         # Download from Kaggle
│   └── sample/             # Small test subset
├── src/
│   ├── train.py            # Transfer learning + fine-tuning
│   ├── evaluate.py         # Metrics, Grad-CAM, ONNX export
│   └── predict.py          # Single-image inference
└── results/
    ├── model.pt            # PyTorch checkpoint
    ├── model.onnx          # ONNX export
    ├── class_names.json
    ├── metrics.json
    └── figures/            # Confusion matrix, Grad-CAM images
```

## Key Libraries

- `torch` / `torchvision` — MobileNetV2 model and data pipeline
- `onnx` / `onnxruntime` — Lightweight CPU inference
- `streamlit` — Demo interface
- `scikit-learn` — Evaluation metrics
- `opencv-python` — Grad-CAM overlay rendering
