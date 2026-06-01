# Quest 1 — Computer Vision: Plant Disease Classification

Classify plant leaf images into 38 disease/healthy categories using transfer learning with MobileNetV2.

## Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) via Kaggle
- **Size**: 54,000+ labelled leaf images
- **Classes**: 38 classes covering 14 crop species (tomato, potato, corn, grape, apple, etc.)
- **Split**: 70% train / 15% validation / 15% test (random split with fixed seed 42)
  - **Train** — used for backpropagation during transfer learning and fine-tuning
  - **Validation** — used for per-epoch monitoring (loss and accuracy); influences hyperparameter decisions
  - **Test** — fully held out, never seen during any phase of training; used only once in `src/test.py` for the final unbiased evaluation

## Approach

| Phase                 | Description                                                | Learning Rate |
| --------------------- | ---------------------------------------------------------- | ------------- |
| 1 — Transfer Learning | MobileNetV2 backbone frozen; train new classification head | 1e-3          |
| 2 — Fine-tuning       | Unfreeze last 2 conv blocks; train end-to-end              | 1e-4          |

- **Architecture**: MobileNetV2 pretrained on ImageNet
- **Choice rationale**: Depthwise separable convolutions make it CPU-friendly
- **Augmentation**: Random flips, ±15° rotation, brightness/contrast jitter

## Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 96.06% |
| Macro F1      | 0.9592 |
| Weighted F1   | 0.9607 |

Per-class metrics, confusion matrix, and Grad-CAM visualisations are available in `results/` after running `src/test.py`.

## Usage

### Option A — Docker (recommended)

```bash
# Build and run training
docker build -t quest-01-cv .
docker run --rm -v "$(pwd)/results:/app/results" quest-01-cv

# Download data first, then train
docker run --rm -v "$(pwd)/data:/app/data" quest-01-cv python src/download_data.py
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" quest-01-cv

# Evaluate on held-out test set
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" quest-01-cv \
  python src/test.py --gradcam-samples 8

# Launch Streamlit demo
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" quest-01-cv \
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
python src/download_data.py
python src/train.py
python src/test.py --gradcam-samples 8
streamlit run app.py
```

## Files

```
quest-01-cv-plant-disease/
├── Dockerfile
├── .dockerignore
├── README.md
├── requirements.txt
├── app.py                     # Streamlit demo
├── data/
│   └── plantvillage_raw/      # Raw class folders (downloaded)
├── src/
│   ├── data_utils.py          # Shared utilities (find_class_root, TransformedSubset)
│   ├── download_data.py       # Download PlantVillage from Kaggle
│   ├── train.py               # Transfer learning + fine-tuning
│   └── test.py                # Evaluation on held-out test set (Grad-CAM, ONNX, metrics)
└── results/
    ├── model.pt               # PyTorch checkpoint
    ├── model.onnx             # ONNX export
    ├── class_names.json       # Class index-to-name mapping
    ├── metrics.json           # Accuracy, per-class precision/recall/F1
    ├── test_indices.json      # Indices of the held-out test split
    ├── figures/
    │   └── confusion_matrix.png
    └── gradcam/               # Grad-CAM overlay visualisations
```

## Key Libraries

- `torch` / `torchvision` — MobileNetV2 model and data pipeline
- `onnx` / `onnxruntime` — Lightweight CPU inference
- `streamlit` — Demo interface
- `scikit-learn` — Evaluation metrics (classification report, confusion matrix)
- `opencv-python` — Grad-CAM overlay rendering
- `matplotlib` — Confusion matrix and Grad-CAM figure generation
- `kaggle` — Dataset download via Kaggle API
- `tqdm` — Download progress bar
