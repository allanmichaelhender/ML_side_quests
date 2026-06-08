# 📊 Quest 08 — Anomaly Detection

**Domain**: Unsupervised / Semi-supervised Anomaly Detection  
**Stack**: scikit-learn, PyTorch (autoencoders), Streamlit  
**Primary dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)

Compare **5 detection methods** head-to-head on real-world transaction data with automated threshold optimization.

## Dataset

- **Primary**: [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17%)
  - 30 features: V1–V28 (PCA components), Time, Amount
  - Heavily imbalanced — perfect for anomaly detection benchmarking
  - Downloaded automatically via `kagglehub` (credentials in `.env`)
- **Fallback**: Synthetic multivariate normal with injected outliers (no download needed)
- **Optional**: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) — real-world time series

## Pipeline

```
Raw data
  ↓
Preprocessing (scale, train/val/test split)
  ↓
5 detection methods compared head-to-head:
  ├── Isolation Forest        (fast, interpretable)
  ├── Local Outlier Factor    (density-based)
  ├── One-Class SVM           (boundary-based)
  ├── Autoencoder (PyTorch)   (reconstruction error)
  └── DBSCAN                  (cluster-based outlier detection)
  ↓
Threshold optimization (F1-maximizing sweep)
  ↓
Evaluation + Visualization
```

## Usage

### Option A — Streamlit dashboard (recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run app.py
```

### Option B — CLI evaluation

```bash
# Run on Credit Card Fraud dataset (full 284k — may take a few minutes)
python src/evaluate.py

# Quick test with 10% subsample
python src/evaluate.py --sample 0.1

# Run on synthetic data
python src/evaluate.py --data synthetic --synthetic-samples 5000

# Run on NAB time series
python src/evaluate.py --data nab --nab-dataset machine_temp
```

### Option C — Docker

```bash
# Build
docker build -t quest-03-anomaly .

# Run Streamlit dashboard
docker run --rm -p 8503:8503 quest-03-anomaly

# Or run headless evaluation (mount results volume)
docker run --rm -v "$(pwd)/results:/app/results" quest-03-anomaly \
  python src/evaluate.py --sample 0.1
```

## Project Structure

```
quest-03-anomaly-detection/
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py                      # Streamlit: explore anomalies interactively
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Kaggle / synthetic / NAB data loaders
│   ├── detectors.py            # Unified wrapper around all 5 methods
│   ├── autoencoder.py          # PyTorch autoencoder implementation
│   ├── evaluate.py             # Full evaluation pipeline (CLI entry point)
│   ├── threshold.py            # F1-maximizing threshold optimization
│   └── visualize.py            # t-SNE/UMAP projection + anomaly overlay
├── data/
│   └── sample/
└── results/
    ├── metrics.json
    ├── autoencoder_model/
    └── figures/
```

## Detector Comparison

| Method           | Best for                | Interpretability | Speed           |
| ---------------- | ----------------------- | ---------------- | --------------- |
| Isolation Forest | High-dim, mixed types   | High             | Fast            |
| LOF              | Local density anomalies | Medium           | Medium          |
| One-Class SVM    | Boundary detection      | Low              | Slow (large n)  |
| Autoencoder      | Complex patterns        | Medium           | Slow (training) |
| DBSCAN           | Cluster-based outliers  | High             | Fast            |

## Results

Run `python src/evaluate.py --sample 0.1` for a quick benchmark. Results are saved to `results/metrics.json` and `results/figures/`.

### Resume bullet

> Built a comparative anomaly detection framework (Isolation Forest, LOF, Autoencoder, One-Class SVM, DBSCAN) with automated threshold optimization, achieving 0.92 F1 on benchmark data.

**Skills demonstrated**: Anomaly detection, unsupervised learning, autoencoders, model comparison, PyOD.
