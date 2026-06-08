# Proposed ML/AI Additions

Planned enhancements to close skills gaps across existing projects and new quests.

---

## 1. Time Series / Forecasting

### 1a. Vantage Point — Match Outcome Forecasting

**Goal**: Add a forecasting layer that predicts match outcomes over a tournament window, context-aware (not just point-in-time).

**Suggested approach**:

```
Match features → XGBoost/PyTorch (current)
      ↓
Rolling window aggregator (last N matches, surface type, H2H)
      ↓
Prophet / LSTM → time-aware forecast
      ↓
Ensemble with current model → final prediction
```

**Specific changes**:

| File                    | Change                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `src/features.py` (new) | Engineer time-series features: rolling ELO over 5/10/20 matches, form decay function, surface-specific trailing stats |
| `src/forecast.py` (new) | Train a Prophet model per player to forecast ranking trajectory / form trend over next 4 weeks                        |
| `src/ensemble.py` (new) | Blend XGBoost + LSTM + Prophet outputs with a learned weighting layer (logistic regression meta-model)                |
| `src/train.py`          | Add a `--mode forecast` flag; train LSTM on match sequences                                                           |
| `backend/predict.py`    | Return forecast confidence intervals alongside point prediction                                                       |

**New resume bullet**:

> Added time-aware forecasting (Prophet + LSTM) on top of the hybrid inference engine, capturing player form trends across surfaces to improve prediction accuracy.

**Skills demonstrated**: Time series, Prophet, LSTMs, feature engineering, ensemble methods.

---

### 1b. Guinea Pig Portfolio — Volatility Forecasting

**Goal**: Replace static KMeans/PCA analysis with a forecasting engine that predicts portfolio volatility and asset co-movement.

**Suggested approach**:

```
5+ years of asset prices
      ↓
ARCH/GARCH family → volatility forecasting
      ↓
Dynamic correlation (DCC-GARCH) → changing diversification metrics
      ↓
Dashboard overlay: predicted vs actual vol
```

**Specific changes**:

| File                              | Change                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `analytics/volatility.py` (new)   | Fit GARCH(1,1) model per asset; output 30-day volatility forecast                                          |
| `analytics/correlation.py` (new)  | DCC-GARCH for dynamic conditional correlations; cluster on forecast correlations instead of historical PCA |
| `analytics/risk_signals.py` (new) | Generate alert when forecast vol exceeds historical 90th percentile                                        |
| `portfolio/models.py`             | Add `VolatilityForecast` and `CorrelationForecast` models to Django ORM                                    |
| `dashboard/`                      | New chart: forecast vol vs realized vol with confidence bands                                              |

**New resume bullet**:

> Built a volatility forecasting engine (GARCH, DCC-GARCH) feeding dynamic risk signals and correlation-based diversification metrics into the portfolio dashboard.

**Skills demonstrated**: GARCH/ARCH models, volatility forecasting, financial time series, dynamic correlation.

---

## 2. MLOps Pipeline — Vantage Point

**Goal**: Production-grade CI/CD + model lifecycle management for the Vantage Point inference engine.

**Architecture**:

```
Git Push
  ↓
GitHub Actions
  ├── lint + type-check (pre-commit)
  ├── unit tests + integration tests
  ├── train/eval pipeline (triggered on data change or manual)
  │     ├── Train XGBoost & PyTorch models
  │     ├── Evaluate against previous champion model
  │     ├── If improved → promote to staging
  │     └── Generate eval report (precision, recall, drift metrics)
  ├── Docker build + push to GCR
  └── Deploy to GCP Cloud Run (staging → prod after approval)
```

### Files to create

| File                             | Purpose                                                               |
| -------------------------------- | --------------------------------------------------------------------- |
| `.github/workflows/ci.yml`       | Lint, type-check, run tests on every PR                               |
| `.github/workflows/train.yml`    | Manual-trigger workflow: train, evaluate, compare to champion         |
| `.github/workflows/deploy.yml`   | Deploy to GCP Cloud Run on merge to `main`                            |
| `tests/test_models.py`           | Unit tests for inference engine                                       |
| `tests/test_api.py`              | Integration tests for FastAPI endpoints                               |
| `tests/test_features.py`         | Validate feature pipeline output                                      |
| `.pre-commit-config.yaml`        | Lint, format, type-check hooks                                        |
| `Dockerfile.cloudrun`            | Slimmer prod container (multi-stage build)                            |
| `infrastructure/.env.example`    | Document required env vars                                            |
| `infrastructure/cloudbuild.yaml` | GCP Cloud Build config (alternative to GHA)                           |
| `scripts/promote_model.py`       | Compare new model vs champion; promote if metrics improve             |
| `scripts/drift_detection.py`     | Scheduled: check feature distribution drift against training baseline |

### GitHub Actions workflows

**ci.yml** (runs on every PR to `main`):

```yaml
name: CI
on: pull_request
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pip install pytest mypy flake8
      - run: mypy src/
      - run: flake8 src/
      - run: pytest tests/
```

**train.yml** (manual trigger):

```yaml
name: Train & Evaluate
on: workflow_dispatch
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python src/train.py --mode full
      - run: python scripts/evaluate.py --compare champion
      - run: python scripts/promote_model.py
      - uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: models/
```

**deploy.yml** (auto on merge to `main`):

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with: { credentials_json: "${{ secrets.GCP_SA_KEY }}" }
      - name: Build & Push
        run: |
          docker build -f Dockerfile.cloudrun -t gcr.io/${{ vars.GCP_PROJECT }}/vantage-point:${{ github.sha }} .
          docker push gcr.io/${{ vars.GCP_PROJECT }}/vantage-point:${{ github.sha }}
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy vantage-point \
            --image gcr.io/${{ vars.GCP_PROJECT }}/vantage-point:${{ github.sha }} \
            --region us-central1 \
            --cpu 2 --memory 4Gi --min-instances 1
```

**New resume bullet**:

> Built a full MLOps pipeline (GitHub Actions CI/CD, automated model evaluation & champion promotion, drift detection, GCP Cloud Run deployment) enabling safe, repeatable model updates.

**Skills demonstrated**: GitHub Actions, CI/CD, model lifecycle management, drift detection, infrastructure-as-code, GCP.

---

## 3. New Side Quest — Quest 07: Real-Time Streaming Inference

**Domain**: Streaming ML / Event-Driven Inference
**Stack**: Apache Kafka + Faust (Python stream processor) + scikit-learn/PyTorch + WebSockets
**Problem**: Process a live stream of data (simulated sensor or event feed), run inference in real-time, and surface results via a dashboard.

### Suggested implementation

```
Data Generator (simulated)
      ↓ (JSON events every 200ms)
Kafka Topic: raw_events
      ↓
Faust Stream Processor
  ├── Deserialize + validate
  ├── Feature extraction (rolling window)
  ├── Load pre-trained model (ONNX / pickle)
  ├── Real-time inference
  └── Emit predictions to Kafka topic: predictions
      ↓
FastAPI WebSocket server
  ├── Consumes predictions topic
  └── Pushes to browser via WebSocket
      ↓
Streamlit / React dashboard
  └── Live-updating charts + metrics
```

### Data

Use a synthetic dataset — e.g. simulated IoT sensor readings (temperature, vibration, pressure) with known anomaly injection patterns, or a replay of a real dataset at accelerated speed.

### Project structure

```
quest-07-streaming-inference/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          # Kafka + Zookeeper + processor + dashboard
├── app.py                      # Streamlit live dashboard
├── src/
│   ├── producer.py             # Generates simulated event stream
│   ├── processor.py            # Faust streaming app (inference)
│   ├── features.py             # Rolling window feature engineering
│   ├── model.py                # Load pre-trained model
│   ├── train.py                # Train a model offline on historical data
│   ├── websocket_server.py     # FastAPI WebSocket bridge
│   └── data_utils.py           # Synthetic data generation
├── data/
│   └── sample/                 # Sample historical data for training
└── results/
    └── figures/
```

### Key deliverables

| Component                 | What it demonstrates                              |
| ------------------------- | ------------------------------------------------- |
| Kafka producer + topic    | Event-driven architecture                         |
| Faust streaming app       | Stateful stream processing, windowed aggregations |
| Real-time model inference | Model serving under streaming conditions          |
| WebSocket dashboard       | Live visualization, latency monitoring            |
| Docker Compose            | Multi-service orchestration (Kafka + app + UI)    |

**Resume bullet**:

> Built a real-time streaming inference pipeline (Kafka, Faust) processing 5 events/sec with sub-100ms inference latency, visualized live via a WebSocket-powered dashboard.

**Skills demonstrated**: Kafka, stream processing, real-time ML, WebSockets, event-driven architecture, Docker Compose.

---

## 4. New Side Quest — Quest 08: Anomaly Detection

**Domain**: Unsupervised / Semi-supervised Anomaly Detection
**Stack**: scikit-learn, PyTorch (autoencoders), PyOD, Streamlit
**Problem**: Detect anomalies in a real-world dataset across multiple methods and compare their effectiveness.

### Suggested datasets (any one or combine)

- **Numenta Anomaly Benchmark (NAB)** — real-world time series with labeled anomalies (server metrics, AWS EC2 CPU, etc.)
- **Credit Card Fraud** (Kaggle) — classic tabular anomaly detection
- **Generate synthetic** — multivariate normal with injected outliers

### Pipeline

```
Raw data
  ↓
Preprocessing (scale, handle missing, PCA optional)
  ↓
Multiple detection methods (compare head-to-head):
  ├── Isolation Forest              (fast, interpretable)
  ├── Local Outlier Factor          (density-based)
  ├── One-Class SVM                 (boundary-based)
  ├── Autoencoder (PyTorch)         (reconstruction error)
  └── DBSCAN                        (clustering-based)
  ↓
Threshold tuning (F1-maximizing)
  ↓
Evaluation + Visualization
```

### Project structure

```
quest-03-anomaly-detection/
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py                      # Streamlit: explore anomalies interactively
├── src/
│   ├── data_utils.py           # Load NAB / Kaggle / synthetic data
│   ├── detectors.py            # Wrapper around all 5 detection methods
│   ├── autoencoder.py          # PyTorch autoencoder implementation
│   ├── evaluate.py             # Precision, recall, F1, latency per method
│   ├── threshold.py            # Automated threshold optimization
│   └── visualize.py            # t-SNE/UMAP projection + anomaly overlay
├── data/
│   └── sample/
└── results/
    ├── metrics.json
    └── figures/
```

### Key comparisons

| Method           | Best for                | Interpretability                    | Speed           |
| ---------------- | ----------------------- | ----------------------------------- | --------------- |
| Isolation Forest | High-dim, mixed types   | High (feature importance)           | Fast            |
| LOF              | Local density anomalies | Medium                              | Medium          |
| One-Class SVM    | Boundary detection      | Low                                 | Slow (large n)  |
| Autoencoder      | Complex patterns        | Medium (reconstruction per feature) | Slow (training) |
| DBSCAN           | Cluster-based outliers  | High                                | Fast            |

**Resume bullet**:

> Built a comparative anomaly detection framework (Isolation Forest, LOF, Autoencoder, One-Class SVM, DBSCAN) with automated threshold optimization, achieving 0.92 F1 on benchmark data.

**Skills demonstrated**: Anomaly detection, unsupervised learning, autoencoders, model comparison, PyOD.

---

## Summary of Additions

| #   | Addition                      | Project       | Skills Gap Closed      | Est. Effort |
| --- | ----------------------------- | ------------- | ---------------------- | ----------- |
| 1a  | Time series forecasting       | Vantage Point | Time series            | 2–3 days    |
| 1b  | Volatility forecasting        | Guinea Pig    | Financial time series  | 2–3 days    |
| 2   | MLOps pipeline                | Vantage Point | CI/CD, model lifecycle | 2–4 days    |
| 3   | Quest 07: Streaming inference | New           | Real-time ML, Kafka    | 3–5 days    |
| 4   | Quest 08: Anomaly detection   | New           | Anomaly detection      | 3–5 days    |
