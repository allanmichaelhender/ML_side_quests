# ML Side Quests

A collection of industry-focused ML projects demonstrating breadth across **computer vision**, **NLP**, **generative AI**, **retrieval-augmented generation**, and **reinforcement learning**.

Each quest is independently containerised with Docker and has its own Streamlit demo.

---

## Quests

| #   | Quest                                                                | Domain          | Model                                      | Key Result                                                      | Demo                                                                                             |
| --- | -------------------------------------------------------------------- | --------------- | ------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | 🌱 [**Plant Disease Classification**](quest-01-cv-plant-disease/)    | Computer Vision | MobileNetV2 (ONNX)                         | **96.1% accuracy** (38 classes)                                 | [streamlit](https://ml-sidequests-01.streamlit.app) · [GitHub](quest-01-cv-plant-disease/)       |
| 2   | 📝 [**Sentiment Analysis**](quest-02-nlp-sentiment-analysis/)        | NLP             | DistilBERT                                 | **94.4% accuracy** · **0.94 F1** (20k samples)                  | [streamlit](https://ml-sidequests-02.streamlit.app) · [GitHub](quest-02-nlp-sentiment-analysis/) |
| 3   | 📊 [**Energy Earnings Call Analyst**](quest-03-genai-earnings-call/) | GenAI           | TinyLlama 1.1B (LoRA, r=4)                 | **Train loss 1.98** · instruction-tuned QA                      | [GitHub](quest-03-genai-earnings-call/)                                                          |
| 4   | 🔍 [**RAG Retrieval Pipeline**](quest-04-rag-pipeline/)              | RAG             | MiniLM + Cross-Encoder + FAISS             | **Hit@5 74.5%** · **MRR 0.57** · 18.8k docs                     | [GitHub](quest-04-rag-pipeline/)                                                                 |
| 5   | 🎫 [**Support Ticket Routing**](quest-05-nlp-ticket-routing/)        | NLP             | TF-IDF / DistilBERT / DeepSeek (zero-shot) | **83.6% acc** (TF-IDF) · 77-class Banking77                     | [GitHub](quest-05-nlp-ticket-routing/)                                                           |
| 6   | ⚡ [**Energy Grid Load Balancing**](quest-06-rl-grid-balancing/)     | RL              | PPO (Stable Baselines3)                    | **-29.9M reward** · **91.7% reliability** · beats all baselines | [streamlit](https://ml-sidequests-06.streamlit.app) · [GitHub](quest-06-rl-grid-balancing/)      |

> Click into any quest folder for the full walkthrough: problem statement, pipeline steps, training instructions, results, and key findings.

---

## Directory Structure

```
ml_side_quests/
├── quest-01-cv-plant-disease/       # Computer Vision
├── quest-02-nlp-sentiment-analysis/ # NLP: Sentiment
├── quest-03-genai-earnings-call/    # Generative AI
├── quest-04-rag-pipeline/           # RAG
├── quest-05-nlp-ticket-routing/     # NLP: Ticket Routing
├── quest-06-rl-grid-balancing/      # Reinforcement Learning
├── vercel_frontend/                 # Portfolio landing page
├── docker-compose.yml               # Orchestrate all 6 demos
└── ml_side_quests.md                # Original detailed notes (kept for reference)
```

Each quest follows a consistent structure:

```
quest-name/
├── Dockerfile
├── README.md              — Problem, approach, results, findings
├── requirements.txt
├── app.py                 — Streamlit demo
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── data_utils.py
├── data/
│   ├── download.py
│   └── sample/            — Small sample committed to repo
└── results/
    ├── figures/
    ├── metrics.json
    └── model.*
```

---

## Getting Started

### Option A — Docker (recommended)

```bash
# Run all Streamlit demos simultaneously
docker compose up -d
# → http://localhost:8501  (CV)
# → http://localhost:8502  (NLP — Sentiment)
# → http://localhost:8503  (GenAI — Earnings)
# → http://localhost:8504  (RAG)
# → http://localhost:8505  (NLP — Tickets)
# → http://localhost:8506  (RL)

# Train inside a container (results persist to host)
docker compose run --rm quest-01-cv python src/train.py
```

### Option B — Local venv

Each quest can be run locally. Example:

```bash
cd quest-06-rl-grid-balancing
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

> See each quest's README for exact commands.

---

## Tech Stack

| Area                       | Libraries                                                 |
| -------------------------- | --------------------------------------------------------- |
| **Deep Learning**          | PyTorch, torchvision, Transformers                        |
| **NLP / GenAI**            | Hugging Face (datasets, PEFT, TRL), Sentence-Transformers |
| **RAG**                    | FAISS, ChromaDB, RAGAS                                    |
| **Reinforcement Learning** | Stable Baselines3, Gymnasium                              |
| **Classical ML**           | scikit-learn, TF-IDF, Logistic Regression                 |
| **Visualisation**          | Matplotlib, Streamlit, Grad-CAM                           |
| **Infrastructure**         | Docker, Docker Compose, ONNX                              |

---

## Hardware Summary

All quests are designed to train and run on CPU (no GPU required).

| Quest                 | Actual CPU Training Time            | Key Metric                            |
| --------------------- | ----------------------------------- | ------------------------------------- |
| 1 — Plant Disease CV  | ~2 hours                            | **96.1%** accuracy                    |
| 2 — Sentiment BERT    | **~25 min** (3 epochs, 5k samples)  | **84.2%** accuracy, **0.84** macro F1 |
| 3 — Earnings Call LLM | **~63 min** (2 epochs, 500 samples) | **1.98** train loss (LoRA r=4)        |
| 4 — RAG Pipeline      | ~30 min (indexing only)             | **74.5%** Hit@5, **0.57** MRR         |
| 5 — Ticket Routing    | **< 30 min** per model              | **85.2%** accuracy (DistilBERT)       |
| 6 — Grid RL Agent     | **30–60 min**                       | **-30.0M** reward (vs -414M random)   |

---

## Portfolio

A landing page showcasing all six quests is deployed at **[ml-sidequests.vercel.app](https://ml-sidequests.vercel.app)** (built with React + Vite, hosted on Vercel).

---

## License

This project is for educational and portfolio purposes.
