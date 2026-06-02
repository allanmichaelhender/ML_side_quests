# ML Side Quests

[![Vercel](https://img.shields.io/badge/Portfolio-vercel_frontend-818cf8?style=flat&logo=vercel)](https://ml-sidequests.vercel.app)

A collection of industry-focused ML projects demonstrating breadth across **computer vision**, **NLP**, **generative AI**, **retrieval-augmented generation**, and **reinforcement learning**.

Each quest is independently containerised with Docker and has its own Streamlit demo.

---

## Quests

| #   | Quest                               | Domain          | Model                               | Demo                                                |
| --- | ----------------------------------- | --------------- | ----------------------------------- | --------------------------------------------------- |
| 1   | 🌱 **Plant Disease Classification** | Computer Vision | MobileNetV2 (ONNX)                  | [streamlit](https://ml-sidequests-01.streamlit.app) |
| 2   | 📝 **Sentiment Analysis**           | NLP             | DistilBERT                          | [streamlit](https://ml-sidequests-02.streamlit.app) |
| 3   | 📊 **Energy Earnings Call Analyst** | GenAI           | TinyLlama / Phi-2 (LoRA)            | —                                                   |
| 4   | 🔍 **RAG Retrieval Pipeline**       | RAG             | MiniLM + Cross-Encoder              | —                                                   |
| 5   | 🎫 **Support Ticket Routing**       | NLP             | TF-IDF / DistilBERT / Zero-shot LLM | —                                                   |
| 6   | ⚡ **Energy Grid Load Balancing**   | RL              | PPO (Stable Baselines3)             | [streamlit](https://ml-sidequests-06.streamlit.app) |

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

| Quest                 | Estimated CPU Training Time    |
| --------------------- | ------------------------------ |
| 1 — Plant Disease CV  | 1–3 hours                      |
| 2 — Sentiment BERT    | 1–2 hours                      |
| 3 — Earnings Call LLM | 6–10 hours (LoRA)              |
| 4 — RAG Pipeline      | No training — indexing ~30 min |
| 5 — Ticket Routing    | < 30 minutes                   |
| 6 — Grid RL Agent     | 30–60 minutes                  |

---

## Portfolio

A landing page showcasing all six quests is deployed at **[ml-sidequests.vercel.app](https://ml-sidequests.vercel.app)** (built with React + Vite, hosted on Vercel).

---

## License

This project is for educational and portfolio purposes.
