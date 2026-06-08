# ML Side Quests — Detailed Notes

## Quest Overview & Results

| #   | Quest                           | Model                           | Training Time          | Top Result                                                      |
| --- | ------------------------------- | ------------------------------- | ---------------------- | --------------------------------------------------------------- |
| 1   | 🌱 Plant Disease Classification | MobileNetV2 (ONNX)              | ~2 hours CPU           | **96.1%** accuracy across 38 plant disease classes              |
| 2   | 📝 Sentiment Analysis           | DistilBERT                      | **~25 min** (3 epochs) | **84.2%** accuracy, **0.84** macro F1 on 5k reviews             |
| 3   | RAG Retrieval Pipeline          | MiniLM + Cross-Encoder          | ~30 min (indexing)     | **74.5%** Hit@5, **0.57** MRR over 200 queries                  |
| 5   | 🎫 Support Ticket Routing       | TF-IDF / DistilBERT / Zero-shot | **< 30 min** per model | **85.2%** accuracy (DistilBERT) across 77 intent classes        |
| 6   | ⚡ Energy Grid Load Balancing   | PPO (Stable Baselines3)         | **30–60 min**          | **-30.0M** reward (beats random -414M and rule-based baselines) |

---

## Quest 03 — Architecture Deep-Dive

The GenAI quest uses **LoRA fine-tuning** on TinyLlama 1.1B:

```
TinyLlama 1.1B (frozen)
  ├── Layer 1  →  + LoRA adapter (trainable)
  ├── Layer 2  →  + LoRA adapter (trainable)
  ├── ...
  └── Layer 22 →  + LoRA adapter (trainable)
```

- **~1.1B parameters frozen** (original TinyLlama weights)
- **~4 MB trainable** (LoRA adapters — only 0.1% of total)
- Adapter saved to `results/model/` — ready for Hugging Face Hub upload

### Prompt Format

**Training** (full sequence — model learns to predict response tokens):

```
<|system|>
You are a financial analyst...</s>
<|user|>
What was XOM's revenue for FY2024?

[filing text excerpt]</s>
<|assistant|>
For FY2024, Exxon Mobil (XOM) reported revenue of $344.58B.</s>
```

**Inference** (model generates from `<|assistant|>` onward):

```
<|system|>
You are a financial analyst...</s>
<|user|>
What was XOM's revenue for FY2024?

[filing text excerpt]</s>
<|assistant|>
```

The model learns to **extract** the metric from the provided filing text context, rather than memorising a lookup table.

---

## Repository Structure

```
ml_side_quests/
├── quest-01-cv-plant-disease/       # Computer Vision — MobileNetV2
├── quest-02-nlp-sentiment-analysis/ # NLP — DistilBERT fine-tuning
├── quest-04-rag-pipeline/           # RAG — MiniLM + FAISS
├── quest-05-nlp-ticket-routing/     # NLP — Multi-model comparison
├── quest-06-rl-grid-balancing/      # RL — PPO energy grid agent
├── vercel_frontend/                 # Portfolio landing page (React + Vite)
├── docker-compose.yml               # Orchestrate all demos
└── README.md                        # Main entry point
```

Each quest follows a consistent layout:

```
quest-name/
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py              — Streamlit demo
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── data_utils.py
├── data/
│   ├── cache/          — Downloaded/cached datasets
│   └── sample/         — Small sample committed to repo
└── results/
    ├── figures/
    ├── metrics.json
    ├── instruction_pairs.json  (quest 03)
    └── model/
```

---

## Running Locally

```bash
# All quests via Docker
docker compose up -d

# Or individual quest in a venv
cd quest-04-rag-pipeline
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/embed_and_index.py
streamlit run app.py     # view results
```

All quests designed for **CPU-only** training and inference.
