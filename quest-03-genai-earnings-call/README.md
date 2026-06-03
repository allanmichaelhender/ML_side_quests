# 📊 Quest 03 — Energy Earnings Call Analyst

Fine-tune **TinyLlama 1.1B** with **LoRA** on SEC filing data from major energy companies to create an AI earnings call analyst.

## Problem

Investors and analysts need to quickly extract key financial metrics from dense SEC filings. Manually reading 10-Ks, 10-Qs, and 8-Ks across dozens of companies is time-consuming. This quest fine-tunes a small LLM to answer questions about energy company financials directly.

## Model & Approach

| Component          | Choice              | Rationale                                         |
| ------------------ | ------------------- | ------------------------------------------------- |
| **Base model**     | TinyLlama 1.1B Chat | Smallest viable instruction-tuned LLM             |
| **Fine-tuning**    | LoRA (r=4)          | Only ~1.1M trainable params (0.1% of total)       |
| **Target modules** | `q_proj`, `v_proj`  | Attention projections — most effective for LoRA   |
| **Data source**    | SEC EDGAR XBRL      | Structured financial filings for energy companies |
| **Format**         | Instruction Q&A     | Natural language questions → structured answers   |

## Companies Tracked

| Ticker | Company                   | Sector                   |
| ------ | ------------------------- | ------------------------ |
| XOM    | Exxon Mobil               | Integrated Oil & Gas     |
| CVX    | Chevron                   | Integrated Oil & Gas     |
| COP    | ConocoPhillips            | Exploration & Production |
| EOG    | EOG Resources             | Exploration & Production |
| PXD    | Pioneer Natural Resources | Exploration & Production |
| OXY    | Occidental Petroleum      | Exploration & Production |
| SLB    | Schlumberger              | Oilfield Services        |
| BKR    | Baker Hughes              | Oilfield Services        |

## Training

### Quick Start (Fast — ~2-3 hours on Ryzen 5600X)

```bash
cd quest-03-genai-earnings-call

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Train (fast config)
python src/train.py
```

### Fast-Training Configuration

The training is aggressively optimised for CPU:

| Parameter     | Value            | Why                                         |
| ------------- | ---------------- | ------------------------------------------- |
| `max_samples` | 500              | Keeps dataset small but representative      |
| `max_length`  | 512              | Shorter sequences → faster compute          |
| `lora_r`      | 4                | Minimal LoRA rank → fewest trainable params |
| `num_epochs`  | 2                | Enough for domain adaptation                |
| `batch_size`  | 1 + grad accum 4 | Memory efficient, effective batch = 4       |
| `cpu_threads` | 12               | Utilises all Ryzen 5600X threads            |

**Expected time: ~1-1.5h per epoch → ~2-3 hours total**

### Training Details

- **Loss**: Causal language modelling (next-token prediction)
- **Optimizer**: AdamW with cosine LR schedule
- **Warmup**: 20 steps
- **Learning rate**: 3e-4 (standard for LoRA)
- **Evaluation metric**: Perplexity

## Dataset

The pipeline:

1. Fetches XBRL financial data via `edgar-sec` (or uses realistic mock data if SEC API is unavailable)
2. Creates instruction-response pairs: _"What was Exxon Mobil's revenue for FY2024?"_ → _"For FY2024, Exxon Mobil (XOM) reported revenue of $344.6B."_
3. Generates multi-metric summary questions: _"Summarize the key financial results for Chevron for FY2024."_
4. Formats everything in TinyLlama's ChatML style

## Evaluation

```bash
python src/evaluate.py
```

Generates responses to test prompts and reports perplexity.

## Demo

```bash
streamlit run app.py
```

Opens a chat interface where you can ask questions about energy company financials.

## Results

_To be filled in after training._

## Key Findings

_To be filled in after training._
