# Quest 6 — Reinforcement Learning: Data Center Resource Scheduling Agent

Train a **PPO** (Proximal Policy Optimization) agent to allocate machine instances
from multiple types (general-purpose, compute-optimised, memory-optimised, GPU,
storage-optimised) to meet computing workload demand at minimum cost, energy,
and maximum reliability.

## Dataset

- **Workload patterns**: Parameterised by diurnal/weekly cycles observed in
  [Google Cluster Data 2011](https://github.com/google/cluster-data) traces
  (Borg cells, ~12,500 machines, May 2011)
- **Resource demand**: CPU (vCPU cores) and memory (GB) with realistic
  diurnal peaks, weekday/weekend scaling, and stochastic noise
- **Machine types**: Based on real GCP instance families with approximate
  on-demand pricing (general, compute-opt, memory-opt, GPU, storage-opt)
- **Machine availability**: Markov-chain model of failures and repairs,
  mimicking real data center maintenance patterns
- Data is **synthetically generated** using patterns from real Google traces
  (optional: download actual Google Cluster Data via `download_google_cluster_data()`)
- **No API keys required** for synthetic generation

## Approach

| Component        | Description                                                                             |
| ---------------- | --------------------------------------------------------------------------------------- |
| **Algorithm**    | PPO (Proximal Policy Optimization) via Stable Baselines3                                |
| **Environment**  | Custom `ClusterDispatchEnv` (OpenAI Gym) wrapping cluster simulation                    |
| **State space**  | CPU demand, memory demand, available instances (5 types), prices (5), time encoding (4) |
| **Action space** | Continuous [0,1]⁵ — fraction of available instances to allocate per type                |
| **Reward**       | `-(cost) − λ×(energy) − μ×(unmet CPU² + unmet mem²) − ν×(stranded)²`                    |

### Reward Components

| Component         | Description                                | Weight |
| ----------------- | ------------------------------------------ | ------ |
| Compute cost      | Sum of (instances × $/hr) across all types | 1.0    |
| Energy            | Total kW consumed                          | 0.001  |
| Unmet CPU demand  | Squared penalty for CPU shortfall          | 0.5    |
| Unmet mem demand  | Squared penalty for memory shortfall       | 0.5    |
| Stranded capacity | Penalty for >20% over-provisioning         | 0.2    |

### Baselines

| Policy          | Description                                     |
| --------------- | ----------------------------------------------- |
| **Random**      | Uniform random allocation fractions             |
| **Cost First**  | Cheapest machine types first (greedy heuristic) |
| **Equal Split** | All types allocated at equal fraction of demand |
| **PPO (ours)**  | Trained RL agent                                |

## Usage

### Option A — Docker (recommended)

```bash
# Build and run training
docker build -t quest-06-rl .
docker run --rm -v "$(pwd)/results:/app/results" quest-06-rl

# Launch Streamlit demo
docker run --rm -p 8506:8506 -v "$(pwd)/results:/app/results" quest-06-rl \
  streamlit run app.py --server.port=8506 --server.address=0.0.0.0

# Run evaluation
docker run --rm -v "$(pwd)/results:/app/results" quest-06-rl python src/evaluate.py

# Generate figures
docker run --rm -v "$(pwd)/results:/app/results" quest-06-rl python src/visualize.py

# Or use docker compose from the root
cd ..
docker compose up quest-06-rl
```

### Option B — Local venv

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install torch first (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Train the agent
python src/train.py

# Quick test run (200k timesteps)
python src/train.py --total-timesteps 200000

# Evaluate against baselines
python src/evaluate.py

# Generate figures
python src/visualize.py

# Launch demo
streamlit run app.py
```

## Pipeline

1. **Generate workload data** (`src/data_utils.py`) — synthetic CPU/memory demand,
   machine availability, and pricing based on Google Cluster Data patterns
2. **Build environment** (`src/cluster_env.py`) — `ClusterDispatchEnv` with
   Gymnasium interface, 16-dim state, 5-dim continuous action
3. **Train PPO agent** (`src/train.py`) — 500k–1M timesteps, MLP policy,
   VecNormalize
4. **Evaluate** (`src/evaluate.py`) — compare PPO against random, cost-first,
   and equal-split baselines
5. **Visualize** (`src/visualize.py`) — learning curves, allocation schedules,
   reward breakdowns
6. **Demo** (`app.py`) — Streamlit app with live allocation, results dashboard,
   and what-if scenarios

## Files

```
quest-06-rl-grid-balancing/
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py                     # Streamlit demo (Live Allocation, Results, What-If)
├── data/
│   ├── sample/                # Small sample workloads for quick testing
│   └── workload.pkl           # Generated workload sequence
├── src/
│   ├── data_utils.py          # Machine type definitions & workload generation
│   ├── cluster_env.py         # Custom Gymnasium environment
│   ├── train.py               # PPO training with Stable Baselines3
│   ├── evaluate.py            # Baseline comparison & metrics
│   └── visualize.py           # Plotting utilities
└── results/
    ├── model.zip              # Trained PPO model
    ├── vecnormalize.pkl       # Observation normalization stats
    ├── training_metadata.json # Hyperparameters & timing
    ├── eval_results.json      # Post-training evaluation
    ├── metrics.json           # Cross-policy comparison metrics
    ├── checkpoints/           # Training checkpoints
    ├── best_model/            # Best model from eval callback
    ├── eval_logs/             # SB3 evaluation logs
    └── figures/               # Generated plots
```

## Hardware Note

PPO on a small state/action space (16-dim obs, 5-dim action) trains entirely
on CPU. 1M timesteps completes in approximately 30–60 minutes with
Stable Baselines3.

## Real Data Option

To download and use actual Google Cluster Data 2011 traces:

```python
from src.data_utils import download_google_cluster_data
download_google_cluster_data()
```

This downloads a sample of machine events and task events from Google's
public dataset. The full trace is ~40 GB; the sample is ~500 MB. Use the
returned `ClusterParams` objects with `ClusterDispatchEnv` just like the
synthetic data.
