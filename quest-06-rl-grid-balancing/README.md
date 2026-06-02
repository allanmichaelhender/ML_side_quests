# Quest 6 — Reinforcement Learning: Energy Grid Load Balancing Agent

Train a **PPO** (Proximal Policy Optimization) agent to dispatch power from multiple generation sources (coal, gas, solar, wind, hydro) to meet hourly electricity demand at minimum cost, emissions, and maximum reliability.

## Dataset

- **Demand curves**: Parameterised by EIA hourly electricity demand patterns — synthetic profiles with realistic double-peak daily shape and seasonal variation
- **Generation mix**: Inspired by NREL renewable generation profiles — solar availability follows diurnal/seasonal cycles, wind follows Weibull distributions, hydro varies by season
- **Spot pricing**: Variable operating cost plus stochastic noise, reflecting real-world electricity market dynamics
- Data is **synthetically generated** using realistic distributions (no API keys required)

## Approach

| Component        | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| **Algorithm**    | PPO (Proximal Policy Optimization) via Stable Baselines3          |
| **Environment**  | Custom `GridDispatchEnv` (OpenAI Gym) wrapping grid simulation    |
| **State space**  | Demand, available capacity (5 sources), prices (5), time encoding |
| **Action space** | Continuous [0,1]^5 — fraction of available capacity per source    |
| **Reward**       | `-(cost) − λ×(emissions) − μ×(unmet demand)²`                     |

### Reward Components

| Component       | Description                              | Weight |
| --------------- | ---------------------------------------- | ------ |
| Dispatch cost   | Sum of (MW × $/MWh) across all sources   | 1.0    |
| CO₂ emissions   | Total kg CO₂ emitted                     | 0.005  |
| Unmet demand    | Squared penalty for load shedding        | 2.0    |
| Over-generation | Squared penalty for >10% excess dispatch | 0.5    |

### Baselines

| Policy          | Description                                          |
| --------------- | ---------------------------------------------------- |
| **Random**      | Uniform random dispatch fractions                    |
| **Merit Order** | Cheapest available sources first (economic dispatch) |
| **Equal Split** | All sources dispatched at equal fraction of demand   |
| **PPO (ours)**  | Trained RL agent                                     |

## Results

| Metric             | Random       | Merit Order | Equal Split | PPO (trained) |
| ------------------ | ------------ | ----------- | ----------- | ------------- |
| Mean Reward        | −414,989,300 | −13,540,191 | −15,536,892 | −19,968,724   |
| Supply Reliability | 2.2%         | 48.8%       | 46.4%       | **91.7%**     |
| Avg Cost ($/MWh)   | $20.76       | $29.08      | $33.78      | $39.96        |
| Avg Emissions (kg) | 1.53         | 2.51        | 2.47        | 3.00          |

> **Key finding**: The PPO agent achieves **91.7% supply reliability** — nearly double the best baseline (48.8% for merit order) — at a modest increase in cost ($39.96 vs $29.08/MWh). This represents the classic reliability-vs-cost tradeoff in grid operations.

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

1. **Generate grid parameters** (`src/data_utils.py`) — synthetic demand, renewable availability, and pricing data
2. **Build environment** (`src/grid_env.py`) — `GridDispatchEnv` with Gymnasium interface
3. **Train PPO agent** (`src/train.py`) — 500k–1M timesteps, MLP policy, VecNormalize
4. **Evaluate** (`src/evaluate.py`) — compare PPO against random, merit-order, and equal-split baselines
5. **Visualize** (`src/visualize.py`) — learning curves, dispatch schedules, reward breakdowns
6. **Demo** (`app.py`) — Streamlit app with live dispatch, results dashboard, and what-if scenarios

## Files

```
quest-06-rl-grid-balancing/
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py                     # Streamlit demo (Live Dispatch, Results, What-If)
├── data/
│   ├── sample/                # Small sample scenarios for quick testing
│   └── scenarios.pkl          # Generated scenario sequence
├── src/
│   ├── data_utils.py          # Grid parameter generation & source definitions
│   ├── grid_env.py            # Custom Gymnasium environment
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

PPO on a small state/action space (11-dim obs, 5-dim action) trains entirely on CPU. 1M timesteps completes in approximately 30–60 minutes with Stable Baselines3.

## Ablation: Cost vs Emissions Tradeoff

By varying the emissions weight λ in the reward function, the agent learns different dispatch policies:

- **Low λ** → favours cheap but dirty sources (coal, gas)
- **High λ** → favours clean but potentially costlier sources (solar, wind, hydro)

This produces a Pareto frontier showing the inherent tradeoff between economic and environmental objectives.
