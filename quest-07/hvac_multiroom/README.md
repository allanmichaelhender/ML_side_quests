# 🏠 Multi-Zone HVAC RL Agent (Phase 2)

A **DQN** reinforcement learning agent controlling air conditioning in a
3-room building with inter-zone heat transfer, solar gain, and diurnal
outdoor temperature cycles.

## Building Layout

```
        ┌──────────┐
        │  Room 1   │  South-facing living room (solar gain)
        ├────┬──────┤
        │R2  │  R3  │  Kitchen (internal) & Bedroom (north-facing)
        └────┴──────┘
```

## Project Structure

```
hvac_multiroom/
├── app.py               # Streamlit dashboard
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   └── sample/
├── results/
│   ├── model.zip        # Trained DQN model
│   ├── metrics.json     # Training evaluation metrics
│   ├── eval_results.json # Per-scenario evaluation results
│   ├── training_meta.json # Hyperparameter record
│   ├── best_model/      # Best checkpoint during training
│   └── figures/         # Evaluation plots
└── src/
    ├── __init__.py
    ├── config.py         # Building thermal parameters
    ├── env.py            # MultiRoomHVACEnv (Gymnasium)
    ├── train.py          # DQN training script
    └── evaluate.py       # Multi-climate evaluation
```

## Usage

### Train the agent

```bash
cd quest-07/hvac_multiroom
python src/train.py --timesteps 500000
```

### Evaluate on climate scenarios

```bash
python src/evaluate.py
```

### Launch Streamlit dashboard

```bash
streamlit run app.py --server.port=8507
```

## Environment Details

| Aspect          | Detail                                                   |
| --------------- | -------------------------------------------------------- | -------------- | ------------------------ |
| **Observation** | `[T_living, T_kitchen, T_bedroom, outdoor, hour/24]`     |
| **Action**      | `MultiDiscrete([2,2,2])` — AC OFF/ON per zone (8 combos) |
| **Reward**      | `Σ -                                                     | T_i - target_i | `+`-1.5 × zones_with_AC` |
| **Algorithm**   | SB3 DQN (MlpPolicy, 2×128 hidden, γ=0.95)                |
| **Episode**     | 72 steps (3 days), diurnal outdoor temp repeats          |

## What the Agent Should Learn

1. **Solar-aware scheduling** — pre-cool the south room before midday solar peak
2. **Inter-zone coordination** — exploit heat transfer between adjacent zones
3. **Night flushing** — turn OFF AC when outdoor drops below indoor at night
4. **Priority shifting** — prioritise bedrooms at night, living room during day
