# HVAC RL Project Plan

## Overview

Reinforcement learning for building HVAC control, progressing from a single-room toy model to a realistic multi-zone building simulation using industry-standard tools (Gymnasium, Stable-Baselines3).

---

## Phase 1 ✅ — Single room (Gymnasium + SB3)

**File:** `dqn_gymnasium.py`

| Aspect          | Detail                                                              |
| --------------- | ------------------------------------------------------------------- | ------------- | ----------------- |
| **Environment** | `gym.Env` — diurnal outdoor temp sine wave, randomized base 22–32°C |
| **Observation** | `[indoor_temp, outdoor_temp, hour/24]`                              |
| **Action**      | Discrete 2 — AC OFF / ON                                            |
| **Reward**      | `-                                                                  | temp - target | `+`-1.5` if AC ON |
| **Algorithm**   | SB3 DQN (`MlpPolicy`, 2×64 hidden)                                  |
| **Episode**     | 24 steps (1 day)                                                    |
| **Training**    | 100k timesteps (~4166 episodes)                                     |

**Key change from hand-written DQN:**

- Gymnasium enforces standard `reset(seed) → (obs, info)` / `step → (obs, reward, terminated, truncated, info)` API
- SB3 replaces all boilerplate: network, replay buffer, target network, ε-greedy schedule
- `"MlpPolicy"` auto-sizes the network from `observation_space` and `action_space`

---

## Phase 2 🏗️ — 3-room multi-zone building

### Project structure

```
quest-07/hvac_multiroom/
├── __init__.py          # Package marker
├── env.py               # MultiRoomHVACEnv
├── train.py             # SB3 training script
├── evaluate.py          # Test on multiple weather profiles
└── config.py            # Building parameters (room sizes, insulation, etc.)
```

### Environment design — `MultiRoomHVACEnv`

#### Building layout

```
        ┌──────────┐
        │  Room 1   │  ← South-facing (gets solar gain)
        │ (living)  │
        ├────┬──────┤
        │R2  │  R3  │  ← Internal and north-facing
        │(kitchen)│(bedroom)│
        └────┴──────┘
```

#### State space

```
Observation: [temp_z1, temp_z2, temp_z3, outdoor, hour/24]
             └─── 5 continuous values ──┘
```

#### Physics — per-room temperature update

Each zone's temperature evolves as:

```
ΔT_i = α_out_i × (outdoor - T_i)      ← heat exchange with outside
     + Σ α_ij × (T_j - T_i)           ← inter-zone conduction
     + solar_i(hour)                   ← solar gain (Room 1 only)
     + β_i × action_i                  ← AC cooling
```

Where:

- `α_out_i` — insulation quality per room (varies: south wall thinner?)
- `α_ij` — thermal conductivity between adjacent zones
- `solar_i(hour)` — sinusoidal solar gain peaking at noon (south-facing room)
- `β_i` — AC cooling power per room

#### Action space — two approaches

**Option A — Central agent (single brain):**

```python
action = [ac_z1, ac_z2, ac_z3]  # each 0 or 1
# MultiDiscrete([2, 2, 2]) → 8 possible combos
```

**Option B — Per-zone setpoint (continuous):**

```python
action = [setpoint_z1, setpoint_z2, setpoint_z3]
# Box(15, 35, shape=(3,))
# Needs SAC/TD3 instead of DQN (continuous actions)
```

#### Reward

```
total_reward = Σ comfort_i + energy_penalty
comfort_i    = -|T_i - target_i|      ← each room can have different target
energy_cost  = -1.5 × sum(action_i)   ← shared AC energy penalty
```

#### What the agent should learn

1. **Solar-aware scheduling** — pre-cool the south room before midday solar peak
2. **Inter-zone coordination** — cool the living room less if adjacent kitchen AC is already running (heat transfer)
3. **Night flushing** — open (turn OFF AC) when outdoor drops below indoor at night
4. **Priority shifting** — if energy budget is tight, prioritize bedrooms at night, living room during day

---

## Possible extensions

| Extension               | How                                                                 |
| ----------------------- | ------------------------------------------------------------------- |
| **Continuous control**  | SAC/TD3 with setpoint temperatures instead of ON/OFF                |
| **Real weather data**   | Replace sine wave with actual `.epw` weather file                   |
| **EnergyPlus**          | Co-simulate with `gym-energym` for realistic building physics       |
| **Multi-agent**         | One DQN per room, shared energy cost — trains faster per agent      |
| **Occupancy**           | Add occupancy sensor (binary or count) — modulate comfort target    |
| **Electricity pricing** | Time-of-use tariff — shift cooling to cheap hours (thermal battery) |
| **Model-based RL**      | Learn the physics model and plan ahead (Dreamer, PETS)              |

---

## Requirements

```
gymnasium>=0.29
stable-baselines3>=2.0
numpy
torch
```
