"""
Custom Gym environment for energy grid load balancing.

GridDispatchEnv is a discrete-time simulation of an electricity grid
where an RL agent must dispatch power from multiple generation sources
to meet demand at minimum cost, emissions, and with high reliability.

State space (11 dims):
  - Current demand (MW)
  - Available capacity per source: coal, gas, solar, wind, hydro (MW)
  - Current price per source: coal, gas, solar, wind, hydro ($/MWh)
  - Time encoding: hour_sin, hour_cos, day_sin, day_cos

Action space (5 dims, continuous Box):
  - Fraction of available capacity to dispatch from each source [0, 1]

Reward:
  r = -dispatch_cost - λ * emissions - μ * (unmet_demand + over_gen)^2
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import (
    SOURCE_DEFS,
    GridParams,
    generate_scenario_sequence,
    sample_grid_params,
)

# ── Reward weights ──────────────────────────────────────────────────────────
LAMBDA_COST = 1.0  # weight on $cost in reward
LAMBDA_EMISSIONS = 0.005  # weight on kg CO₂
LAMBDA_UNMET = 2.0  # weight on unmet demand penalty
MU_OVERGEN = 0.5  # weight on over-generation penalty

# How many timesteps in an episode
DEFAULT_EPISODE_LENGTH = 24 * 7  # one week


class GridDispatchEnv(gym.Env):
    """
    Grid load-balancing environment.

    Parameters
    ----------
    scenario_sequence : List[GridParams], optional
        Pre-generated sequence of hourly scenarios. If None, generates on-the-fly.
    episode_length : int
        Number of timesteps per episode (default 168 = one week).
    seed : int
        Random seed for scenario generation.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        scenario_sequence: Optional[List[GridParams]] = None,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.scenario_sequence = scenario_sequence
        self.episode_length = episode_length
        self.render_mode = render_mode
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Number of generation sources
        self.n_sources = len(SOURCE_DEFS)
        self.source_names = [s["name"] for s in SOURCE_DEFS]

        # ── Observation space ───────────────────────────────────────────────
        # demand(1) + avail_cap(5) + prices(5) + hour_sin/cos(2) + day_sin/cos(2)
        obs_dim = 1 + self.n_sources + self.n_sources + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ────────────────────────────────────────────────────
        # Fraction of available capacity to dispatch from each source [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_sources,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self._scenarios: List[GridParams] = []
        self._last_scenario: Optional[GridParams] = None
        self._total_reward = 0.0
        self._reward_breakdown: Dict[str, List[float]] = {
            "cost": [],
            "emissions": [],
            "unmet": [],
            "overgen": [],
            "total": [],
        }

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector from the current scenario."""
        s = self._last_scenario
        hour = s.hour
        day = s.day_of_week

        # Cyclical time encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        avail_caps = np.array(
            [
                s.source_availability[name] * src["max_cap"]
                for name, src in zip(self.source_names, SOURCE_DEFS)
            ],
            dtype=np.float32,
        )

        prices = np.array(
            [s.source_prices[name] for name in self.source_names], dtype=np.float32
        )

        obs = np.concatenate(
            [
                [s.demand_mw],
                avail_caps,
                prices,
                [hour_sin, hour_cos, day_sin, day_cos],
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(
        self,
        dispatch_mw: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown given dispatch decisions."""
        s = self._last_scenario
        total_dispatch = dispatch_mw.sum()

        cost = 0.0
        emissions = 0.0
        for i, src_name in enumerate(self.source_names):
            price = s.source_prices[src_name]
            cost += dispatch_mw[i] * price
            emissions += dispatch_mw[i] * SOURCE_DEFS[i]["co2_rate"]

        # Unmet demand
        unmet = max(0.0, s.demand_mw - total_dispatch)
        overgen = max(
            0.0, total_dispatch - s.demand_mw * 1.10
        )  # >10% over is penalised

        # Reward components (all negative penalties)
        cost_penalty = LAMBDA_COST * cost
        emissions_penalty = LAMBDA_EMISSIONS * emissions
        unmet_penalty = LAMBDA_UNMET * (unmet**2)
        overgen_penalty = MU_OVERGEN * (overgen**2)

        total_reward = -(
            cost_penalty + emissions_penalty + unmet_penalty + overgen_penalty
        )

        breakdown = {
            "cost": -cost_penalty,
            "emissions": -emissions_penalty,
            "unmet": -unmet_penalty,
            "overgen": -overgen_penalty,
            "total": total_reward,
        }
        return total_reward, breakdown

    def _dispatch_from_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalised action [0,1]^5 to actual MW dispatch."""
        s = self._last_scenario
        dispatch = np.zeros(self.n_sources, dtype=np.float32)
        for i, src_name in enumerate(self.source_names):
            avail_mw = s.source_availability[src_name] * SOURCE_DEFS[i]["max_cap"]
            min_mw = SOURCE_DEFS[i]["min_cap"]
            # Action is fraction of available capacity
            raw = action[i] * avail_mw
            # Enforce min stable generation if dispatching
            if raw > 0 and raw < min_mw:
                raw = min_mw
            dispatch[i] = np.clip(raw, 0.0, avail_mw)
        return dispatch

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self._reward_breakdown = {
            "cost": [],
            "emissions": [],
            "unmet": [],
            "overgen": [],
            "total": [],
        }
        self._total_reward = 0.0

        if self.scenario_sequence is not None:
            # Use the provided sequence; wrap around if needed
            self._scenarios = self.scenario_sequence
        else:
            # Generate fresh scenarios on-the-fly
            self._scenarios = generate_scenario_sequence(
                self.episode_length,
                seed=self._seed,
            )

        self._last_scenario = self._scenarios[0]
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a dispatch action and advance one hour."""
        # Dispatch
        dispatch_mw = self._dispatch_from_action(action)
        reward, breakdown = self._compute_reward(dispatch_mw)

        # Record
        self._total_reward += reward
        for k, v in breakdown.items():
            self._reward_breakdown[k].append(v)

        # Advance
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        if not terminated:
            self._last_scenario = self._scenarios[self.current_step]
        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = self._get_info()
        if terminated:
            info["episode"] = {
                "r": self._total_reward,
                "l": self.episode_length,
            }
            # Compute aggregate metrics
            info["avg_cost_per_mwh"] = self._aggregate_metric("cost", "cost")
            info["avg_emissions_per_mwh"] = self._aggregate_metric("emissions", "cost")
            info["reliability"] = self._compute_reliability()

        return obs, reward, terminated, truncated, info

    def _aggregate_metric(self, reward_key: str, denominator_key: str) -> float:
        """Compute a per-MWh metric from reward components."""
        values = self._reward_breakdown[reward_key]
        if not values:
            return 0.0
        total = -sum(values)  # flip sign (reward is negative penalty)
        # Approximate denominator: total dispatched MWh
        dispatched = len(values) * np.mean(
            [s.demand_mw for s in self._scenarios[: len(values)]]
        )
        return total / dispatched if dispatched > 0 else 0.0

    def _compute_reliability(self) -> float:
        """Fraction of timesteps where demand was fully met."""
        if not self._reward_breakdown["unmet"]:
            return 1.0
        # Count steps where unmet penalty was zero (demand met)
        unmet_steps = sum(1 for v in self._reward_breakdown["unmet"] if v == 0.0)
        return unmet_steps / len(self._reward_breakdown["unmet"])

    def _get_info(self) -> dict:
        """Return auxiliary info about the current state."""
        if self._last_scenario is None:
            return {}
        s = self._last_scenario
        return {
            "season": s.season,
            "hour": s.hour,
            "day_of_week": s.day_of_week,
            "demand_mw": s.demand_mw,
        }

    def render(self):
        """Render the environment state."""
        if self._last_scenario is None:
            return
        s = self._last_scenario
        out = (
            f"Grid State — Season: {s.season}, "
            f"Day {s.day_of_week}, Hour {s.hour:02d}:00 | "
            f"Demand: {s.demand_mw:.0f} MW | "
            f"Step: {self.current_step}/{self.episode_length}"
        )
        if self.render_mode == "human":
            print(out)
        return out


# ── Helper to create environment with pre-generated scenarios ──────────────


def make_env(
    scenario_sequence: Optional[List[GridParams]] = None,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    seed: int = 42,
    render_mode: Optional[str] = None,
) -> GridDispatchEnv:
    """Factory function that returns a configured GridDispatchEnv."""
    return GridDispatchEnv(
        scenario_sequence=scenario_sequence,
        episode_length=episode_length,
        seed=seed,
        render_mode=render_mode,
    )


if __name__ == "__main__":
    # Quick test
    env = make_env(episode_length=24)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Obs: {obs}")
    total = 0.0
    for _ in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"Step {env.current_step}: reward={reward:.2f}, total={total:.2f}")
        if terminated:
            break
    print(f"Episode total reward: {total:.2f}")
    print(f"Reliability: {info.get('reliability', 'N/A')}")
    print("Environment test complete.")
