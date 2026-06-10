"""
Custom Gym environment for data center resource scheduling.

ClusterDispatchEnv is a discrete-time simulation of a data center cluster
where an RL agent must allocate machines (from multiple types) to meet
computing workload demand at minimum cost, energy, and with high reliability.

State space (21 dims):
  - CPU demand (vCPU cores)
  - Memory demand (GB)
  - GPU demand (GPU-accelerated tasks)
  - Storage IOPS demand
  - Power budget (kW)
  - Carbon intensity (kg CO2/kWh)
  - Available instances per type: general, compute_opt, memory_opt, gpu, storage_opt
  - Current price per type ($/hr)
  - Time encoding: hour_sin, hour_cos, day_sin, day_cos

Action space (5 dims, continuous Box):
  - Fraction of available instances to allocate from each type [0, 1]

Reward:
  r = -(cost + λ*energy + μ_c*(unmet_cpu²) + μ_m*(unmet_mem²)
          + μ_g*(unmet_gpu²) + μ_i*(unmet_iops²)
          + β*power_overuse² + γ*carbon_cost + ν*stranded²)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import (
    MACHINE_DEFS,
    ClusterParams,
    generate_workload_sequence,
    estimate_total_cpu_capacity,
    estimate_total_mem_capacity,
)

# ── Reward weights (all linear — no squaring) ──────────────────────────────
# Weights are calibrated so each component contributes ~200-2000 when "bad"
LAMBDA_COST = 2.0  # weight on $ cost (lower so meeting demand is prioritized)
LAMBDA_ENERGY = 0.001  # weight on energy penalty (largely captured by cost + carbon)
LAMBDA_UNMET_CPU = 2.0  # weight on unmet CPU demand (primary SLI)
LAMBDA_UNMET_MEM = 1.0  # weight on unmet memory demand
LAMBDA_UNMET_GPU = 150.0  # weight on unmet GPU demand (tasks literally can't run)
LAMBDA_UNMET_IOPS = 0.05  # weight on unmet IOPS demand (performance degrades)
LAMBDA_POWER_OVERUSE = 80.0  # weight on exceeding power budget (kW)
LAMBDA_CARBON = 30.0  # weight on carbon emissions cost (per kg CO2)
NU_STRANDED = 1000.0  # weight on stranded over-provisioning (ratio scale)

# How many timesteps in an episode
DEFAULT_EPISODE_LENGTH = 24 * 7  # one week


class ClusterDispatchEnv(gym.Env):
    """
    Data center resource scheduling environment.

    Parameters
    ----------
    workload_sequence : List[ClusterParams], optional
        Pre-generated sequence of hourly workload scenarios.
        If None, generates on-the-fly.
    episode_length : int
        Number of timesteps per episode (default 168 = one week).
    seed : int
        Random seed for workload generation.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        workload_sequence: Optional[List[ClusterParams]] = None,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.workload_sequence = workload_sequence
        self.episode_length = episode_length
        self.render_mode = render_mode
        self._seed = seed

        # Number of machine types
        self.n_types = len(MACHINE_DEFS)
        self.type_names = [m["name"] for m in MACHINE_DEFS]

        # ── Observation space ───────────────────────────────────────────────
        # cpu_demand(1) + mem_demand(1) + gpu_demand(1) + iops_demand(1)
        # + power_budget(1) + carbon_intensity(1) = 6
        # + avail_instances(5) + prices(5)
        # + hour_sin/cos(2) + day_sin/cos(2) = 4
        # Total: 6 + 10 + 4 = 20
        obs_dim = 6 + self.n_types + self.n_types + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ────────────────────────────────────────────────────
        # Fraction of available instances to allocate from each type [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_types,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self._scenarios: List[ClusterParams] = []
        self._last_scenario: Optional[ClusterParams] = None
        self._total_reward = 0.0
        self._reward_breakdown: Dict[str, List[float]] = {
            "cost": [],
            "energy": [],
            "unmet_cpu": [],
            "unmet_mem": [],
            "unmet_gpu": [],
            "unmet_iops": [],
            "power_overuse": [],
            "carbon": [],
            "stranded": [],
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

        avail_instances = np.array(
            [
                s.machine_availability[name] * m["max_instances"]
                for name, m in zip(self.type_names, MACHINE_DEFS)
            ],
            dtype=np.float32,
        )

        prices = np.array(
            [s.machine_prices[name] for name in self.type_names], dtype=np.float32
        )

        obs = np.concatenate(
            [
                [s.cpu_demand],
                [s.mem_demand],
                [float(s.gpu_demand)],
                [s.storage_iops_demand],
                [s.power_budget_kw],
                [s.carbon_intensity],
                avail_instances,
                prices,
                [hour_sin, hour_cos, day_sin, day_cos],
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(
        self,
        allocated_instances: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown given allocation decisions."""
        s = self._last_scenario

        # Compute what each machine type provides
        total_cpu = 0.0
        total_mem = 0.0
        total_gpu = 0.0
        total_iops = 0.0
        cost = 0.0
        energy = 0.0

        for i, type_name in enumerate(self.type_names):
            n_allocated = allocated_instances[i]
            m = MACHINE_DEFS[i]
            total_cpu += n_allocated * m["cpu_per_instance"]
            total_mem += n_allocated * m["mem_per_instance"]
            total_gpu += n_allocated * m.get("gpu_per_instance", 0)
            total_iops += n_allocated * m.get("iops_per_instance", 0)
            cost += n_allocated * s.machine_prices[type_name]
            energy += n_allocated * m["power_watts"] / 1000.0  # kW

        # Unmet CPU and memory demand
        unmet_cpu = max(0.0, s.cpu_demand - total_cpu)
        unmet_mem = max(0.0, s.mem_demand - total_mem)

        # Unmet GPU demand — tasks that need GPU acceleration
        unmet_gpu = max(0.0, s.gpu_demand - total_gpu)

        # Unmet IOPS demand — storage throughput shortfall
        unmet_iops = max(0.0, s.storage_iops_demand - total_iops)

        # Power overuse penalty — exceeding data center power budget
        total_power_kw = energy  # already in kW from power_watts / 1000
        power_overuse = max(0.0, total_power_kw - s.power_budget_kw)

        # Carbon cost — emissions from energy consumption
        carbon_kg = energy * s.carbon_intensity

        # Stranded (over-provisioned) instances — allocated but not needed
        cpu_ratio = total_cpu / s.cpu_demand if s.cpu_demand > 0 else 1.0
        mem_ratio = total_mem / s.mem_demand if s.mem_demand > 0 else 1.0
        stranded_penalty = 0.0
        if cpu_ratio > 1.2 and mem_ratio > 1.2:
            stranded_penalty = min(cpu_ratio, mem_ratio) - 1.0

        # Reward components (all linear penalties for balanced scaling)
        cost_penalty = LAMBDA_COST * cost
        energy_penalty = LAMBDA_ENERGY * energy
        unmet_cpu_penalty = LAMBDA_UNMET_CPU * unmet_cpu
        unmet_mem_penalty = LAMBDA_UNMET_MEM * unmet_mem
        unmet_gpu_penalty = LAMBDA_UNMET_GPU * unmet_gpu
        unmet_iops_penalty = LAMBDA_UNMET_IOPS * unmet_iops
        power_penalty = LAMBDA_POWER_OVERUSE * power_overuse
        carbon_penalty = LAMBDA_CARBON * carbon_kg
        stranded = NU_STRANDED * stranded_penalty

        total_reward = -(
            cost_penalty
            + energy_penalty
            + unmet_cpu_penalty
            + unmet_mem_penalty
            + unmet_gpu_penalty
            + unmet_iops_penalty
            + power_penalty
            + carbon_penalty
            + stranded
        )

        breakdown = {
            "cost": -cost_penalty,
            "energy": -energy_penalty,
            "unmet_cpu": -unmet_cpu_penalty,
            "unmet_mem": -unmet_mem_penalty,
            "unmet_gpu": -unmet_gpu_penalty,
            "unmet_iops": -unmet_iops_penalty,
            "power_overuse": -power_penalty,
            "carbon": -carbon_penalty,
            "stranded": -stranded,
            "total": total_reward,
        }
        return total_reward, breakdown

    def _allocate_from_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalised action [0,1]^5 to actual instance allocation."""
        s = self._last_scenario
        allocated = np.zeros(self.n_types, dtype=np.float32)
        for i, type_name in enumerate(self.type_names):
            avail = s.machine_availability[type_name] * MACHINE_DEFS[i]["max_instances"]
            min_inst = MACHINE_DEFS[i]["min_instances"]
            # Action is fraction of available instances
            raw = action[i] * avail
            # Enforce minimum instances if allocating
            if raw > 0 and raw < min_inst:
                raw = min_inst
            allocated[i] = np.clip(raw, 0.0, avail)
        return allocated

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

        self.current_step = 0
        self._reward_breakdown = {
            "cost": [],
            "energy": [],
            "unmet_cpu": [],
            "unmet_mem": [],
            "unmet_gpu": [],
            "unmet_iops": [],
            "power_overuse": [],
            "carbon": [],
            "stranded": [],
            "total": [],
        }
        self._total_reward = 0.0

        if self.workload_sequence is not None:
            self._scenarios = self.workload_sequence
        else:
            self._scenarios = generate_workload_sequence(
                self.episode_length,
                seed=self._seed,
            )

        self._last_scenario = self._scenarios[0]
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take an allocation action and advance one hour."""
        allocated = self._allocate_from_action(action)
        reward, breakdown = self._compute_reward(allocated)

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
            info["avg_cost_per_hour"] = self._aggregate_metric("cost", "cost")
            info["avg_energy_kw"] = self._aggregate_metric("energy", "cost")
            info["cpu_reliability"] = self._compute_reliability("unmet_cpu")
            info["mem_reliability"] = self._compute_reliability("unmet_mem")
            info["gpu_reliability"] = self._compute_reliability("unmet_gpu")
            info["iops_reliability"] = self._compute_reliability("unmet_iops")
            info["avg_power_overuse_kw"] = self._aggregate_metric(
                "power_overuse", "cost"
            )
            info["avg_carbon_kg"] = self._aggregate_metric("carbon", "cost")

        return obs, reward, terminated, truncated, info

    def _aggregate_metric(self, reward_key: str, _denom_key: str) -> float:
        """Compute per-timestep average of a metric from reward components."""
        values = self._reward_breakdown[reward_key]
        if not values:
            return 0.0
        total = -sum(values)  # flip sign (reward is negative penalty)
        return total / len(values)

    def _compute_reliability(self, unmet_key: str) -> float:
        """Fraction of timesteps where demand was fully met for a resource."""
        if not self._reward_breakdown[unmet_key]:
            return 1.0
        met = sum(1 for v in self._reward_breakdown[unmet_key] if v == 0.0)
        return met / len(self._reward_breakdown[unmet_key])

    def _get_info(self) -> dict:
        """Return auxiliary info about the current state."""
        if self._last_scenario is None:
            return {}
        s = self._last_scenario
        return {
            "hour": s.hour,
            "day_of_week": s.day_of_week,
            "cpu_demand": s.cpu_demand,
            "mem_demand": s.mem_demand,
            "gpu_demand": s.gpu_demand,
            "iops_demand": s.storage_iops_demand,
            "power_budget_kw": s.power_budget_kw,
            "carbon_intensity": s.carbon_intensity,
        }

    def render(self):
        """Render the environment state."""
        if self._last_scenario is None:
            return
        s = self._last_scenario
        out = (
            f"Cluster State — Day {s.day_of_week}, Hour {s.hour:02d}:00 | "
            f"CPU: {s.cpu_demand:.0f} vCores | "
            f"Mem: {s.mem_demand:.0f} GB | "
            f"GPU tasks: {s.gpu_demand:.0f} | "
            f"IOPS: {s.storage_iops_demand:.0f} | "
            f"Power budget: {s.power_budget_kw:.1f} kW | "
            f"Carbon: {s.carbon_intensity:.3f} kg/kWh | "
            f"Step: {self.current_step}/{self.episode_length}"
        )
        if self.render_mode == "human":
            print(out)
        return out


# ── Helper to create environment ────────────────────────────────────────────


def make_env(
    workload_sequence: Optional[List[ClusterParams]] = None,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    seed: int = 42,
    render_mode: Optional[str] = None,
) -> ClusterDispatchEnv:
    """Factory function that returns a configured ClusterDispatchEnv."""
    return ClusterDispatchEnv(
        workload_sequence=workload_sequence,
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
    print(f"Total CPU capacity: {estimate_total_cpu_capacity():.0f} vCores")
    print(f"Total mem capacity: {estimate_total_mem_capacity():.0f} GB")
    total = 0.0
    for _ in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
    print(f"Total reward (random policy, 24h): {total:.2f}")
    print("Cluster environment ready.")
