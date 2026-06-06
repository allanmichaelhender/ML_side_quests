"""
Evaluation script for data center resource scheduling agents.

Compares PPO-trained agent against baselines:
  - Random policy
  - Cost-first policy (cheapest machine type first)
  - Equal-split policy (all types equally)

Generates metrics and comparison plots.
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import (
    MACHINE_DEFS,
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    generate_workload_sequence,
    load_workload_sequence,
)
from cluster_env import ClusterDispatchEnv, make_env

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ── Baseline policies ──────────────────────────────────────────────────────


def random_policy(obs: np.ndarray, env: ClusterDispatchEnv) -> np.ndarray:
    """Random allocation: sample uniform fractions."""
    return env.action_space.sample()


def cost_first_policy(obs: np.ndarray, env: ClusterDispatchEnv) -> np.ndarray:
    """
    Cost-first allocation: allocate cheapest machine types first
    until CPU demand is met. Simple greedy heuristic.
    """
    s = env._last_scenario
    # Sort machine types by price (cheapest first)
    priced = sorted(
        [
            (s.machine_prices[m["name"]], m, s.machine_availability[m["name"]])
            for m in MACHINE_DEFS
        ],
        key=lambda x: x[0],
    )

    action = np.zeros(len(MACHINE_DEFS), dtype=np.float32)
    remaining_cpu = s.cpu_demand
    remaining_mem = s.mem_demand

    for price, m, avail_frac in priced:
        idx = next(i for i, m2 in enumerate(MACHINE_DEFS) if m2["name"] == m["name"])
        avail = avail_frac * m["max_instances"]
        if remaining_cpu <= 0 and remaining_mem <= 0:
            break

        # How many instances needed to meet remaining demand
        need_cpu = (
            max(0, remaining_cpu / m["cpu_per_instance"])
            if m["cpu_per_instance"] > 0
            else 0
        )
        need_mem = (
            max(0, remaining_mem / m["mem_per_instance"])
            if m["mem_per_instance"] > 0
            else 0
        )
        need = max(need_cpu, need_mem)

        take = min(avail, need)
        action[idx] = take / avail if avail > 0 else 0.0
        remaining_cpu -= take * m["cpu_per_instance"]
        remaining_mem -= take * m["mem_per_instance"]

    return action


def equal_split_policy(obs: np.ndarray, env: ClusterDispatchEnv) -> np.ndarray:
    """
    Equal-split allocation: allocate all machine types at equal fraction,
    scaled to meet CPU demand.
    """
    s = env._last_scenario
    total_avail_instances = sum(
        s.machine_availability[m["name"]] * m["max_instances"] for m in MACHINE_DEFS
    )
    if total_avail_instances <= 0:
        return np.zeros(len(MACHINE_DEFS), dtype=np.float32)

    total_avail_cpu = sum(
        s.machine_availability[m["name"]] * m["max_instances"] * m["cpu_per_instance"]
        for m in MACHINE_DEFS
    )
    fraction = min(1.0, s.cpu_demand / total_avail_cpu) if total_avail_cpu > 0 else 0.0
    action = np.full(len(MACHINE_DEFS), fraction, dtype=np.float32)
    return action


# ── Evaluation runner ──────────────────────────────────────────────────────


def evaluate_policy(
    policy_fn: Callable,
    env: ClusterDispatchEnv,
    n_episodes: int = 10,
    policy_name: str = "policy",
) -> Dict:
    """Run a policy for multiple episodes and aggregate metrics."""
    all_rewards = []
    all_info = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        all_rewards.append(ep_reward)
        all_info.append(info)

    # Aggregate
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    cpu_reliabilities = [inf.get("cpu_reliability", 0.0) for inf in all_info]
    mem_reliabilities = [inf.get("mem_reliability", 0.0) for inf in all_info]
    costs = [inf.get("avg_cost_per_hour", 0.0) for inf in all_info]
    energy = [inf.get("avg_energy_kw", 0.0) for inf in all_info]

    results = {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "mean_cpu_reliability": round(float(np.mean(cpu_reliabilities)), 4),
        "mean_mem_reliability": round(float(np.mean(mem_reliabilities)), 4),
        "mean_cost_per_hour": round(float(np.mean(costs)), 2),
        "mean_energy_kw": round(float(np.mean(energy)), 2),
    }
    return results


def load_trained_model(model_path: Path, vec_norm_path: Optional[Path] = None):
    """Load a trained PPO model and optionally its VecNormalize stats."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(str(model_path))

    vec_norm = None
    if vec_norm_path and vec_norm_path.exists():
        # Create a dummy venv just to host VecNormalize for manual normalization
        dummy_venv = DummyVecEnv(
            [
                lambda: make_env(
                    workload_sequence=generate_workload_sequence(1, seed=0),
                    episode_length=1,
                    seed=0,
                )
            ]
        )
        vec_norm = VecNormalize.load(str(vec_norm_path), venv=dummy_venv)
        vec_norm.training = False
        vec_norm.norm_reward = False

    return model, vec_norm


def trained_agent_policy(model, vec_norm=None):
    """Return a policy function that uses the trained PPO model."""

    def policy_fn(obs, env):
        # Normalize observation if we have VecNormalize stats
        if vec_norm is not None:
            obs_normed = vec_norm.normalize_obs(obs[np.newaxis, :])[0]
        else:
            obs_normed = obs
        action, _ = model.predict(obs_normed[np.newaxis, :], deterministic=True)
        return action[0]

    return policy_fn


# ── Main evaluation ────────────────────────────────────────────────────────


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Data Center Resource Scheduling — Evaluation")
    print("=" * 60)

    # ── Generate evaluation scenarios ──────────────────────────────────────
    print("\nGenerating evaluation workload scenarios...")
    eval_scenarios = generate_workload_sequence(
        n_hours=24 * 365,  # 1 year
        seed=42,
    )

    # ── Create environments ────────────────────────────────────────────────
    env = make_env(
        workload_sequence=eval_scenarios,
        episode_length=24 * 7,  # 1 week episodes
        seed=42,
    )

    # ── Baseline policies ──────────────────────────────────────────────────
    print("\nEvaluating baselines...")
    baseline_policies = {
        "Random": random_policy,
        "Cost First": cost_first_policy,
        "Equal Split": equal_split_policy,
    }

    all_results = []
    for name, policy_fn in baseline_policies.items():
        print(f"  Running {name}...")
        results = evaluate_policy(
            policy_fn,
            env,
            n_episodes=10,
            policy_name=name,
        )
        all_results.append(results)
        print(f"    Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"    CPU Reliability: {results['mean_cpu_reliability']:.4f}")
        print(f"    Cost: ${results['mean_cost_per_hour']:.2f}/hr")
        print(f"    Energy: {results['mean_energy_kw']:.2f} kW")

    # ── Trained model ──────────────────────────────────────────────────────
    model_path = RESULTS_DIR / "model.zip"
    vec_norm_path = RESULTS_DIR / "vecnormalize.pkl"
    if model_path.exists():
        print(f"\nLoading trained model from {model_path}...")
        try:
            model, vec_norm = load_trained_model(model_path, vec_norm_path)
        except (AssertionError, ValueError, RuntimeError) as e:
            print(f"  Could not load model (likely trained on different env): {e}")
            print(
                "  Skipping PPO evaluation. Train a new model with: python src/train.py"
            )
            model = None

        if model is not None:
            policy_fn = trained_agent_policy(model, vec_norm)
            print("Evaluating PPO agent...")
            results = evaluate_policy(
                policy_fn,
                env,
                n_episodes=10,
                policy_name="PPO",
            )
            all_results.append(results)
            print(
                f"    Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
            )
            print(f"    CPU Reliability: {results['mean_cpu_reliability']:.4f}")
            print(f"    Cost: ${results['mean_cost_per_hour']:.2f}/hr")
            print(f"    Energy: {results['mean_energy_kw']:.2f} kW")
    else:
        print(f"\nNo trained model found at {model_path}. Skipping PPO evaluation.")
        print("Train a model first with: python src/train.py")

    # ── Save comparison metrics ────────────────────────────────────────────
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    header = (
        f"{'Policy':<20} {'Reward':>10} {'CPU Rel':>8} "
        f"{'Mem Rel':>8} {'Cost($)':>10} {'Energy':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['policy']:<20} {r['mean_reward']:>8.0f} ±{r['std_reward']:>5.0f} "
            f"{r['mean_cpu_reliability']:>7.3f}  "
            f"{r['mean_mem_reliability']:>7.3f}  "
            f"{r['mean_cost_per_hour']:>7.1f}  "
            f"{r['mean_energy_kw']:>7.1f}"
        )


if __name__ == "__main__":
    main()
