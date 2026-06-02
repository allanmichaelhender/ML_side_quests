"""
Evaluation script for grid load-balancing agents.

Compares PPO-trained agent against baselines:
  - Random policy
  - Merit-order dispatch (cheapest source first)
  - Equal-split dispatch (all sources equally)

Generates metrics and comparison plots.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import (
    SOURCE_DEFS,
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    generate_scenario_sequence,
    load_scenario_sequence,
)
from grid_env import GridDispatchEnv, make_env

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ── Baseline policies ──────────────────────────────────────────────────────


def random_policy(obs: np.ndarray, env: GridDispatchEnv) -> np.ndarray:
    """Random dispatch: sample uniform fractions."""
    return env.action_space.sample()


def merit_order_policy(obs: np.ndarray, env: GridDispatchEnv) -> np.ndarray:
    """
    Merit-order dispatch: dispatch cheapest available sources first
    until demand is met. This mimics real-world economic dispatch.
    """
    s = env._last_scenario
    # Sort sources by price (cheapest first)
    priced = sorted(
        [
            (s.source_prices[src["name"]], src, s.source_availability[src["name"]])
            for src in SOURCE_DEFS
        ],
        key=lambda x: x[0],
    )

    action = np.zeros(len(SOURCE_DEFS), dtype=np.float32)
    remaining = s.demand_mw

    for price, src, avail_frac in priced:
        idx = next(i for i, s2 in enumerate(SOURCE_DEFS) if s2["name"] == src["name"])
        avail_mw = avail_frac * src["max_cap"]
        if remaining <= 0:
            break
        take = min(avail_mw, remaining)
        action[idx] = take / avail_mw if avail_mw > 0 else 0.0
        remaining -= take

    return action


def equal_split_policy(obs: np.ndarray, env: GridDispatchEnv) -> np.ndarray:
    """
    Equal-split dispatch: dispatch all sources at equal fraction,
    scaled to meet demand.
    """
    s = env._last_scenario
    total_avail = sum(
        s.source_availability[src["name"]] * src["max_cap"] for src in SOURCE_DEFS
    )
    if total_avail <= 0:
        return np.zeros(len(SOURCE_DEFS), dtype=np.float32)

    fraction = min(1.0, s.demand_mw / total_avail)
    action = np.full(len(SOURCE_DEFS), fraction, dtype=np.float32)
    return action


# ── Evaluation runner ──────────────────────────────────────────────────────


def evaluate_policy(
    policy_fn,
    env: GridDispatchEnv,
    n_episodes: int = 10,
    policy_name: str = "policy",
    deterministic: bool = True,
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
    reliabilities = [inf.get("reliability", 0.0) for inf in all_info]
    costs = [inf.get("avg_cost_per_mwh", 0.0) for inf in all_info]
    emissions = [inf.get("avg_emissions_per_mwh", 0.0) for inf in all_info]

    results = {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "mean_reliability": round(float(np.mean(reliabilities)), 4),
        "std_reliability": round(float(np.std(reliabilities)), 4),
        "mean_cost_per_mwh": round(float(np.mean(costs)), 2),
        "mean_emissions_per_mwh": round(float(np.mean(emissions)), 2),
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
                    scenario_sequence=generate_scenario_sequence(1, seed=0),
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
    print("Grid Load Balancing — Evaluation")
    print("=" * 60)

    # ── Generate evaluation scenarios ──────────────────────────────────────
    print("\nGenerating evaluation scenarios...")
    eval_scenarios = generate_scenario_sequence(
        n_hours=24 * 365,  # 1 year
        seed=42,
    )

    # ── Create environments ────────────────────────────────────────────────
    env = make_env(
        scenario_sequence=eval_scenarios,
        episode_length=24 * 7,  # 1 week episodes
        seed=42,
    )

    # ── Baseline policies ──────────────────────────────────────────────────
    print("\nEvaluating baselines...")
    baseline_policies = {
        "Random": random_policy,
        "Merit Order": merit_order_policy,
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
        print(f"    Reliability: {results['mean_reliability']:.4f}")
        print(f"    Cost: ${results['mean_cost_per_mwh']:.2f}/MWh")
        print(f"    Emissions: {results['mean_emissions_per_mwh']:.2f} kg/MWh")

    # ── Trained model ──────────────────────────────────────────────────────
    model_path = RESULTS_DIR / "model.zip"
    vec_norm_path = RESULTS_DIR / "vecnormalize.pkl"
    if model_path.exists():
        print(f"\nLoading trained model from {model_path}...")
        model, vec_norm = load_trained_model(model_path, vec_norm_path)
        policy_fn = trained_agent_policy(model, vec_norm)

        print("Evaluating PPO agent...")
        results = evaluate_policy(
            policy_fn,
            env,
            n_episodes=10,
            policy_name="PPO",
        )
        all_results.append(results)
        print(f"    Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"    Reliability: {results['mean_reliability']:.4f}")
        print(f"    Cost: ${results['mean_cost_per_mwh']:.2f}/MWh")
        print(f"    Emissions: {results['mean_emissions_per_mwh']:.2f} kg/MWh")
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
    header = f"{'Policy':<20} {'Reward':>10} {'Reliability':>12} {'Cost($)':>10} {'CO₂(kg)':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['policy']:<20} {r['mean_reward']:>8.0f} ±{r['std_reward']:>5.0f} "
            f"{r['mean_reliability']:>10.3f}  "
            f"{r['mean_cost_per_mwh']:>7.1f}  "
            f"{r['mean_emissions_per_mwh']:>7.1f}"
        )


if __name__ == "__main__":
    main()
