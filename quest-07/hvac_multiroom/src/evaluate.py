"""
Evaluate the trained multi-zone HVAC agent across different climate scenarios.

Produces:
  results/eval_results.json — per-scenario metrics
  results/figures/           — comparison plots
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

# Ensure project root is on sys.path for package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import BuildingConfig
from src.env import MultiRoomHVACEnv


# Helper: decode integer action 0-7 to per-zone bits
def _action_bits(action: int):
    return MultiRoomHVACEnv._decode_action(action).tolist()


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_PATH = RESULTS_DIR / "model.zip"
EVAL_PATH = RESULTS_DIR / "eval_results.json"

# ── Climate scenarios ──────────────────────────────────────────────────────
SCENARIOS = [
    ("Cool climate  (22°C base)", 22.0),
    ("Warm climate  (27°C base)", 27.0),
    ("Hot climate   (32°C base)", 32.0),
]


def run_episode(env: MultiRoomHVACEnv, model: DQN, base_temp: float) -> dict:
    """Run one deterministic episode and record all state/action/reward data."""
    env.base_temp = base_temp
    obs, _ = env.reset()
    # Override base_temp after reset since reset randomises it
    env.base_temp = base_temp

    history = {
        "hours": [],
        "temps": [],  # list of [T_living, T_kitchen, T_bedroom]
        "outdoor": [],
        "actions": [],  # list of [ac1, ac2, ac3]
        "rewards": [],
        "total_reward": 0.0,
    }

    for step in range(env.cfg.episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        history["hours"].append(env.hour)
        history["temps"].append(obs[:3].tolist())
        history["outdoor"].append(float(obs[3]))
        history["actions"].append(_action_bits(action))

        obs, reward, terminated, truncated, _ = env.step(action)
        history["rewards"].append(reward)
        history["total_reward"] += reward

    return history


def plot_scenario(history: dict, label: str, color: str, save_path: Path) -> None:
    """Generate a multi-panel figure for one climate scenario."""
    hours = np.arange(len(history["hours"]))
    temps = np.array(history["temps"])
    outdoor = np.array(history["outdoor"])
    actions = np.array(history["actions"])
    rewards = np.array(history["rewards"])

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── Panel 1: Temperature traces ────────────────────────────────────────
    ax = axs[0]
    zone_names = ["Living", "Kitchen", "Bedroom"]
    zone_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    targets = [21.0, 22.0, 20.0]

    outdoor_color = "#f39c12"
    ax.plot(hours, outdoor, "--", color=outdoor_color, alpha=0.5, label="Outdoor")
    for i, (zn, zc) in enumerate(zip(zone_names, zone_colors)):
        ax.plot(hours, temps[:, i], "-", color=zc, label=f"{zn}", linewidth=2)
        ax.axhline(targets[i], color=zc, linestyle=":", alpha=0.4)

    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{label} — Temperature Traces")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add day separators
    for day in range(1, 4):
        ax.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.3)

    # ── Panel 2: AC usage ──────────────────────────────────────────────────
    ax = axs[1]
    bottom = np.zeros(len(hours))
    for i, (zn, zc) in enumerate(zip(zone_names, zone_colors)):
        ax.bar(
            hours,
            actions[:, i],
            bottom=bottom,
            label=zn,
            color=zc,
            alpha=0.7,
            width=0.8,
        )
        bottom += actions[:, i]

    ax.set_ylabel("Zones with AC ON")
    ax.set_title("AC Activation per Zone")
    ax.set_ylim(0, env.n_zones + 0.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for day in range(1, 4):
        ax.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.3)

    # ── Panel 3: Reward & comfort ──────────────────────────────────────────
    ax = axs[2]
    ax.plot(hours, rewards, "-", color="#9b59b6", label="Step reward", linewidth=1.5)
    ax.fill_between(hours, 0, rewards, alpha=0.15, color="#9b59b6")

    # Cumulative reward
    cumreward = np.cumsum(rewards)
    ax_twin = ax.twinx()
    ax_twin.plot(
        hours, cumreward, "--", color="#2c3e50", label="Cumulative", linewidth=2
    )
    ax_twin.set_ylabel("Cumulative Reward", color="#2c3e50")

    ax.set_ylabel("Step Reward")
    ax.set_xlabel("Hour")
    ax.set_title("Reward Progression")
    ax.legend(loc="upper left", fontsize=8)
    ax_twin.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    for day in range(1, 4):
        ax.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"ERROR: No trained model found at {MODEL_PATH}")
        print("Run `python src/train.py` first.")
        return

    print("=" * 55)
    print("3-Room Multi-Zone HVAC — Evaluation")
    print("=" * 55)

    # ── Load model ─────────────────────────────────────────────────────────
    model = DQN.load(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")

    # ── Evaluate on each scenario ──────────────────────────────────────────
    env = MultiRoomHVACEnv()
    all_results = {}

    for label, base_temp in SCENARIOS:
        print(f"\n{label}...")
        history = run_episode(env, model, base_temp)

        # Save data
        all_results[label] = {
            "base_temp": base_temp,
            "total_reward": round(history["total_reward"], 2),
            "hourly": [
                {
                    "hour": h,
                    "temps": t,
                    "outdoor": o,
                    "actions": a,
                    "reward": r,
                }
                for h, t, o, a, r in zip(
                    history["hours"],
                    history["temps"],
                    history["outdoor"],
                    history["actions"],
                    history["rewards"],
                )
            ],
        }

        # Compute summary stats
        temps_arr = np.array(history["temps"])
        actions_arr = np.array(history["actions"])
        targets = np.array(env.cfg.target_temps)

        comfort_deviation = np.mean(np.abs(temps_arr - targets), axis=0)
        ac_usage_pct = np.mean(actions_arr, axis=0) * 100

        all_results[label]["summary"] = {
            "mean_comfort_deviation_per_zone": {
                env.cfg.zone_names[i]: round(float(comfort_deviation[i]), 2)
                for i in range(env.n_zones)
            },
            "ac_usage_pct_per_zone": {
                env.cfg.zone_names[i]: round(float(ac_usage_pct[i]), 1)
                for i in range(env.n_zones)
            },
            "total_ac_hours": int(np.sum(actions_arr)),
        }

        print(f"  Total reward: {history['total_reward']:.2f}")
        for i, zn in enumerate(env.cfg.zone_names):
            print(
                f"  {zn:10s}: mean comfort deviation {comfort_deviation[i]:.2f}°C, "
                f"AC usage {ac_usage_pct[i]:.1f}%"
            )

        # Plot
        scenario_key = label.split("(")[0].strip().lower().replace(" ", "_")
        plot_path = FIGURES_DIR / f"{scenario_key}.png"
        colors = {"cool": "#2ecc71", "warm": "#f39c12", "hot": "#e74c3c"}
        plot_color = next((c for k, c in colors.items() if k in scenario_key), "#888")
        plot_scenario(history, label, plot_color, plot_path)
        print(f"  Figure saved to {plot_path}")

    env.close()

    # ── Save aggregated results ────────────────────────────────────────────
    with open(EVAL_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {EVAL_PATH}")

    # ── Summary comparison table ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"{'Scenario':<30} {'Total Reward':<15} {'AC Hours':<10}")
    print("-" * 55)
    for label in [s[0] for s in SCENARIOS]:
        r = all_results[label]
        print(
            f"{label:<30} {r['total_reward']:<15.2f} {r['summary']['total_ac_hours']:<10}"
        )
    print()


if __name__ == "__main__":
    main()
