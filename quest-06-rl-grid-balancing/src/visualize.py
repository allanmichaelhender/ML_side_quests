"""
Visualization utilities for the data center resource scheduling project.

Produces:
  - Learning curves (reward vs timestep)
  - 24-hour allocation schedule plots
  - Reward breakdown pie/bar charts
  - Cost vs energy tradeoff curves (ablation)
  - Comparison bar charts vs baselines
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data_utils import MACHINE_DEFS, ClusterParams, generate_workload_sequence

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.size"] = 11

# Colour palette for machine types
MACHINE_COLORS = {
    "general": "#0072B2",
    "compute_opt": "#D55E00",
    "memory_opt": "#CC79A7",
    "gpu": "#F0E442",
    "storage_opt": "#009E73",
}

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES_DIR = PROJECT / "results" / "figures"


def plot_learning_curve(
    eval_log_path: Path,
    save_path: Optional[Path] = None,
    title: str = "PPO Learning Curve",
):
    """
    Plot reward vs timestep from SB3 eval logs.

    Expects 'evaluations.npz' from EvalCallback in eval_log_path.
    """
    try:
        data = np.load(eval_log_path / "evaluations.npz")
    except FileNotFoundError:
        print(f"No evaluations.npz found in {eval_log_path}")
        return

    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_evals, n_eval_episodes)

    mean = results.mean(axis=1)
    std = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean, color="#0072B2", linewidth=1.5, label="Mean reward")
    ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color="#0072B2")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episodic Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved learning curve to {save_path}")
    plt.close(fig)


def plot_allocation_schedule(
    allocation_history: List[Dict[str, float]],
    cpu_demand_history: List[float],
    mem_demand_history: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    title: str = "24-Hour Resource Allocation Schedule",
):
    """
    Plot stacked bar chart of machine allocation over a period.

    Parameters
    ----------
    allocation_history : list of dict
        Each dict maps machine type name -> instances allocated.
    cpu_demand_history : list of float
        CPU demand (vCPU cores) for each timestep.
    mem_demand_history : list of float, optional
        Memory demand (GB) for each timestep.
    """
    n_hours = len(allocation_history)
    hours = np.arange(n_hours)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Stacked bars (instances allocated)
    bottom = np.zeros(n_hours)
    type_names = [m["name"] for m in MACHINE_DEFS]

    for type_name in type_names:
        values = np.array([d.get(type_name, 0.0) for d in allocation_history])
        ax.bar(
            hours,
            values,
            bottom=bottom,
            label=type_name.replace("_", " ").title(),
            color=MACHINE_COLORS.get(type_name, "#888888"),
            width=0.8,
            alpha=0.85,
        )
        bottom += values

    ax.set_xlabel("Hour")
    ax.set_ylabel("Instances Allocated")
    ax.set_title(title)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in hours], rotation=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved allocation schedule to {save_path}")
    plt.close(fig)


def plot_reward_breakdown(
    reward_breakdown: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Reward Breakdown by Component",
):
    """
    Plot the contribution of each reward component over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Stacked area of individual components
    ax = axes[0]
    components = ["cost", "energy", "unmet_cpu", "unmet_mem", "stranded"]
    colors = ["#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#999999"]
    x = np.arange(len(reward_breakdown[components[0]]))
    bottom = np.zeros_like(x, dtype=float)

    for comp, color in zip(components, colors):
        vals = np.array(reward_breakdown[comp])
        ax.fill_between(
            x,
            bottom,
            bottom + vals,
            label=comp.replace("_", " ").title(),
            alpha=0.7,
            color=color,
        )
        bottom += vals

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward Component")
    ax.set_title(f"{title} — Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Total reward
    ax = axes[1]
    total = np.array(reward_breakdown["total"])
    ax.plot(x, total, color="#0072B2", linewidth=1.5)
    ax.fill_between(x, 0, total, alpha=0.15, color="#0072B2")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"{title} — Total")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved reward breakdown to {save_path}")
    plt.close(fig)


def plot_comparison(
    metrics: List[Dict],
    metric_key: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    title: str = "Policy Comparison",
):
    """
    Bar chart comparing policies on a given metric.
    """
    policies = [m["policy"] for m in metrics]
    values = [m[metric_key] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        policies,
        values,
        color=["#0072B2", "#D55E00", "#009E73", "#E69F00"],
        edgecolor="white",
        linewidth=1.2,
    )

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    plt.close(fig)


def plot_emissions_tradeoff(
    lambda_values: List[float],
    cost_metrics: List[float],
    emissions_metrics: List[float],
    save_path: Optional[Path] = None,
    title: str = "Cost vs Emissions Tradeoff",
):
    """
    Plot the Pareto frontier of cost vs emissions as λ varies.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_cost = "#D55E00"
    color_emissions = "#009E73"

    ax1.set_xlabel("Emissions weight λ")
    ax1.set_ylabel("Cost ($/MWh)", color=color_cost)
    ax1.plot(
        lambda_values, cost_metrics, "o-", color=color_cost, linewidth=2, label="Cost"
    )
    ax1.tick_params(axis="y", labelcolor=color_cost)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Emissions (kg CO₂/MWh)", color=color_emissions)
    ax2.plot(
        lambda_values,
        emissions_metrics,
        "s--",
        color=color_emissions,
        linewidth=2,
        label="Emissions",
    )
    ax2.tick_params(axis="y", labelcolor=color_emissions)

    ax1.set_title(title)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved tradeoff curve to {save_path}")
    plt.close(fig)


def generate_all_figures(
    results_dir: Path = FIGURES_DIR,
    eval_log_path: Optional[Path] = None,
):
    """Generate all standard figures."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Learning curve (if eval logs exist)
    if eval_log_path and eval_log_path.exists():
        plot_learning_curve(
            eval_log_path,
            save_path=results_dir / "learning_curve.png",
        )

    # 2. Comparison chart (if metrics exist)
    metrics_path = results_dir.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        plot_comparison(
            metrics,
            "mean_reward",
            "Mean Episodic Reward",
            save_path=results_dir / "comparison_reward.png",
        )
        plot_comparison(
            metrics,
            "mean_cpu_reliability",
            "CPU Reliability",
            save_path=results_dir / "comparison_cpu_reliability.png",
        )
        plot_comparison(
            metrics,
            "mean_cost_per_hour",
            "Cost ($/hr)",
            save_path=results_dir / "comparison_cost.png",
        )
        plot_comparison(
            metrics,
            "mean_energy_kw",
            "Energy (kW)",
            save_path=results_dir / "comparison_energy.png",
        )

    print("All figures generated.")


if __name__ == "__main__":
    # Quick test
    from data_utils import generate_workload_sequence

    scenarios = generate_workload_sequence(24, seed=42)
    cpu = [s.cpu_demand for s in scenarios]
    mem = [s.mem_demand for s in scenarios]
    print(
        f"Sample workload — CPU: {cpu[0]:.0f} ± {np.std(cpu):.0f} vCores, "
        f"Mem: {mem[0]:.0f} ± {np.std(mem):.0f} GB"
    )
    print("Data center visualization utilities ready.")
