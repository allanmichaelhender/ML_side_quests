"""
Streamlit app for Data Center Resource Scheduling Agent.

Tabs:
  - Live Allocation   — watch the agent allocate machines in real-time
  - Results Dashboard — view training/evaluation findings
  - What-If Scenarios — explore how the agent handles different conditions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Patch

from src.data_utils import (
    MACHINE_DEFS,
    ClusterParams,
    generate_workload_sequence,
)
from src.cluster_env import ClusterDispatchEnv, make_env
from src.visualize import (
    MACHINE_COLORS,
    plot_allocation_schedule,
    plot_comparison,
)

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.json"
MODEL_PATH = RESULTS_DIR / "model.zip"
TRAINING_META = RESULTS_DIR / "training_metadata.json"
EVAL_RESULTS = RESULTS_DIR / "eval_results.json"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Center RL Scheduler",
    page_icon="🖥️",
    layout="wide",
)

st.title("🖥️ Data Center Resource Scheduling Agent")
st.markdown(
    "A **PPO** reinforcement learning agent trained to allocate "
    "machine instances (general, compute-optimised, memory-optimised, GPU, "
    "storage-optimised) to meet computing workload demand at minimum cost, "
    "energy, and maximum reliability."
)

# ── Helpers ────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """Load the trained PPO model (cached in memory)."""
    if not MODEL_PATH.exists():
        return None
    from stable_baselines3 import PPO

    return PPO.load(str(MODEL_PATH))


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_training_meta():
    if not TRAINING_META.exists():
        return None
    with open(TRAINING_META) as f:
        return json.load(f)


def run_allocation_episode(
    model,
    hours: int = 24,
    seed: int = 100,
) -> Dict:
    """Run an allocation episode and return full history."""
    scenarios = generate_workload_sequence(
        n_hours=hours,
        seed=seed,
    )
    env = make_env(workload_sequence=scenarios, episode_length=hours, seed=seed)

    obs, _ = env.reset()
    allocation_history = []
    cpu_demand_history = []
    mem_demand_history = []
    total_reward = 0.0

    for step in range(hours):
        # Get allocation action
        if model is not None:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = action[0]
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Record state
        s = env._last_scenario
        alloc = {}
        for i, type_name in enumerate(env.type_names):
            avail = s.machine_availability[type_name] * MACHINE_DEFS[i]["max_instances"]
            alloc[type_name] = round(action[i] * avail, 1)

        allocation_history.append(alloc)
        cpu_demand_history.append(round(s.cpu_demand, 1))
        mem_demand_history.append(round(s.mem_demand, 1))

    env.close()
    return {
        "allocation_history": allocation_history,
        "cpu_demand_history": cpu_demand_history,
        "mem_demand_history": mem_demand_history,
        "total_reward": round(total_reward, 2),
    }


# ── Tab: Live Dispatch ─────────────────────────────────────────────────────

tab_live, tab_results, tab_whatif = st.tabs(
    [
        "🎮 Live Dispatch",
        "📊 Results Dashboard",
        "🔮 What-If Scenarios",
    ]
)

with tab_live:
    st.header("🎮 Live Resource Allocation")
    st.markdown(
        "Watch the RL agent allocate machine instances hour-by-hour. "
        "Select the workload scenario below."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        episode_hours = st.slider("Simulation hours", 6, 168, 24, step=6)
    with col2:
        pass  # placeholder
    with col3:
        use_trained = st.checkbox("Use trained agent", value=MODEL_PATH.exists())

    model = load_model() if use_trained else None

    if st.button("▶ Run Allocation", type="primary"):
        with st.spinner("Running resource allocation simulation..."):
            result = run_allocation_episode(
                model,
                hours=episode_hours,
            )

        ah = result["allocation_history"]
        cpu_demand = result["cpu_demand_history"]

        st.subheader(f"Allocation over {len(ah)} hours")
        st.metric("Total Reward", f"{result['total_reward']:.2f}")

        # ── Allocation schedule plot ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        hours = np.arange(len(ah))
        bottom = np.zeros(len(ah))
        type_names = [m["name"] for m in MACHINE_DEFS]

        for type_name in type_names:
            values = np.array([d.get(type_name, 0.0) for d in ah])
            ax.bar(
                hours,
                values,
                bottom=bottom,
                label=type_name.replace("_", " ").title(),
                color=MACHINE_COLORS.get(type_name, "#888"),
                width=0.8,
                alpha=0.85,
            )
            bottom += values

        ax.set_xlabel("Hour")
        ax.set_ylabel("Instances Allocated")
        ax.set_title("Resource Allocation Schedule")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h % 24:02d}:00" for h in hours], rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Summary metrics ────────────────────────────────────────────────
        st.subheader("Summary")
        total_instances = sum(sum(d.values()) for d in ah)
        avg_cpu_demand = np.mean(cpu_demand)

        # Cost estimate
        total_cost = 0.0
        for i, d in enumerate(ah):
            for type_name, instances in d.items():
                m_def = next(m for m in MACHINE_DEFS if m["name"] == type_name)
                total_cost += instances * m_def["cost_per_hour"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg CPU Demand", f"{avg_cpu_demand:,.0f} vCores")
        col2.metric("Total Instances Used", f"{total_instances:,.0f}")
        col3.metric("Total Cost", f"${total_cost:,.0f}")
        col4.metric("Est. Energy", f"{total_cost * 0.5:,.0f} kWh")

        # ── Machine mix pie chart ──────────────────────────────────────────
        st.subheader("Machine Allocation Mix")
        total_by_type = {
            type_name: sum(d.get(type_name, 0.0) for d in ah)
            for type_name in type_names
        }

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        labels = [
            k.replace("_", " ").title() for k, v in total_by_type.items() if v > 0
        ]
        sizes = [v for v in total_by_type.values() if v > 0]
        colors = [
            MACHINE_COLORS.get(k, "#888") for k, v in total_by_type.items() if v > 0
        ]
        ax2.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax2.set_title("Total Instance-Hours by Machine Type")
        st.pyplot(fig2)
        plt.close(fig2)


# ── Tab: Results Dashboard ─────────────────────────────────────────────────

with tab_results:
    st.header("📊 Results Dashboard")

    metrics = load_metrics()
    training_meta = load_training_meta()

    if training_meta:
        st.subheader("Training Configuration")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Algorithm", training_meta.get("algorithm", "PPO"))
        col2.metric("Total Timesteps", f"{training_meta.get('total_timesteps', 0):,}")
        col3.metric(
            "Training Time", f"{training_meta.get('training_time_minutes', 0)} min"
        )
        col4.metric("Learning Rate", training_meta.get("learning_rate", "N/A"))

    # Evaluation results
    eval_path = EVAL_RESULTS
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        st.subheader("Post-Training Evaluation")
        st.json(eval_data)

    # Comparison figures
    st.subheader("Policy Comparison")
    if FIGURES_DIR.exists():
        fig_files = {
            "Reward": FIGURES_DIR / "comparison_reward.png",
            "CPU Reliability": FIGURES_DIR / "comparison_cpu_reliability.png",
            "Cost": FIGURES_DIR / "comparison_cost.png",
            "Energy": FIGURES_DIR / "comparison_energy.png",
        }
        cols = st.columns(2)
        for i, (label, path) in enumerate(fig_files.items()):
            if path.exists():
                cols[i % 2].image(str(path), caption=label, use_container_width=True)

    # Learning curve
    lc_path = FIGURES_DIR / "learning_curve.png"
    if lc_path.exists():
        st.subheader("Learning Curve")
        st.image(str(lc_path), use_container_width=True)

    # Metrics table
    if metrics:
        st.subheader("Comparison Metrics")
        st.dataframe(
            [
                {
                    "Policy": m["policy"],
                    "Mean Reward": m["mean_reward"],
                    "CPU Reliability": f"{m['mean_cpu_reliability']:.3f}",
                    "Mem Reliability": f"{m['mean_mem_reliability']:.3f}",
                    "Cost ($/hr)": m["mean_cost_per_hour"],
                    "Energy (kW)": m["mean_energy_kw"],
                }
                for m in metrics
            ],
            use_container_width=True,
            hide_index=True,
        )

    # Raw metrics JSON
    with st.expander("Raw metrics JSON"):
        if metrics:
            st.json(metrics)


# ── Tab: What-If Scenarios ────────────────────────────────────────────────

with tab_whatif:
    st.header("🔮 What-If Scenarios")
    st.markdown("Explore how the agent handles different data center conditions.")

    col1, col2 = st.columns(2)
    with col1:
        scenario_type = st.selectbox(
            "Scenario type",
            [
                "Normal conditions",
                "Peak workload (high CPU demand)",
                "Memory burst (high memory demand)",
                "GPU cluster outage",
                "Multi-type failure event",
            ],
        )
    with col2:
        pass

    model = load_model()

    if st.button("▶ Run Scenario", type="primary"):
        if model is None:
            st.warning("No trained model found. Using random policy instead.")

        seed_map = {
            "Normal conditions": 42,
            "Peak workload (high CPU demand)": 43,
            "Memory burst (high memory demand)": 44,
            "GPU cluster outage": 45,
            "Multi-type failure event": 46,
        }

        with st.spinner(f"Running {scenario_type}..."):
            result = run_allocation_episode(
                model,
                hours=24,
                seed=seed_map.get(scenario_type, 42),
            )

        st.subheader(f"{scenario_type} — Total Reward: {result['total_reward']:.2f}")

        ah = result["allocation_history"]
        cpu_demand = result["cpu_demand_history"]
        type_names = [m["name"] for m in MACHINE_DEFS]

        fig, ax = plt.subplots(figsize=(14, 6))
        hours = np.arange(len(ah))
        bottom = np.zeros(len(ah))

        for type_name in type_names:
            values = np.array([d.get(type_name, 0.0) for d in ah])
            ax.bar(
                hours,
                values,
                bottom=bottom,
                label=type_name.replace("_", " ").title(),
                color=MACHINE_COLORS.get(type_name, "#888"),
                width=0.8,
                alpha=0.85,
            )
            bottom += values

        ax.set_xlabel("Hour")
        ax.set_ylabel("Instances")
        ax.set_title(f"Allocation — {scenario_type}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        total_instances = sum(sum(d.values()) for d in ah)
        total_cost = 0.0
        for i, d in enumerate(ah):
            for type_name, instances in d.items():
                m_def = next(m for m in MACHINE_DEFS if m["name"] == type_name)
                total_cost += instances * m_def["cost_per_hour"]

        st.metric("Total Instances Used", f"{total_instances:,.0f}")
        st.metric("Total Cost", f"${total_cost:,.0f}")


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.header("About")
st.sidebar.markdown(
    """
    **Quest 6 — Data Center RL Scheduling**

    A PPO agent trained with Stable Baselines3 to allocate
    a data center cluster with 5 machine types.

    **Machine types:**
    - 🔵 General-purpose (balanced CPU/memory)
    - 🟠 Compute-optimised (high vCPU count)
    - 🟣 Memory-optimised (high RAM)
    - 🟡 GPU / Accelerator
    - 🟢 Storage-optimised (high I/O)

    **Reward function:**
    - Minimise compute cost
    - Minimise energy consumption
    - Maximise CPU & memory reliability
    - Penalise over-provisioning (stranded capacity)
    """
)

model_status = "✅ Loaded" if MODEL_PATH.exists() else "❌ Not trained"
st.sidebar.markdown(f"**Model status:** {model_status}")

if MODEL_PATH.exists():
    meta = load_training_meta()
    if meta:
        st.sidebar.markdown(f"**Timesteps:** {meta.get('total_timesteps', 0):,}")
        st.sidebar.markdown(
            f"**Training time:** {meta.get('training_time_minutes', 0)} min"
        )
