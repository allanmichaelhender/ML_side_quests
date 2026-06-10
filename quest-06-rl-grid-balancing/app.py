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
VECNORM_PATH = RESULTS_DIR / "vecnormalize.pkl"
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
    """Load the trained PPO model + VecNormalize stats (cached in memory)."""
    if not MODEL_PATH.exists():
        return None, None
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    model = PPO.load(str(MODEL_PATH))
    vec_norm = None
    if VECNORM_PATH.exists():
        from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
        from src.cluster_env import make_env

        dummy = DummyVecEnv([lambda: make_env(episode_length=1, seed=0)])
        vec_norm = VecNormalize.load(str(VECNORM_PATH), venv=dummy)
        vec_norm.training = False
        vec_norm.norm_reward = False
    return model, vec_norm


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
    vec_norm=None,
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
    gpu_demand_history = []
    iops_demand_history = []
    power_budget_history = []
    carbon_history = []
    total_reward = 0.0

    # Track provisioned resources
    cpu_provided = []
    mem_provided = []
    gpu_provided = []
    iops_provided = []
    power_used = []

    for step in range(hours):
        # Get allocation action
        if model is not None:
            normed_obs = obs[np.newaxis, :]
            if vec_norm is not None:
                normed_obs = vec_norm.normalize_obs(normed_obs)
            action, _ = model.predict(normed_obs, deterministic=True)
            action = action[0]
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Convert action to actual allocations
        s = env._last_scenario
        alloc = {}
        total_cpu = 0.0
        total_mem = 0.0
        total_gpu = 0.0
        total_iops = 0.0
        total_power_w = 0.0

        for i, type_name in enumerate(env.type_names):
            m = MACHINE_DEFS[i]
            avail = s.machine_availability[type_name] * m["max_instances"]
            n_instances = action[i] * avail
            alloc[type_name] = round(n_instances, 1)
            total_cpu += n_instances * m["cpu_per_instance"]
            total_mem += n_instances * m["mem_per_instance"]
            total_gpu += n_instances * m.get("gpu_per_instance", 0)
            total_iops += n_instances * m.get("iops_per_instance", 0)
            total_power_w += n_instances * m["power_watts"]

        allocation_history.append(alloc)
        cpu_demand_history.append(round(s.cpu_demand, 1))
        mem_demand_history.append(round(s.mem_demand, 1))
        gpu_demand_history.append(s.gpu_demand)
        iops_demand_history.append(round(s.storage_iops_demand, 1))
        power_budget_history.append(round(s.power_budget_kw, 2))
        carbon_history.append(round(s.carbon_intensity, 4))
        cpu_provided.append(round(total_cpu, 1))
        mem_provided.append(round(total_mem, 1))
        gpu_provided.append(round(total_gpu, 1))
        iops_provided.append(round(total_iops, 1))
        power_used.append(round(total_power_w / 1000.0, 2))  # kW

    env.close()
    return {
        "allocation_history": allocation_history,
        "cpu_demand_history": cpu_demand_history,
        "mem_demand_history": mem_demand_history,
        "gpu_demand_history": gpu_demand_history,
        "iops_demand_history": iops_demand_history,
        "power_budget_history": power_budget_history,
        "carbon_history": carbon_history,
        "cpu_provided": cpu_provided,
        "mem_provided": mem_provided,
        "gpu_provided": gpu_provided,
        "iops_provided": iops_provided,
        "power_used": power_used,
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

    # Auto-run on first load with default 25 hours
    if "auto_ran" not in st.session_state:
        st.session_state.auto_ran = False

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        episode_hours = st.slider("Simulation hours", 6, 168, 24, step=6)
    with col2:
        pass  # placeholder
    with col3:
        use_trained = st.checkbox("Use trained agent", value=MODEL_PATH.exists())

    model, vec_norm = load_model() if use_trained else (None, None)

    run_clicked = st.button("▶ Run Allocation", type="primary")

    if run_clicked or not st.session_state.auto_ran:
        st.session_state.auto_ran = True
        with st.spinner("Running resource allocation simulation..."):
            result = run_allocation_episode(
                model,
                vec_norm=vec_norm,
                hours=episode_hours,
            )
        st.session_state.last_result = result

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        ah = result["allocation_history"]
        cpu_demand = result["cpu_demand_history"]
        cpu_provided = result["cpu_provided"]
        gpu_demand = result["gpu_demand_history"]
        gpu_provided = result["gpu_provided"]
        iops_demand = result["iops_demand_history"]
        iops_provided = result["iops_provided"]
        power_budget = result["power_budget_history"]
        power_used = result["power_used"]

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

        # ── Demand vs Provisioned plots ──────────────────────────────────
        fig2, axs = plt.subplots(2, 2, figsize=(14, 8))

        # CPU demand vs provided
        axs[0, 0].plot(hours, cpu_demand, "r-", label="Demand", linewidth=2)
        axs[0, 0].plot(hours, cpu_provided, "b--", label="Provided", linewidth=2)
        axs[0, 0].fill_between(
            hours,
            cpu_provided,
            cpu_demand,
            where=np.array(cpu_provided) < np.array(cpu_demand),
            color="r",
            alpha=0.15,
            label="Unmet",
        )
        axs[0, 0].set_title("CPU (vCores)")
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].set_xticks(hours[:: max(1, len(hours) // 12)])
        axs[0, 0].tick_params(axis="x", rotation=45, labelsize=8)

        # GPU demand vs provided
        axs[0, 1].plot(hours, gpu_demand, "r-", label="Demand", linewidth=2)
        axs[0, 1].plot(hours, gpu_provided, "b--", label="Provided", linewidth=2)
        axs[0, 1].fill_between(
            hours,
            gpu_provided,
            gpu_demand,
            where=np.array(gpu_provided) < np.array(gpu_demand),
            color="r",
            alpha=0.15,
            label="Unmet",
        )
        axs[0, 1].set_title("GPU Tasks")
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_xticks(hours[:: max(1, len(hours) // 12)])
        axs[0, 1].tick_params(axis="x", rotation=45, labelsize=8)

        # IOPS demand vs provided
        axs[1, 0].plot(hours, iops_demand, "r-", label="Demand", linewidth=2)
        axs[1, 0].plot(hours, iops_provided, "b--", label="Provided", linewidth=2)
        axs[1, 0].set_title("Storage IOPS")
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_xticks(hours[:: max(1, len(hours) // 12)])
        axs[1, 0].tick_params(axis="x", rotation=45, labelsize=8)

        # Power usage vs budget
        axs[1, 1].plot(hours, power_budget, "g-", label="Budget", linewidth=2)
        axs[1, 1].plot(hours, power_used, "orange", label="Used", linewidth=2)
        axs[1, 1].fill_between(
            hours,
            power_used,
            power_budget,
            where=np.array(power_used) > np.array(power_budget),
            color="r",
            alpha=0.15,
            label="Over budget",
        )
        axs[1, 1].set_title("Power (kW)")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].set_xticks(hours[:: max(1, len(hours) // 12)])
        axs[1, 1].tick_params(axis="x", rotation=45, labelsize=8)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Summary metrics ────────────────────────────────────────────────
        st.subheader("Summary")
        total_instances = sum(sum(d.values()) for d in ah)

        # Cost estimate
        total_cost = 0.0
        total_carbon = 0.0
        for i, d in enumerate(ah):
            for type_name, instances in d.items():
                m_def = next(m for m in MACHINE_DEFS if m["name"] == type_name)
                total_cost += instances * m_def["cost_per_hour"]
            total_carbon += power_used[i] * result["carbon_history"][i]

        avg_cpu_rel = np.mean(
            [1.0 if p >= d else 0.0 for p, d in zip(cpu_provided, cpu_demand)]
        )
        avg_gpu_rel = np.mean(
            [1.0 if p >= d else 0.0 for p, d in zip(gpu_provided, gpu_demand)]
        )
        avg_power_violation = np.mean(
            [max(0, u - b) for u, b in zip(power_used, power_budget)]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CPU Reliability", f"{avg_cpu_rel:.1%}")
        col2.metric("GPU Reliability", f"{avg_gpu_rel:.1%}")
        col3.metric("Total Cost", f"${total_cost:,.0f}")
        col4.metric("Total Carbon", f"{total_carbon:,.1f} kg CO₂")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Instances Used", f"{total_instances:,.0f}")
        col6.metric("Avg Power", f"{np.mean(power_used):.1f} kW")
        col7.metric("Avg Power Overuse", f"{avg_power_violation:.2f} kW")
        col8.metric(
            "Avg Carbon Intensity", f"{np.mean(result['carbon_history']):.3f} kg/kWh"
        )

        # ── Machine mix pie chart ──────────────────────────────────────────
        st.subheader("Machine Allocation Mix")
        total_by_type = {
            type_name: sum(d.get(type_name, 0.0) for d in ah)
            for type_name in type_names
        }

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        labels = [
            k.replace("_", " ").title() for k, v in total_by_type.items() if v > 0
        ]
        sizes = [v for v in total_by_type.values() if v > 0]
        colors = [
            MACHINE_COLORS.get(k, "#888") for k, v in total_by_type.items() if v > 0
        ]
        ax3.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax3.set_title("Total Instance-Hours by Machine Type")
        st.pyplot(fig3)
        plt.close(fig3)


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
        rows = []
        for m in metrics:
            row = {
                "Policy": m["policy"],
                "Reward": m["mean_reward"],
                "CPU Rel.": f"{m.get('mean_cpu_reliability', 0):.3f}",
                "Mem Rel.": f"{m.get('mean_mem_reliability', 0):.3f}",
                "GPU Rel.": f"{m.get('mean_gpu_reliability', 0):.3f}",
                "IOPS Rel.": f"{m.get('mean_iops_reliability', 0):.3f}",
                "Cost ($/hr)": m.get("mean_cost_per_hour", 0),
                "Energy (kW)": m.get("mean_energy_kw", 0),
                "Carbon (kg)": m.get("mean_carbon_kg", 0),
                "Power Over. (kW)": m.get("mean_power_overuse_kw", 0),
            }
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)

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

    model, vec_norm = load_model()

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
                vec_norm=vec_norm,
                hours=24,
                seed=seed_map.get(scenario_type, 42),
            )

        st.subheader(f"{scenario_type} — Total Reward: {result['total_reward']:.2f}")

        ah = result["allocation_history"]
        cpu_demand = result["cpu_demand_history"]
        cpu_provided = result["cpu_provided"]
        gpu_provided = result["gpu_provided"]
        gpu_demand = result["gpu_demand_history"]
        power_used = result["power_used"]
        power_budget = result["power_budget_history"]
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
        total_carbon = 0.0
        for i, d in enumerate(ah):
            for type_name, instances in d.items():
                m_def = next(m for m in MACHINE_DEFS if m["name"] == type_name)
                total_cost += instances * m_def["cost_per_hour"]
            total_carbon += power_used[i] * result["carbon_history"][i]

        avg_gpu_rel = np.mean(
            [1.0 if p >= d else 0.0 for p, d in zip(gpu_provided, gpu_demand)]
        )
        avg_power_violation = np.mean(
            [max(0, u - b) for u, b in zip(power_used, power_budget)]
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Instances Used", f"{total_instances:,.0f}")
        col2.metric("Total Cost", f"${total_cost:,.0f}")
        col3.metric("Total Carbon", f"{total_carbon:,.1f} kg CO₂")

        col4, col5, col6 = st.columns(3)
        col4.metric("GPU Reliability", f"{avg_gpu_rel:.1%}")
        col5.metric("Avg Power", f"{np.mean(power_used):.1f} kW")
        col6.metric("Avg Power Overuse", f"{avg_power_violation:.2f} kW")


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
    - Minimise energy consumption & carbon emissions
    - Maximise CPU, memory, GPU & IOPS reliability
    - Penalise over-provisioning (stranded capacity)
    - Penalise exceeding power budget
    - Carbon-aware scheduling (cleaner power during the day)
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
