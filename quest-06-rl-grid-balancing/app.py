"""
Streamlit app for Energy Grid Load Balancing Agent.

Tabs:
  - Live Dispatch    — watch the agent dispatch the grid in real-time
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
    SOURCE_DEFS,
    GridParams,
    generate_scenario_sequence,
    sample_grid_params,
)
from src.grid_env import GridDispatchEnv, make_env
from src.visualize import (
    SOURCE_COLORS,
    plot_dispatch_schedule,
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
    page_title="Grid Balancing RL Agent",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Grid Load Balancing Agent")
st.markdown(
    "A **PPO** reinforcement learning agent trained to dispatch power from "
    "multiple generation sources (coal, gas, solar, wind, hydro) to meet "
    "electricity demand at minimum cost, emissions, and maximum reliability."
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


def run_dispatch_episode(
    model,
    hours: int = 24,
    season: str = "summer",
    seed: int = 100,
) -> Dict:
    """Run a dispatch episode and return full history."""
    scenarios = generate_scenario_sequence(
        n_hours=hours, seed=seed, start_season=season
    )
    env = make_env(scenario_sequence=scenarios, episode_length=hours, seed=seed)

    obs, _ = env.reset()
    dispatch_history = []
    demand_history = []
    reward_components = {"cost": [], "emissions": [], "unmet": [], "overgen": []}
    total_reward = 0.0

    for step in range(hours):
        # Get dispatch action
        if model is not None:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = action[0]
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Record state
        s = env._last_scenario
        dispatch = {}
        for i, src_name in enumerate(env.source_names):
            avail_mw = s.source_availability[src_name] * SOURCE_DEFS[i]["max_cap"]
            dispatch[src_name] = round(action[i] * avail_mw, 1)

        dispatch_history.append(dispatch)
        demand_history.append(round(s.demand_mw, 1))

    env.close()
    return {
        "dispatch_history": dispatch_history,
        "demand_history": demand_history,
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
    st.header("🎮 Live Grid Dispatch")
    st.markdown(
        "Watch the RL agent dispatch the grid hour-by-hour. Select season and "
        "demand scenario below."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_season = st.selectbox(
            "Season",
            ["spring", "summer", "fall", "winter"],
            index=1,
        )
    with col2:
        episode_hours = st.slider("Simulation hours", 6, 168, 24, step=6)
    with col3:
        use_trained = st.checkbox("Use trained agent", value=MODEL_PATH.exists())

    model = load_model() if use_trained else None

    if st.button("▶ Run Dispatch", type="primary"):
        with st.spinner("Running grid dispatch simulation..."):
            result = run_dispatch_episode(
                model,
                hours=episode_hours,
                season=selected_season,
            )

        dh = result["dispatch_history"]
        dm = result["demand_history"]

        st.subheader(
            f"Dispatch over {len(dh)} hours (Season: {selected_season.capitalize()})"
        )
        st.metric("Total Reward", f"{result['total_reward']:.2f}")

        # ── Dispatch schedule plot ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        hours = np.arange(len(dh))
        bottom = np.zeros(len(dh))
        source_names = [s["name"] for s in SOURCE_DEFS]

        for src_name in source_names:
            values = np.array([d.get(src_name, 0.0) for d in dh])
            ax.bar(
                hours,
                values,
                bottom=bottom,
                label=src_name.capitalize(),
                color=SOURCE_COLORS.get(src_name, "#888"),
                width=0.8,
            )
            bottom += values

        ax.plot(
            hours,
            dm,
            color="red",
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Demand",
            linestyle="--",
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")
        ax.set_title(f"Dispatch Schedule — {selected_season.capitalize()}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h % 24:02d}:00" for h in hours], rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Summary metrics ────────────────────────────────────────────────
        st.subheader("Summary")
        total_dispatch = sum(sum(d.values()) for d in dh)
        total_demand = sum(dm)
        unmet = max(0, total_demand - total_dispatch)

        # Cost estimate
        total_cost = 0.0
        for i, d in enumerate(dh):
            for src_name, mw in d.items():
                src_def = next(s for s in SOURCE_DEFS if s["name"] == src_name)
                total_cost += mw * src_def["opex_var"]

        reliability = (1 - unmet / total_demand) * 100 if total_demand > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Demand", f"{total_demand:,.0f} MWh")
        col2.metric("Total Dispatch", f"{total_dispatch:,.0f} MWh")
        col3.metric("Reliability", f"{reliability:.1f}%")
        col4.metric("Est. Cost", f"${total_cost:,.0f}")

        # ── Generation mix pie chart ───────────────────────────────────────
        st.subheader("Generation Mix")
        total_by_source = {
            src_name: sum(d.get(src_name, 0.0) for d in dh) for src_name in source_names
        }

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        labels = [k.capitalize() for k, v in total_by_source.items() if v > 0]
        sizes = [v for v in total_by_source.values() if v > 0]
        colors = [
            SOURCE_COLORS.get(k, "#888") for k, v in total_by_source.items() if v > 0
        ]
        ax2.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax2.set_title("Total Energy by Source")
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
            "Reliability": FIGURES_DIR / "comparison_reliability.png",
            "Cost": FIGURES_DIR / "comparison_cost.png",
            "Emissions": FIGURES_DIR / "comparison_emissions.png",
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
                    "Reliability": f"{m['mean_reliability']:.3f}",
                    "Cost ($/MWh)": m["mean_cost_per_mwh"],
                    "Emissions (kg/MWh)": m["mean_emissions_per_mwh"],
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
    st.markdown(
        "Explore how the agent handles different grid conditions. Adjust "
        "parameters below and compare the agent's response across seasons."
    )

    col1, col2 = st.columns(2)
    with col1:
        scenario_type = st.selectbox(
            "Scenario type",
            [
                "Normal conditions",
                "Heatwave (peak demand)",
                "Cloudy (low solar)",
                "Wind drought",
                "Hydro outage",
            ],
        )
    with col2:
        compare_seasons = st.multiselect(
            "Compare across seasons",
            ["spring", "summer", "fall", "winter"],
            default=["summer", "winter"],
        )

    model = load_model()

    if st.button("▶ Run Comparison", type="primary"):
        if model is None:
            st.warning("No trained model found. Using random policy instead.")

        results_by_season = {}
        for season in compare_seasons:
            with st.spinner(f"Running {season}..."):
                result = run_dispatch_episode(
                    model,
                    hours=24,
                    season=season,
                )
                results_by_season[season] = result

        # Comparative dispatch plots
        fig, axes = plt.subplots(
            len(compare_seasons),
            1,
            figsize=(14, 5 * len(compare_seasons)),
            squeeze=False,
        )
        source_names = [s["name"] for s in SOURCE_DEFS]

        for idx, season in enumerate(compare_seasons):
            ax = axes[idx, 0]
            dh = results_by_season[season]["dispatch_history"]
            dm = results_by_season[season]["demand_history"]
            hours = np.arange(len(dh))
            bottom = np.zeros(len(dh))

            for src_name in source_names:
                values = np.array([d.get(src_name, 0.0) for d in dh])
                ax.bar(
                    hours,
                    values,
                    bottom=bottom,
                    label=src_name.capitalize(),
                    color=SOURCE_COLORS.get(src_name, "#888"),
                    width=0.8,
                )
                bottom += values

            ax.plot(
                hours,
                dm,
                color="red",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Demand",
                linestyle="--",
            )
            ax.set_title(
                f"{season.capitalize()} — Total Reward: {results_by_season[season]['total_reward']:.2f}"
            )
            ax.set_ylabel("MW")
            ax.set_xticks(hours)
            ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Season comparison table
        st.subheader("Season Comparison")
        comparison_data = []
        for season in compare_seasons:
            r = results_by_season[season]
            total_dispatch = sum(sum(d.values()) for d in r["dispatch_history"])
            total_demand = sum(r["demand_history"])
            reliability = (
                (1 - max(0, total_demand - total_dispatch) / total_demand) * 100
                if total_demand > 0
                else 0
            )
            comparison_data.append(
                {
                    "Season": season.capitalize(),
                    "Total Reward": r["total_reward"],
                    "Reliability": f"{reliability:.1f}%",
                    "Total Demand": f"{total_demand:,.0f} MWh",
                }
            )

        st.dataframe(comparison_data, use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.header("About")
st.sidebar.markdown(
    """
    **Quest 6 — RL Grid Balancing**

    A PPO agent trained with Stable Baselines3 to dispatch
    an energy grid with 5 generation sources.

    **Sources:**
    - 🟤 Coal (baseload, cheap but dirty)
    - 🟡 Gas (flexible, moderate cost)
    - 🟡 Solar (free fuel, weather-dependent)
    - 🔵 Wind (free fuel, variable)
    - 🟢 Hydro (clean, stable)

    **Reward function:**
    - Minimise dispatch cost
    - Minimise CO₂ emissions
    - Maximise reliability (meet demand)
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
