"""
Streamlit app for 3-Room Multi-Zone HVAC RL Agent.

Tabs:
  - Live Simulation   — watch the agent control temperatures in real-time
  - Results Dashboard — view training/evaluation findings
  - Scenario Comparison — compare agent behaviour across climates
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import BuildingConfig
from src.env import MultiRoomHVACEnv

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.json"
MODEL_PATH = RESULTS_DIR / "model.zip"
META_PATH = RESULTS_DIR / "training_meta.json"
EVAL_PATH = RESULTS_DIR / "eval_results.json"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Zone HVAC RL Agent",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Multi-Zone HVAC Reinforcement Learning Agent")
st.markdown(
    "A **DQN** agent controlling air conditioning in a 3-room building "
    "(living room, kitchen, bedroom) with inter-zone heat transfer, "
    "solar gain, and diurnal outdoor temperature cycles."
)

# ── Helpers ────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """Load the trained DQN model (cached in memory)."""
    if not MODEL_PATH.exists():
        return None
    from stable_baselines3 import DQN

    return DQN.load(str(MODEL_PATH))


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_training_meta():
    if not META_PATH.exists():
        return None
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_eval_results():
    if not EVAL_PATH.exists():
        return None
    with open(EVAL_PATH) as f:
        return json.load(f)


def run_simulation_episode(
    base_temp: float = 27.0,
    episode_steps: int = 72,
    use_model: bool = True,
) -> dict:
    """Run a full simulation episode and return all state/action/reward data."""
    cfg = BuildingConfig(episode_steps=episode_steps)
    env = MultiRoomHVACEnv(config=cfg)
    model = load_model() if use_model else None

    # Manually set base temp and reset
    env.base_temp = base_temp
    obs, _ = env.reset()
    env.base_temp = base_temp  # override randomisation

    history = {
        "hours": [],
        "temps": [],
        "outdoor": [],
        "actions": [],
        "rewards": [],
        "total_reward": 0.0,
    }

    for _ in range(episode_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        history["hours"].append(env.hour)
        history["temps"].append(obs[:3].tolist())
        history["outdoor"].append(float(obs[3]))
        history["actions"].append(MultiRoomHVACEnv._decode_action(int(action)).tolist())

        obs, reward, terminated, truncated, _ = env.step(action)
        history["rewards"].append(reward)
        history["total_reward"] += reward

    env.close()
    return history


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("🏠 HVAC Control")
st.sidebar.markdown("**Building Zones**")
st.sidebar.markdown("- 🪟 **Living Room** — South-facing, solar gain")
st.sidebar.markdown("- 🍳 **Kitchen** — Internal zone")
st.sidebar.markdown("- 🛏️ **Bedroom** — North-facing")

metrics = load_metrics()
if metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Training Summary**")
    st.sidebar.metric("Algorithm", metrics.get("algorithm", "DQN"))
    st.sidebar.metric(
        "Mean Eval Reward", f"{metrics.get('mean_episodic_reward', 0):.1f}"
    )
    st.sidebar.metric("Timesteps", f"{metrics.get('total_timesteps', 0):,}")

if MODEL_PATH.exists():
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Model loaded")

# ── Tabs ───────────────────────────────────────────────────────────────────

tab_sim, tab_results, tab_compare = st.tabs(
    [
        "🎮 Live Simulation",
        "📊 Results Dashboard",
        "🌡️ Scenario Comparison",
    ]
)

# ============================================================================
# TAB 1: LIVE SIMULATION
# ============================================================================

with tab_sim:
    st.header("🎮 Live HVAC Simulation")
    st.markdown(
        "Watch the RL agent control temperatures across three zones. "
        "Adjust the climate and episode length below."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        base_temp = st.slider("Outdoor base temperature (°C)", 20.0, 35.0, 27.0, 1.0)
    with col2:
        episode_days = st.slider("Simulation days", 1, 7, 3)
        episode_steps = episode_days * 24
    with col3:
        use_trained = st.checkbox("Use trained agent", value=MODEL_PATH.exists())

    run_clicked = st.button("▶ Run Simulation", type="primary")

    if run_clicked:
        with st.spinner("Running HVAC simulation..."):
            result = run_simulation_episode(
                base_temp=base_temp,
                episode_steps=episode_steps,
                use_model=use_trained,
            )
        st.session_state.sim_result = result

    if "sim_result" in st.session_state:
        result = st.session_state.sim_result
        hours = np.array(result["hours"])
        temps = np.array(result["temps"])
        outdoor = np.array(result["outdoor"])
        actions = np.array(result["actions"])
        rewards = np.array(result["rewards"])

        st.subheader(f"📈 Simulation over {len(hours)} hours")
        st.metric("Total Reward", f"{result['total_reward']:.2f}")

        # ── Temperature trace plot ──────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(14, 5))

        zone_colors = {"living": "#e74c3c", "kitchen": "#2ecc71", "bedroom": "#3498db"}
        zone_names = ["Living", "Kitchen", "Bedroom"]
        target_temps = [21.0, 22.0, 20.0]
        cfg_keys = ["living", "kitchen", "bedroom"]

        ax1.plot(
            hours,
            outdoor,
            "--",
            color="#f39c12",
            alpha=0.5,
            label="Outdoor",
            linewidth=1.5,
        )
        for i, (zn, zc) in enumerate(zip(zone_names, zone_colors.values())):
            ax1.plot(hours, temps[:, i], "-", color=zc, label=zn, linewidth=2)
            ax1.axhline(target_temps[i], color=zc, linestyle=":", alpha=0.3)

        # Day separators
        for day in range(1, episode_days):
            ax1.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.3)
            ax1.text(
                day * 24 - 12,
                ax1.get_ylim()[1] + 1,
                f"Day {day}",
                ha="center",
                fontsize=9,
                color="gray",
            )

        ax1.set_ylabel("Temperature (°C)")
        ax1.set_xlabel("Hour")
        ax1.set_title("Zone Temperature & Outdoor Temperature")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── AC activation heatmap ──────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        im = ax2.imshow(actions.T, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel("Hour")
        ax2.set_title("AC Activation (red = ON)")

        for day in range(1, episode_days):
            ax2.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.3)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Reward plot ────────────────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(14, 3))
        cumreward = np.cumsum(rewards)
        ax3.plot(hours, rewards, "-", color="#9b59b6", label="Step reward", alpha=0.7)
        ax3.fill_between(hours, 0, rewards, alpha=0.1, color="#9b59b6")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(hours, cumreward, "--", color="#2c3e50", label="Cumulative")
        ax3.set_ylabel("Step Reward")
        ax3.set_xlabel("Hour")
        ax3.set_title("Reward Progression")
        ax3.legend(loc="upper left", fontsize=8)
        ax3_twin.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── Summary metrics ────────────────────────────────────────────────
        st.subheader("📊 Zone Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reward", f"{result['total_reward']:.2f}")
        col2.metric("Total AC Hours", f"{int(np.sum(actions))}")
        col3.metric("Avg Outdoor", f"{np.mean(outdoor):.1f}°C")
        col4.metric("Peak Outdoor", f"{np.max(outdoor):.1f}°C")

        zone_metrics_cols = st.columns(3)
        for i, (zn, zc) in enumerate(zip(zone_names, zone_colors.keys())):
            with zone_metrics_cols[i]:
                avg_temp = np.mean(temps[:, i])
                comfort_dev = np.mean(np.abs(temps[:, i] - target_temps[i]))
                ac_pct = np.mean(actions[:, i]) * 100
                st.markdown(f"**{zn}**")
                st.metric("Avg Temp", f"{avg_temp:.1f}°C")
                st.metric("Avg Comfort Deviation", f"{comfort_dev:.2f}°C")
                st.metric("AC ON", f"{ac_pct:.1f}%")

        # ── Raw data expander ─────────────────────────────────────────────
        with st.expander("📋 View hourly data"):
            import pandas as pd

            df_data = {
                "Hour": result["hours"],
                "Outdoor": [f"{o:.1f}" for o in result["outdoor"]],
                "Living °C": [f"{t[0]:.1f}" for t in result["temps"]],
                "Kitchen °C": [f"{t[1]:.1f}" for t in result["temps"]],
                "Bedroom °C": [f"{t[2]:.1f}" for t in result["temps"]],
                "AC Living": [a[0] for a in result["actions"]],
                "AC Kitchen": [a[1] for a in result["actions"]],
                "AC Bedroom": [a[2] for a in result["actions"]],
                "Reward": [f"{r:.2f}" for r in result["rewards"]],
            }
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)


# ============================================================================
# TAB 2: RESULTS DASHBOARD
# ============================================================================

with tab_results:
    st.header("📊 Results Dashboard")

    eval_results = load_eval_results()
    tmetrics = load_metrics()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Details")
        meta = load_training_meta()
        if meta:
            meta_items = {
                "Algorithm": meta.get("algorithm", "N/A"),
                "Total Timesteps": f"{meta.get('total_timesteps', 0):,}",
                "Episode Length": f"{meta.get('episode_steps', 72)} steps (3 days)",
                "Gamma (γ)": meta.get("gamma", "N/A"),
                "Learning Rate": meta.get("learning_rate", "N/A"),
                "Buffer Size": f"{meta.get('buffer_size', 0):,}",
                "Batch Size": meta.get("batch_size", "N/A"),
                "Network Architecture": str(meta.get("net_arch", "N/A")),
                "Seed": meta.get("seed", "N/A"),
            }
            for k, v in meta_items.items():
                st.markdown(f"**{k}:** {v}")
        else:
            st.warning("No training metadata found. Run `python src/train.py` first.")

    with col2:
        st.subheader("Evaluation Metrics")
        if tmetrics:
            st.metric("Algorithm", tmetrics.get("algorithm", "N/A"))
            st.metric(
                "Mean Episodic Reward",
                f"{tmetrics.get('mean_episodic_reward', 0):.2f}",
            )
            st.metric(
                "Std Episodic Reward",
                f"{tmetrics.get('std_episodic_reward', 0):.2f}",
            )
            st.metric("Eval Episodes", tmetrics.get("n_eval_episodes", 0))
            st.metric(
                "Training Time", f"{tmetrics.get('training_time_minutes', 0)} min"
            )

            with st.expander("📋 Episode rewards"):
                rewards_list = tmetrics.get("episode_rewards", [])
                for i, r in enumerate(rewards_list):
                    st.markdown(f"Episode {i + 1}: **{r:.2f}**")
        else:
            st.warning("No evaluation metrics found. Run `python src/train.py` first.")

    # ── Figures gallery ────────────────────────────────────────────────────
    st.subheader("📈 Evaluation Figures")
    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    if figure_files:
        cols = st.columns(min(len(figure_files), 3))
        for i, fig_path in enumerate(figure_files):
            with cols[i % 3]:
                st.image(str(fig_path), use_container_width=True)
                st.caption(fig_path.stem.replace("_", " ").title())
    else:
        st.info("No figures yet. Run `python src/evaluate.py` to generate them.")

    # ── Eval summary table ─────────────────────────────────────────────────
    if eval_results:
        st.subheader("🌡️ Scenario Comparison")
        rows = []
        for label, data in eval_results.items():
            summary = data.get("summary", {})
            rows.append(
                {
                    "Scenario": label,
                    "Total Reward": f"{data['total_reward']:.2f}",
                    "Total AC Hours": summary.get("total_ac_hours", "N/A"),
                }
            )
            for zone, dev in summary.get("mean_comfort_deviation_per_zone", {}).items():
                rows[-1][f"{zone.title()} Comfort Dev"] = f"{dev:.2f}°C"
            for zone, pct in summary.get("ac_usage_pct_per_zone", {}).items():
                rows[-1][f"{zone.title()} AC %"] = f"{pct:.1f}%"

        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============================================================================
# TAB 3: SCENARIO COMPARISON
# ============================================================================

with tab_compare:
    st.header("🌡️ Climate Scenario Comparison")
    st.markdown(
        "Compare the agent's strategy across different outdoor temperature "
        "baselines side-by-side."
    )

    eval_results = load_eval_results()
    if eval_results is None:
        st.warning("No evaluation results found. Run `python src/evaluate.py` first.")
        st.stop()

    scenarios = list(eval_results.keys())
    if len(scenarios) < 2:
        st.info("Need at least 2 scenarios to compare.")
        st.stop()

    # ── Temperature comparison plot ────────────────────────────────────────
    st.subheader("🌡️ Temperature Regulation")

    selected_scenarios = st.multiselect(
        "Select scenarios to display",
        scenarios,
        default=scenarios[:2],
    )

    fig_comp, axs_comp = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    zone_colors = {"living": "#e74c3c", "kitchen": "#2ecc71", "bedroom": "#3498db"}
    scenario_styles = ["-", "--", ":"]
    scenario_colors = ["#2c3e50", "#7f8c8d", "#bdc3c7"]

    zone_keys = ["living", "kitchen", "bedroom"]
    zone_labels = ["Living Room", "Kitchen", "Bedroom"]

    for zi, (zk, zl) in enumerate(zip(zone_keys, zone_labels)):
        ax = axs_comp[zi]
        for si, label in enumerate(selected_scenarios):
            data = eval_results[label]
            hourly = data.get("hourly", [])
            if not hourly:
                continue
            hours_data = [h["hour"] for h in hourly]
            temps_data = [h["temps"][zi] for h in hourly]

            ax.plot(
                hours_data,
                temps_data,
                linestyle=scenario_styles[si % len(scenario_styles)],
                color=scenario_colors[si % len(scenario_colors)],
                label=label.split("(")[0].strip(),
                linewidth=1.5,
            )

        target = [21.0, 22.0, 20.0][zi]
        ax.axhline(target, color=zone_colors[zk], linestyle=":", alpha=0.5)
        ax.set_ylabel("°C")
        ax.set_title(zl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for day in range(1, 4):
            ax.axvline(day * 24 - 0.5, color="gray", linestyle="--", alpha=0.2)

    axs_comp[2].set_xlabel("Hour")
    fig_comp.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # ── AC usage comparison ────────────────────────────────────────────────
    st.subheader("🔌 AC Usage Comparison")

    import pandas as pd

    comp_rows = []
    for label in scenarios:
        data = eval_results[label]
        summary = data.get("summary", {})
        row = {"Scenario": label, "Total Reward": data["total_reward"]}
        for zone, pct in summary.get("ac_usage_pct_per_zone", {}).items():
            row[f"{zone.title()} AC %"] = pct
        row["Total AC Hours"] = summary.get("total_ac_hours", 0)
        for zone, dev in summary.get("mean_comfort_deviation_per_zone", {}).items():
            row[f"{zone.title()} Comfort Dev"] = dev
        comp_rows.append(row)

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    # ── Bar chart: AC usage per zone ──────────────────────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    x = np.arange(len(zone_labels))
    width = 0.8 / len(selected_scenarios)

    for si, label in enumerate(selected_scenarios):
        data = eval_results[label]
        summary = data.get("summary", {})
        ac_pcts = [
            summary.get("ac_usage_pct_per_zone", {}).get(zk, 0) for zk in zone_keys
        ]
        offset = (si - len(selected_scenarios) / 2 + 0.5) * width
        bars = ax_bar.bar(
            x + offset,
            ac_pcts,
            width,
            label=label.split("(")[0].strip(),
            alpha=0.8,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(zone_labels)
    ax_bar.set_ylabel("AC Usage (%)")
    ax_bar.set_title("AC Usage by Zone across Scenarios")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # ── Key takeaways ──────────────────────────────────────────────────────
    st.subheader("💡 Key Takeaways")
    st.markdown("""
    - **Solar-aware cooling** — The agent should pre-cool the south-facing living
      room before the midday solar peak.
    - **Inter-zone coordination** — Cooling one zone helps adjacent zones via
      heat transfer; the agent can exploit this.
    - **Night flushing** — The optimal strategy turns OFF AC when outdoor
      temperature drops below indoor at night.
    - **Climate adaptation** — On hot days the agent must prioritise; expect
      higher comfort deviation in the bedroom vs living room.
    """)
