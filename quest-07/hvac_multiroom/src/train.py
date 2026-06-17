"""
Train a DQN agent on the 3-room multi-zone HVAC environment.

Produces:
  results/model.zip         — trained SB3 model
  results/metrics.json       — training metrics
  results/training_meta.json — hyperparameters used
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Ensure project root is on sys.path for package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import BuildingConfig
from src.env import MultiRoomHVACEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "model.zip"
METRICS_PATH = RESULTS_DIR / "metrics.json"
META_PATH = RESULTS_DIR / "training_meta.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-zone HVAC DQN agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500_000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("3-Room Multi-Zone HVAC — DQN Training")
    print("=" * 55)

    # ── Environment ────────────────────────────────────────────────────────
    cfg = BuildingConfig()
    env = MultiRoomHVACEnv(config=cfg)

    # Evaluation env (same config, different seed for stability check)
    eval_env = MultiRoomHVACEnv(config=cfg)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        tau=1.0,
        target_update_interval=250,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        gamma=0.95,
        verbose=1,
        seed=args.seed,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "best_model"),
        log_path=str(RESULTS_DIR / "eval_logs"),
        eval_freq=max(1, args.timesteps // 20),  # evaluate ~20 times
        deterministic=True,
        render=False,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    start_time = time.time()
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        log_interval=2_000,
    )
    train_time = time.time() - start_time
    print(f"Training complete! ({train_time / 60:.1f} minutes)\n")

    # Save final model
    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    # ── Collect final metrics ──────────────────────────────────────────────
    print("Running final evaluation...")
    n_eval_episodes = 10
    episode_rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)

    metrics = {
        "algorithm": "DQN",
        "n_zones": cfg.n_zones,
        "zone_names": cfg.zone_names,
        "total_timesteps": args.timesteps,
        "training_time_minutes": round(train_time / 60, 1),
        "n_eval_episodes": n_eval_episodes,
        "mean_episodic_reward": round(float(np.mean(episode_rewards)), 2),
        "std_episodic_reward": round(float(np.std(episode_rewards)), 2),
        "episode_rewards": [round(r, 2) for r in episode_rewards],
        "seed": args.seed,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # ── Save training metadata ─────────────────────────────────────────────
    meta = {
        "algorithm": "DQN",
        "total_timesteps": args.timesteps,
        "episode_steps": cfg.episode_steps,
        "seed": args.seed,
        "learning_rate": 0.0005,
        "buffer_size": 100_000,
        "gamma": 0.95,
        "batch_size": 64,
        "net_arch": [128, 128],
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 250,
        "train_freq": 4,
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    env.close()
    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
