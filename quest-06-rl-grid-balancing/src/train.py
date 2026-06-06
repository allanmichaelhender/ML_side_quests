"""
Train a PPO agent for data center resource scheduling using Stable Baselines3.

Usage:
    python src/train.py                           # train with default params
    python src/train.py --total-timesteps 200000   # shorter run for testing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Ensure src/ is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_utils import (
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    generate_workload_sequence,
)
from cluster_env import ClusterDispatchEnv, make_env

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Default hyperparameters ────────────────────────────────────────────────
DEFAULT_TIMESTEPS = 1_000_000
EVAL_EPISODES = 5
EVAL_FREQ = 20_000
CHECKPOINT_FREQ = 100_000
SEED = 42
N_ENVS = 4  # parallel environments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO data center resource scheduler"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Episodes per evaluation (default: {EVAL_EPISODES})",
    )
    parser.add_argument(
        "--no-vec-normalize",
        action="store_true",
        help="Disable VecNormalize (observation scaling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=N_ENVS,
        help=f"Number of parallel environments (default: {N_ENVS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Data Center Resource Scheduling — PPO Training")
    print("=" * 60)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs:   {args.n_envs}")
    print(f"Eval episodes:   {args.eval_episodes}")
    print(f"Seed:            {args.seed}")
    print()

    # ── Generate workload scenarios ─────────────────────────────────────────
    print("Generating training workload scenarios...")
    train_scenarios = generate_workload_sequence(
        n_hours=args.total_timesteps // args.n_envs + 1000,
        seed=args.seed,
    )
    eval_scenarios = generate_workload_sequence(
        n_hours=24 * 30,  # 30 days for eval
        seed=args.seed + 1,
    )

    # ── Create environments ────────────────────────────────────────────────
    def make_train_env(rank: int):
        """Create a single training environment."""
        offset = rank * (len(train_scenarios) // args.n_envs)
        env = make_env(
            workload_sequence=train_scenarios[offset:],
            episode_length=24 * 7,  # 1 week episodes
            seed=args.seed + rank,
        )
        return Monitor(env)

    # Vectorised envs
    train_env = DummyVecEnv([lambda r=i: make_train_env(r) for i in range(args.n_envs)])

    # Normalise observations (optional)
    if not args.no_vec_normalize:
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )

    # Eval env (single, with same normalisation if used)
    eval_env = DummyVecEnv(
        [
            lambda: make_env(
                workload_sequence=eval_scenarios,
                episode_length=24 * 7,
                seed=args.seed + 999,
            )
        ]
    )
    if not args.no_vec_normalize:
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False
        )

    # ── PPO model ──────────────────────────────────────────────────────────
    print("Initialising PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // args.n_envs, 1),
        save_path=str(RESULTS_DIR / "checkpoints"),
        name_prefix="ppo_cluster",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "best_model"),
        log_path=str(RESULTS_DIR / "eval_logs"),
        eval_freq=max(EVAL_FREQ // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # ── Training ───────────────────────────────────────────────────────────
    print(f"\nStarting training ({args.total_timesteps:,} timesteps)...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes.")

    # ── Save final model ───────────────────────────────────────────────────
    model_path = RESULTS_DIR / "model.zip"
    model.save(str(model_path))
    print(f"Final model saved to {model_path}")

    if not args.no_vec_normalize:
        vec_norm_path = RESULTS_DIR / "vecnormalize.pkl"
        train_env.save(str(vec_norm_path))
        print(f"VecNormalize stats saved to {vec_norm_path}")

    # ── Save training metadata ─────────────────────────────────────────────
    metadata = {
        "algorithm": "PPO",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "episode_length_hours": 24 * 7,
        "training_time_minutes": round(training_time / 60, 1),
        "seed": args.seed,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vec_normalize": not args.no_vec_normalize,
    }
    meta_path = RESULTS_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to {meta_path}")

    # ── Quick evaluation (on a fresh, un-normalised env for interpretable rewards) ──
    print("\nRunning post-training evaluation...")
    raw_eval_env = DummyVecEnv(
        [
            lambda: make_env(
                workload_sequence=eval_scenarios,
                episode_length=24 * 7,
                seed=args.seed + 999,
            )
        ]
    )
    evaluate_model(model, raw_eval_env, n_episodes=args.eval_episodes)
    raw_eval_env.close()

    train_env.close()
    eval_env.close()
    print("\nDone!")


def evaluate_model(model, eval_env, n_episodes: int = 5):
    """Run a quick evaluation of the trained model."""
    episode_rewards = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
        episode_rewards.append(ep_reward)
        print(f"  Eval episode {ep + 1}: reward = {ep_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"  Mean episodic reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save eval results
    eval_results = {
        "mean_episodic_reward": round(float(mean_reward), 2),
        "std_episodic_reward": round(float(std_reward), 2),
        "n_episodes": n_episodes,
        "episode_rewards": [round(float(r), 2) for r in episode_rewards],
    }
    eval_path = RESULTS_DIR / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
