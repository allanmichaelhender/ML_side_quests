"""
Single-room HVAC control with diurnal outdoor temperatures.
Uses Gymnasium for the environment and Stable-Baselines3 for DQN.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN


# ============================================================
# 1. GYMNASIUM ENVIRONMENT
# ============================================================
class HVACEnv(gym.Env):
    """Gymnasium-compatible HVAC environment with diurnal outdoor temps.

    Observation: [indoor_temp, outdoor_temp, hour_normalized]  (Box 3,)
    Action: 0 = AC OFF, 1 = AC ON                              (Discrete 2)
    Reward: -|temp - target| + energy_cost (AC ON = -1.5)
    Episode: 24 steps (one full day)
    """

    def __init__(self):
        super().__init__()

        self.target_temp = 21.0
        self.alpha = 0.1  # Heat leakage rate
        self.beta = 1.5  # AC cooling power

        # Gym spaces — lets SB3 auto-configure the network
        self.observation_space = spaces.Box(
            low=np.array([15.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([35.0, 50.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.current_temp = 20.0
        self.hour = 0
        self.base_temp = 27.0

    def _outdoor_temp(self, hour):
        """Sine wave: min at 4 AM, max at 4 PM, ±6°C around base_temp."""
        return self.base_temp + 6.0 * math.sin(2 * math.pi * (hour - 4) / 24)

    def _get_obs(self):
        return np.array(
            [self.current_temp, self._outdoor_temp(self.hour), self.hour / 24.0],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """Gymnasium API: returns (obs, info)."""
        super().reset(seed=seed)
        self.current_temp = 20.0
        self.hour = 0
        self.base_temp = self.np_random.uniform(22.0, 32.0)
        return self._get_obs(), {}

    def step(self, action):
        """Gymnasium API: returns (obs, reward, terminated, truncated, info)."""
        outdoor = self._outdoor_temp(self.hour)

        # Physics
        if action == 1:  # AC ON
            self.current_temp += self.alpha * (outdoor - self.current_temp) - self.beta
        else:  # AC OFF
            self.current_temp += self.alpha * (outdoor - self.current_temp)

        self.current_temp = np.clip(self.current_temp, 15.0, 35.0)

        # Reward
        comfort_penalty = -abs(self.current_temp - self.target_temp)
        energy_cost = -1.5 if action == 1 else 0.0
        reward = comfort_penalty + energy_cost

        # Advance hour
        self.hour += 1
        terminated = self.hour >= 24
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================
TOTAL_TIMESTEPS = 100_000  # ~4166 episodes (×24 steps)

# ============================================================
# 3. TRAIN WITH SB3
# ============================================================
print("=" * 55)
print("Single-room HVAC — Gymnasium + Stable-Baselines3 DQN")
print("=" * 55)

env = HVACEnv()

# SB3's DQN replaces ~150 lines of hand-written code:
#   - Neural network (MlpPolicy = 2 hidden layers × 64 neurons)
#   - Replay buffer
#   - Target network (hard-updated via tau=1.0)
#   - ε-greedy exploration (decays from 1.0 → exploration_final_eps)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=20_000,
    learning_starts=1_000,
    batch_size=64,
    tau=1.0,  # Hard target update (like our old code)
    target_update_interval=10,
    train_freq=1,
    gradient_steps=1,
    exploration_fraction=0.3,  # Explore for first 30% of training
    exploration_final_eps=0.05,  # ε decays from 1.0 to 0.05
    gamma=0.9,
    verbose=1,
    seed=42,
)

print("\nTraining...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1_000)
print("Training complete!\n")


# ============================================================
# 4. TEST — evaluate on 3 different base temps
# ============================================================
for label, base_temp in [
    ("Cool day  (base=22°C)", 22.0),
    ("Warm day  (base=27°C)", 27.0),
    ("Hot day   (base=32°C)", 32.0),
]:
    env.base_temp = base_temp
    env.current_temp = 20.0
    env.hour = 0
    obs = np.array([20.0, env._outdoor_temp(0), 0.0], dtype=np.float32)

    print(f"\n{label}")
    print(
        f"{'Hour':<6} | {'Indoor':<8} | {'Outdoor':<8} | {'Action':<12} | {'Reward':<7}"
    )
    print("-" * 55)

    total_reward = 0.0
    for hour in range(24):
        action, _ = model.predict(obs, deterministic=True)
        action_text = "AC ON" if action == 1 else "OFF"

        display_indoor = obs[0]
        display_outdoor = obs[1]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        print(
            f"{hour:<6} | {display_indoor:<8.2f} | {display_outdoor:<8.2f} | {action_text:<12} | {reward:<7.2f}"
        )

    print(f"{'Total reward:':>30} {total_reward:.2f}")
