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
    Episode: 72 steps (three full days)
    """

    def __init__(self, episode_steps=72):
        super().__init__()

        self.episode_steps = episode_steps
        self.target_temp = 21.0
        self.alpha = 0.1  # Heat leakage rate
        self.beta = 1.5  # AC cooling power

        # Gym spaces — lets SB3 auto-configure the network
        max_hour_norm = episode_steps / 24.0
        self.observation_space = spaces.Box(
            low=np.array([15.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([35.0, 50.0, max_hour_norm], dtype=np.float32),
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
        terminated = self.hour >= self.episode_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================
TOTAL_TIMESTEPS = 300_000  # ~4166 episodes (×72 steps)

# ============================================================
# 3. TRAIN WITH SB3
# ============================================================
print("=" * 55)
print("Single-room HVAC — Gymnasium + Stable-Baselines3 DQN")
print("=" * 55)

env = HVACEnv()


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=50_000,  # ~700 episodes of experience stored
    learning_starts=1_000,  # Fill buffer with random steps before training
    batch_size=64,  # Transitions sampled per gradient step
    tau=1.0,  # Hard target update (full copy every interval)
    target_update_interval=10,  # Copy online → target net every 10 train steps
    train_freq=1,  # Train every environment step
    gradient_steps=1,  # One gradient update per train call
    exploration_fraction=0.3,  # Explore for first 30% of training
    exploration_final_eps=0.05,  # ε decays from 1.0 to 0.05
    gamma=0.95,  # Discount — rewards 70 steps ahead still matter
    verbose=1,
    seed=42,
)

print("\nTraining...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1_000)
print("Training complete!\n")


# ============================================================
# 4. TEST — roll out the learned policy over 3-day scenarios
# ============================================================
for label, base_temp in [
    ("Cool climate  (base=22°C)", 22.0),
    ("Warm climate  (base=27°C)", 27.0),
    ("Hot climate   (base=32°C)", 32.0),
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
    for hour in range(env.episode_steps):
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
