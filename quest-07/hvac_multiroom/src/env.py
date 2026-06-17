"""
3-room multi-zone HVAC environment with inter-zone heat transfer and solar gain.

Observation: [T_living, T_kitchen, T_bedroom, outdoor_temp, hour/24]  (Box 5,)
Action:     Discrete(8) — AC OFF/ON per zone, packed as 3-bit integer    (8 combos)
            Bits: [living, kitchen, bedroom];  0=OFF, 1=ON
            e.g. 5 (101b) = living=ON, kitchen=OFF, bedroom=ON
Reward:     Σ -|T_i - target_i| + energy_penalty × sum(action_i)
Episode:    72 steps (3 days), outdoor temp repeats diurnal cycle
"""

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import BuildingConfig


class MultiRoomHVACEnv(gym.Env):
    """Gymnasium environment for a 3-room building with inter-zone physics."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[BuildingConfig] = None):
        super().__init__()

        self.cfg = config or BuildingConfig()
        self.n_zones = len(self.cfg.zone_names)

        # ── Gym spaces ────────────────────────────────────────────────────
        max_hour_norm = self.cfg.episode_steps / 24.0
        self.observation_space = spaces.Box(
            low=np.array(
                [self.cfg.temp_min] * self.n_zones + [self.cfg.temp_min, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [self.cfg.temp_max] * self.n_zones + [self.cfg.temp_max, max_hour_norm],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        # Action: single integer 0-7, packed as 3-bit zone mask
        #   bit 0 (1) = bedroom, bit 1 (2) = kitchen, bit 2 (4) = living
        self.action_space = spaces.Discrete(8)

        # ── Internal state ────────────────────────────────────────────────
        self.current_temps: np.ndarray = None  # shape (n_zones,)
        self.hour: int = 0
        self.base_temp: float = self.cfg.outdoor_base

    # ── Outdoor temperature ────────────────────────────────────────────────
    def _outdoor_temp(self, hour: int) -> float:
        """Diurnal sine wave, min at phase_hour, max at phase_hour + 12."""
        return self.base_temp + self.cfg.outdoor_amplitude * math.sin(
            2 * math.pi * (hour - self.cfg.outdoor_phase_hour) / 24
        )

    # ── Solar gain factor ──────────────────────────────────────────────────
    def _solar_factor(self, hour: int) -> float:
        """Ramps from 0 at night to 1.0 at noon (hour 12), sinusoidal."""
        return max(0.0, math.sin(math.pi * (hour - 6) / 12))

    # ── Observation ────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        outdoor = self._outdoor_temp(self.hour)
        return np.array(
            list(self.current_temps) + [outdoor, self.hour / 24.0],
            dtype=np.float32,
        )

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Start all zones at 20°C, randomise outdoor baseline
        self.current_temps = np.full(self.n_zones, 20.0, dtype=np.float64)
        self.hour = 0
        self.base_temp = self.np_random.uniform(22.0, 32.0)

        return self._get_obs(), {}

    @staticmethod
    def _decode_action(action: int) -> np.ndarray:
        """Decode integer 0-7 into per-zone bits [living, kitchen, bedroom]."""
        return np.array([(action >> 2) & 1, (action >> 1) & 1, action & 1], dtype=int)

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_bits = self._decode_action(action)
        outdoor = self._outdoor_temp(self.hour)
        solar_factor = self._solar_factor(self.hour)

        # Per-zone temperature update
        for i in range(self.n_zones):
            # Heat exchange with outdoors
            dT = self.cfg.alpha_out[i] * (outdoor - self.current_temps[i])

            # Inter-zone conduction
            for j in range(self.n_zones):
                if i != j:
                    dT += self.cfg.alpha_inter[i][j] * (
                        self.current_temps[j] - self.current_temps[i]
                    )

            # Solar gain
            dT += self.cfg.solar_amp[i] * solar_factor

            # AC cooling (if action bit == 1)
            if action_bits[i] == 1:
                dT -= self.cfg.beta[i]

            self.current_temps[i] += dT

        # Clamp all temperatures
        self.current_temps = np.clip(
            self.current_temps, self.cfg.temp_min, self.cfg.temp_max
        )

        # ── Reward ─────────────────────────────────────────────────────────
        comfort_penalty = -self.cfg.comfort_weight * np.sum(
            np.abs(self.current_temps - np.array(self.cfg.target_temps))
        )
        energy_penalty = -self.cfg.energy_cost * int(np.sum(action_bits))
        reward = comfort_penalty + energy_penalty

        # ── Advance ────────────────────────────────────────────────────────
        self.hour += 1
        terminated = self.hour >= self.cfg.episode_steps
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ── Render ─────────────────────────────────────────────────────────────
    def render(self):
        """Print current state to console."""
        outdoor = self._outdoor_temp(self.hour)
        print(f"\nHour {self.hour:2d} | Outdoor: {outdoor:.1f}°C")
        for i, name in enumerate(self.cfg.zone_names):
            print(
                f"  {name:10s}: {self.current_temps[i]:.1f}°C  "
                f"(target {self.cfg.target_temps[i]:.1f}°C)"
            )
