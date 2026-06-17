"""
Building thermal parameters for the 3-room multi-zone HVAC environment.

Layout:
        ┌──────────┐
        │  Room 1   │  South-facing living room (gets solar gain)
        ├────┬──────┤
        │R2  │  R3  │  Kitchen (internal) & Bedroom (north-facing)
        └────┴──────┘

Adjacency:
  Room 1 ↔ Room 2 (shared wall)
  Room 1 ↔ Room 3 (shared wall)
  Room 2 and Room 3 are NOT adjacent (separated by Room 1)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BuildingConfig:
    """Thermal and physical parameters for each zone."""

    # ── Zone labels ────────────────────────────────────────────────────────
    zone_names: List[str] = field(
        default_factory=lambda: ["living", "kitchen", "bedroom"]
    )

    # ── Target temperatures (°C) per zone ───────────────────────────────────
    target_temps: List[float] = field(default_factory=lambda: [21.0, 22.0, 20.0])

    # ── Heat exchange rate with outdoors per zone ───────────────────────────
    #   Room 1 (living) has large south-facing window → higher leakage
    #   Room 2 (kitchen) is internal → well insulated from outside
    #   Room 3 (bedroom) north-facing → moderate leakage
    alpha_out: List[float] = field(default_factory=lambda: [0.15, 0.05, 0.10])

    # ── Inter-zone conduction matrix α_ij ──────────────────────────────────
    #   α_ij = heat transfer rate from zone j to zone i
    #   Index order: [living, kitchen, bedroom]
    #   Row i, col j: how much T_j affects ΔT_i
    alpha_inter: List[List[float]] = field(
        default_factory=lambda: [
            [0.0, 0.08, 0.08],  # living gets heat from kitchen & bedroom
            [0.08, 0.0, 0.0],  # kitchen gets heat from living only
            [0.08, 0.0, 0.0],  # bedroom gets heat from living only
        ]
    )

    # ── Solar gain amplitude per zone (°C per step) ────────────────────────
    #   Room 1 (south-facing): significant solar gain peaking at noon
    #   Room 2 (internal): no direct solar gain
    #   Room 3 (north-facing): minor indirect gain
    solar_amp: List[float] = field(default_factory=lambda: [3.0, 0.0, 0.5])

    # ── AC cooling power per zone (°C per step) ────────────────────────────
    beta: List[float] = field(default_factory=lambda: [1.5, 1.2, 1.0])

    # ── Outdoor temp parameters ────────────────────────────────────────────
    outdoor_base: float = 27.0  # centre of sine wave (°C)
    outdoor_amplitude: float = 6.0  # ± variation (°C)
    outdoor_phase_hour: int = 4  # hour of minimum temp (4 AM)

    # ── Reward weights ─────────────────────────────────────────────────────
    comfort_weight: float = 1.0  # multiplier on |T - target|
    energy_cost: float = 1.5  # penalty per zone with AC ON

    # ── Default episode ────────────────────────────────────────────────────
    episode_steps: int = 72  # 3 days × 24 hours

    # ── Observation bounds ─────────────────────────────────────────────────
    temp_min: float = 10.0
    temp_max: float = 45.0
