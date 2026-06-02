"""
Data utilities for the grid load-balancing simulation.

Generates synthetic (but realistic) demand, generation availability,
and pricing data parameterised by real-world distributions from
EIA / NREL. This is used to drive the GridDispatchEnv environment.
"""

import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_RESULTS = PROJECT / "results"

# Ensure src/ is on the path for sibling imports
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ── Source definitions ──────────────────────────────────────────────────────

# Each generation source has:
#   name        – display name
#   min_cap     – minimum stable generation (MW)
#   max_cap     – nameplate capacity (MW)
#   opex_var    – variable operating cost ($/MWh)
#   co2_rate    – CO₂ emissions (kg/MWh)
#   renew       – whether it's a renewable source (availability varies)
SOURCE_DEFS = [
    {
        "name": "coal",
        "min_cap": 100,
        "max_cap": 1200,
        "opex_var": 35.0,
        "co2_rate": 900.0,
        "renew": False,
    },
    {
        "name": "gas",
        "min_cap": 50,
        "max_cap": 1000,
        "opex_var": 55.0,
        "co2_rate": 450.0,
        "renew": False,
    },
    {
        "name": "solar",
        "min_cap": 0,
        "max_cap": 600,
        "opex_var": 5.0,
        "co2_rate": 0.0,
        "renew": True,
    },
    {
        "name": "wind",
        "min_cap": 0,
        "max_cap": 800,
        "opex_var": 8.0,
        "co2_rate": 0.0,
        "renew": True,
    },
    {
        "name": "hydro",
        "min_cap": 20,
        "max_cap": 500,
        "opex_var": 12.0,
        "co2_rate": 10.0,
        "renew": True,
    },
]

# Seasonal demand multipliers
SEASONAL_PROFILES = {
    "spring": {"base_demand": 1500, "peak_demand": 2300, "temp_factor": 0.0},
    "summer": {"base_demand": 1700, "peak_demand": 2800, "temp_factor": 0.15},
    "fall": {"base_demand": 1500, "peak_demand": 2400, "temp_factor": 0.0},
    "winter": {"base_demand": 1800, "peak_demand": 2600, "temp_factor": 0.10},
}

# Hourly demand shape (normalised 0..1) — typical double-peak pattern
HOURLY_DEMAND_SHAPE = np.array(
    [
        0.55,
        0.50,
        0.48,
        0.47,
        0.48,
        0.52,  # 00–05
        0.60,
        0.72,
        0.85,
        0.92,
        0.95,
        0.97,  # 06–11
        0.93,
        0.90,
        0.88,
        0.87,
        0.89,
        0.94,  # 12–17
        1.00,
        0.98,
        0.92,
        0.82,
        0.72,
        0.62,  # 18–23
    ]
)

# Solar availability by hour (normalised 0..1)
SOLAR_HOURLY_SHAPE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,  # 00–05
        0.10,
        0.30,
        0.55,
        0.75,
        0.88,
        0.95,  # 06–11
        0.97,
        0.93,
        0.85,
        0.72,
        0.55,
        0.35,  # 12–17
        0.15,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,  # 18–23
    ]
)

# Wind availability varies by season (normalised 0..1)
WIND_SEASONAL_FACTOR = {
    "spring": 0.40,
    "summer": 0.25,
    "fall": 0.45,
    "winter": 0.55,
}


@dataclass
class GridParams:
    """Parameters that define a single grid scenario."""

    season: str
    hour: int  # 0–23
    day_of_week: int  # 0=Monday
    demand_mw: float
    source_availability: Dict[str, float]  # fraction of max_cap available
    source_prices: Dict[str, float]  # $/MWh (can differ from opex_var for spot)


def sample_hourly_demand(season: str, hour: int, noise_std: float = 50.0) -> float:
    """Sample a realistic demand value for a given season and hour."""
    profile = SEASONAL_PROFILES[season]
    base = profile["base_demand"]
    peak = profile["peak_demand"]
    shape_val = HOURLY_DEMAND_SHAPE[hour]
    demand = base + (peak - base) * shape_val
    demand += np.random.normal(0, noise_std)
    return max(demand, profile["base_demand"] * 0.6)


def sample_renewable_availability(
    season: str,
    hour: int,
    source_name: str,
) -> float:
    """Return fraction (0..1) of max capacity available for a renewable source."""
    if source_name == "solar":
        base = SOLAR_HOURLY_SHAPE[hour]
        # Add cloud noise
        cloud_factor = np.random.beta(2, 5) * 0.3  # occasional clouds
        return max(0.0, base - cloud_factor)
    elif source_name == "wind":
        seasonal = WIND_SEASONAL_FACTOR[season]
        # Wind is stochastic — Weibull-ish
        gust = np.random.weibull(2.0) * 0.25
        return np.clip(seasonal + gust, 0.05, 1.0)
    elif source_name == "hydro":
        # Hydro is fairly stable but slightly seasonal
        seasonal_factor = {"spring": 0.90, "summer": 0.70, "fall": 0.80, "winter": 0.85}
        base = seasonal_factor[season]
        return np.clip(base + np.random.normal(0, 0.05), 0.5, 1.0)
    return 1.0  # dispatchable


def sample_grid_params(
    season: str,
    hour: int,
    day_of_week: int,
) -> GridParams:
    """Sample a full set of grid parameters for one timestep."""
    demand = sample_hourly_demand(season, hour)
    avail = {}
    prices = {}
    for src in SOURCE_DEFS:
        if src["renew"]:
            avail[src["name"]] = sample_renewable_availability(
                season, hour, src["name"]
            )
        else:
            avail[src["name"]] = 1.0  # fully dispatchable
        # Spot price = opex_var + small noise
        prices[src["name"]] = src["opex_var"] + np.random.normal(0, 3.0)
    return GridParams(
        season=season,
        hour=hour,
        day_of_week=day_of_week,
        demand_mw=demand,
        source_availability=avail,
        source_prices=prices,
    )


def generate_scenario_sequence(
    n_hours: int,
    seed: int = 42,
    start_season: str = "summer",
) -> List[GridParams]:
    """
    Generate a contiguous sequence of hourly grid scenarios
    spanning the given number of hours, cycling through seasons.
    """
    rng = np.random.default_rng(seed)
    seasons = ["spring", "summer", "fall", "winter"]
    season_idx = seasons.index(start_season)

    scenarios = []
    for i in range(n_hours):
        hour = i % 24
        # Season changes every ~90 days (2160 hours)
        if i > 0 and i % 2160 == 0:
            season_idx = (season_idx + 1) % 4
        season = seasons[season_idx]
        day_of_week = (i // 24) % 7
        scenarios.append(sample_grid_params(season, hour, day_of_week))
    return scenarios


def save_scenario_sequence(
    scenarios: List[GridParams],
    path: Path = DEFAULT_DATA / "scenarios.pkl",
):
    """Save generated scenarios to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"Saved {len(scenarios)} scenarios to {path}")


def load_scenario_sequence(
    path: Path = DEFAULT_DATA / "scenarios.pkl",
) -> List[GridParams]:
    """Load scenarios from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_sample_data(num_hours: int = 24 * 7):
    """Generate and save a small sample dataset (one week by default)."""
    scenarios = generate_scenario_sequence(num_hours, seed=42)
    save_scenario_sequence(scenarios, DEFAULT_DATA / "sample" / "scenarios.pkl")
    # Also save a human-readable JSON summary
    summary = []
    for s in scenarios:
        d = asdict(s)
        d["source_availability"] = {
            k: round(v, 3) for k, v in d["source_availability"].items()
        }
        d["source_prices"] = {k: round(v, 1) for k, v in d["source_prices"].items()}
        d["demand_mw"] = round(d["demand_mw"], 1)
        summary.append(d)
    json_path = DEFAULT_DATA / "sample" / "scenarios_sample.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sample data saved to {json_path}")


if __name__ == "__main__":
    generate_sample_data()
    print("Data utilities ready.")
