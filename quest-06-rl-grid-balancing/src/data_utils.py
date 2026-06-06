import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA = PROJECT / "data"
DEFAULT_RESULTS = PROJECT / "results"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MACHINE_DEFS = [
    {
        "name": "general",
        "min_instances": 50,
        "max_instances": 2000,
        "cpu_per_instance": 8,
        "mem_per_instance": 32,
        "cost_per_hour": 0.32,
        "power_watts": 180,
        "co2_rate": 0.00018,
        "gpu": False,
    },
    {
        "name": "compute_opt",
        "min_instances": 20,
        "max_instances": 1000,
        "cpu_per_instance": 16,
        "mem_per_instance": 32,
        "cost_per_hour": 0.48,
        "power_watts": 220,
        "co2_rate": 0.00022,
        "gpu": False,
    },
    {
        "name": "memory_opt",
        "min_instances": 10,
        "max_instances": 500,
        "cpu_per_instance": 8,
        "mem_per_instance": 128,
        "cost_per_hour": 0.64,
        "power_watts": 200,
        "co2_rate": 0.00020,
        "gpu": False,
    },
    {
        "name": "gpu",
        "min_instances": 5,
        "max_instances": 200,
        "cpu_per_instance": 16,
        "mem_per_instance": 64,
        "cost_per_hour": 3.50,
        "power_watts": 550,
        "co2_rate": 0.00055,
        "gpu": True,
    },
    {
        "name": "storage_opt",
        "min_instances": 10,
        "max_instances": 300,
        "cpu_per_instance": 8,
        "mem_per_instance": 64,
        "cost_per_hour": 0.40,
        "power_watts": 260,
        "co2_rate": 0.00026,
        "gpu": False,
    },
]

HOURLY_WORKLOAD_SHAPE = np.array(
    [
        0.25,
        0.22,
        0.20,
        0.18,
        0.18,
        0.20,
        0.30,
        0.50,
        0.70,
        0.85,
        0.92,
        0.95,
        0.90,
        0.88,
        0.85,
        0.82,
        0.85,
        0.90,
        0.95,
        0.85,
        0.70,
        0.55,
        0.42,
        0.32,
    ]
)

WEEKEND_CPU_SCALE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.60, 6: 0.50}
WEEKEND_MEM_SCALE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.65, 6: 0.55}

BASE_CPU_DEMAND = 4000
BASE_MEM_DEMAND = 12000

FAILURE_RATE = 0.005
REPAIR_RATE = 0.3


@dataclass
class ClusterParams:
    hour: int
    day_of_week: int
    cpu_demand: float
    mem_demand: float
    machine_availability: Dict[str, float]
    machine_prices: Dict[str, float]


def sample_workload_demand(
    hour: int,
    day_of_week: int,
    noise_std: float = 0.08,
) -> Tuple[float, float]:
    shape_val = HOURLY_WORKLOAD_SHAPE[hour]
    cpu_demand = BASE_CPU_DEMAND * shape_val * WEEKEND_CPU_SCALE[day_of_week]
    mem_demand = BASE_MEM_DEMAND * shape_val * WEEKEND_MEM_SCALE[day_of_week]
    cpu_demand *= np.random.lognormal(0, noise_std)
    mem_demand *= np.random.lognormal(0, noise_std)
    return max(cpu_demand, 100), max(mem_demand, 500)


def sample_machine_availability(
    machine_type: Dict,
    prev_available: Optional[float] = None,
) -> float:
    n_max = machine_type["max_instances"]
    if prev_available is None:
        steady_state = 1.0 - FAILURE_RATE / (FAILURE_RATE + REPAIR_RATE)
        noise = np.random.normal(0, 0.02)
        return np.clip(steady_state + noise, 0.85, 1.0)
    n_available = int(prev_available * n_max)
    n_failed = n_max - n_available
    new_failures = np.random.binomial(n_available, FAILURE_RATE)
    repairs = np.random.binomial(n_failed, REPAIR_RATE)
    n_available = n_available - new_failures + repairs
    frac = n_available / n_max
    return np.clip(frac, 0.0, 1.0)


def sample_cluster_params(
    hour: int,
    day_of_week: int,
    prev_availability: Optional[Dict[str, float]] = None,
) -> ClusterParams:
    cpu_demand, mem_demand = sample_workload_demand(hour, day_of_week)
    avail = {}
    prices = {}
    for m in MACHINE_DEFS:
        prev = prev_availability.get(m["name"]) if prev_availability else None
        avail[m["name"]] = sample_machine_availability(m, prev)
        prices[m["name"]] = m["cost_per_hour"] + np.random.normal(0, 0.02)
    return ClusterParams(
        hour=hour,
        day_of_week=day_of_week,
        cpu_demand=cpu_demand,
        mem_demand=mem_demand,
        machine_availability=avail,
        machine_prices=prices,
    )


def generate_workload_sequence(
    n_hours: int,
    seed: int = 42,
) -> List[ClusterParams]:
    np.random.seed(seed)
    scenarios = []
    prev_avail = None
    for i in range(n_hours):
        hour = i % 24
        day_of_week = (i // 24) % 7
        sp = sample_cluster_params(hour, day_of_week, prev_avail)
        scenarios.append(sp)
        prev_avail = sp.machine_availability
    return scenarios


def estimate_total_cpu_capacity() -> float:
    return float(sum(m["max_instances"] * m["cpu_per_instance"] for m in MACHINE_DEFS))


def estimate_total_mem_capacity() -> float:
    return float(sum(m["max_instances"] * m["mem_per_instance"] for m in MACHINE_DEFS))


def save_workload_sequence(
    scenarios: List[ClusterParams],
    path: Path = DEFAULT_DATA / "workload.pkl",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"Saved {len(scenarios)} timesteps to {path}")


def load_workload_sequence(
    path: Path = DEFAULT_DATA / "workload.pkl",
) -> List[ClusterParams]:
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_sample_data(num_hours: int = 24 * 7):
    scenarios = generate_workload_sequence(num_hours, seed=42)
    save_workload_sequence(scenarios, DEFAULT_DATA / "sample" / "workload.pkl")
    summary = []
    for s in scenarios:
        d = asdict(s)
        d["machine_availability"] = {
            k: round(v, 3) for k, v in d["machine_availability"].items()
        }
        d["machine_prices"] = {k: round(v, 3) for k, v in d["machine_prices"].items()}
        d["cpu_demand"] = round(d["cpu_demand"], 1)
        d["mem_demand"] = round(d["mem_demand"], 1)
        summary.append(d)
    json_path = DEFAULT_DATA / "sample" / "workload_sample.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sample data saved to {json_path}")


def download_google_cluster_data(
    output_dir: Path = DEFAULT_DATA / "google_cluster",
):
    import urllib.request

    base_url = "https://commondatastorage.googleapis.com/clusterdata-2011-2"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Google Cluster Data sample to {output_dir}...")
    print("  Downloading machine events...")
    machine_url = f"{base_url}/machine_events/part-00000-of-00001.csv.gz"
    local_path = output_dir / "machine_events.csv.gz"
    try:
        urllib.request.urlretrieve(machine_url, local_path)
        print(f"  Saved {local_path}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Falling back to synthetic data generator.")
        return None
    print("  Downloading task events (sample)...")
    task_url = f"{base_url}/task_events/part-00000-of-00500.csv.gz"
    local_path = output_dir / "task_events_sample.csv.gz"
    try:
        urllib.request.urlretrieve(task_url, local_path)
        print(f"  Saved {local_path}")
    except Exception as e:
        print(f"  Task events download failed: {e}")
    print("Done.")
    return output_dir


if __name__ == "__main__":
    generate_sample_data()
    print(f"Total CPU capacity: {estimate_total_cpu_capacity():.0f} vCores")
    print(f"Total memory capacity: {estimate_total_mem_capacity():.0f} GB")
    print("Data center data utilities ready.")
