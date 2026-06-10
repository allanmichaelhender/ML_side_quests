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
        "min_instances": 5,
        "max_instances": 150,
        "cpu_per_instance": 8,
        "mem_per_instance": 32,
        "cost_per_hour": 0.32,
        "power_watts": 180,
        "co2_rate": 0.00018,
        "gpu": False,
        "gpu_per_instance": 0,
        "iops_per_instance": 1000,
    },
    {
        "name": "compute_opt",
        "min_instances": 3,
        "max_instances": 60,
        "cpu_per_instance": 16,
        "mem_per_instance": 32,
        "cost_per_hour": 0.48,
        "power_watts": 220,
        "co2_rate": 0.00022,
        "gpu": False,
        "gpu_per_instance": 0,
        "iops_per_instance": 500,
    },
    {
        "name": "memory_opt",
        "min_instances": 2,
        "max_instances": 40,
        "cpu_per_instance": 8,
        "mem_per_instance": 128,
        "cost_per_hour": 0.64,
        "power_watts": 200,
        "co2_rate": 0.00020,
        "gpu": False,
        "gpu_per_instance": 0,
        "iops_per_instance": 500,
    },
    {
        "name": "gpu",
        "min_instances": 1,
        "max_instances": 25,
        "cpu_per_instance": 16,
        "mem_per_instance": 64,
        "cost_per_hour": 3.50,
        "power_watts": 550,
        "co2_rate": 0.00055,
        "gpu": True,
        "gpu_per_instance": 1,
        "iops_per_instance": 1000,
    },
    {
        "name": "storage_opt",
        "min_instances": 2,
        "max_instances": 30,
        "cpu_per_instance": 8,
        "mem_per_instance": 64,
        "cost_per_hour": 0.40,
        "power_watts": 260,
        "co2_rate": 0.00026,
        "gpu": False,
        "gpu_per_instance": 0,
        "iops_per_instance": 10000,
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
WEEKEND_GPU_SCALE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.90, 5: 0.55, 6: 0.40}
WEEKEND_IOPS_SCALE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.70, 6: 0.60}

BASE_CPU_DEMAND = 4000
BASE_MEM_DEMAND = 12000
BASE_GPU_DEMAND = 8  # GPU-accelerated tasks (e.g. ML training jobs)
BASE_IOPS_DEMAND = 60000  # Storage IOPS throughput
BASE_POWER_CAPACITY_KW = 55  # Data center power budget in kW

# Carbon intensity (kg CO2/kWh) varies by hour — lower when solar/wind available
HOURLY_CARBON_SHAPE = np.array(
    [
        0.80,
        0.75,
        0.70,
        0.65,
        0.65,
        0.70,  # midnight-5am: wind available
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.15,  # 6-11am: solar ramps up
        0.12,
        0.10,
        0.10,
        0.15,
        0.20,
        0.30,  # noon-5pm: peak solar
        0.50,
        0.70,
        0.85,
        0.90,
        0.85,
        0.80,  # 6-11pm: solar fades, gas/coal
    ]
)

# Power budget scaling — higher during day (more cooling efficiency + solar)
HOURLY_POWER_SHAPE = np.array(
    [
        0.75,
        0.72,
        0.70,
        0.68,
        0.68,
        0.70,  # night: less efficient cooling
        0.75,
        0.82,
        0.88,
        0.95,
        1.00,
        1.00,  # morning: ramping up
        1.00,
        1.00,
        0.98,
        0.95,
        0.92,
        0.90,  # afternoon: peak solar helps
        0.88,
        0.85,
        0.82,
        0.80,
        0.78,
        0.76,  # evening: cooling costs rise
    ]
)

FAILURE_RATE = 0.008  # Slightly higher failure rate for more volatility
REPAIR_RATE = 0.25


@dataclass
class ClusterParams:
    hour: int
    day_of_week: int
    cpu_demand: float
    mem_demand: float
    gpu_demand: float  # GPU-accelerated tasks requiring GPU instances
    storage_iops_demand: float  # Storage IOPS throughput demand
    power_budget_kw: float  # Max power draw before penalty (kW)
    carbon_intensity: float  # kg CO2 per kWh at this hour
    machine_availability: Dict[str, float]
    machine_prices: Dict[str, float]


def sample_workload_demand(
    hour: int,
    day_of_week: int,
    noise_std: float = 0.08,
) -> Tuple[float, float, float, float]:
    shape_val = HOURLY_WORKLOAD_SHAPE[hour]
    cpu_demand = BASE_CPU_DEMAND * shape_val * WEEKEND_CPU_SCALE[day_of_week]
    mem_demand = BASE_MEM_DEMAND * shape_val * WEEKEND_MEM_SCALE[day_of_week]
    gpu_demand = BASE_GPU_DEMAND * shape_val * WEEKEND_GPU_SCALE[day_of_week]
    iops_demand = BASE_IOPS_DEMAND * shape_val * WEEKEND_IOPS_SCALE[day_of_week]
    cpu_demand *= np.random.lognormal(0, noise_std)
    mem_demand *= np.random.lognormal(0, noise_std)
    gpu_demand *= np.random.lognormal(0, noise_std * 0.5)
    iops_demand *= np.random.lognormal(0, noise_std)
    return (
        max(cpu_demand, 100),
        max(mem_demand, 500),
        max(round(gpu_demand), 0),
        max(iops_demand, 5000),
    )


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
    cpu_demand, mem_demand, gpu_demand, iops_demand = sample_workload_demand(
        hour, day_of_week
    )
    avail = {}
    prices = {}
    for m in MACHINE_DEFS:
        prev = prev_availability.get(m["name"]) if prev_availability else None
        avail[m["name"]] = sample_machine_availability(m, prev)
        prices[m["name"]] = m["cost_per_hour"] + np.random.normal(0, 0.02)
    power_budget = BASE_POWER_CAPACITY_KW * HOURLY_POWER_SHAPE[hour]
    power_budget *= np.random.lognormal(0, 0.03)
    carbon_intensity = HOURLY_CARBON_SHAPE[hour] * np.random.lognormal(0, 0.05)
    return ClusterParams(
        hour=hour,
        day_of_week=day_of_week,
        cpu_demand=cpu_demand,
        mem_demand=mem_demand,
        gpu_demand=gpu_demand,
        storage_iops_demand=iops_demand,
        power_budget_kw=power_budget,
        carbon_intensity=carbon_intensity,
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
