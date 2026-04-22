import os
import statistics
import time
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from ase.build import bulk
from matris.applications.base import MatRISCalculator


MODEL_NAME = "matris_10m_oam"
LATTICE_A = 5.43
TASKS = ("e", "ef", "efs")
WARMUP_RUNS = 5
MEASURE_RUNS = 20
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "benchmark_single_e2e.log")


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str) -> float:
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def build_atoms():
    return bulk("Cu", a=LATTICE_A, cubic=True)


def build_calculator(task: str, device: str) -> MatRISCalculator:
    return MatRISCalculator(
        model=MODEL_NAME,
        task=task,
        device=device,
    )


def run_single_measurement(task: str, device: str, calc) -> tuple[float, float]:
    atoms = build_atoms()
    atoms.calc = calc

    reset_peak_memory(device)
    sync_if_needed(device)
    start = time.perf_counter()

    if task == "e":
        _ = atoms.get_potential_energy()
    elif task == "ef":
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
    elif task == "efs":
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        _ = atoms.get_stress()
    else:
        raise ValueError(f"Unsupported task: {task}")

    sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_mem_mb = get_peak_memory_mb(device)
    return elapsed_ms, peak_mem_mb


def benchmark_task(task: str, device: str) -> dict:
    print(f"\n=== Benchmark task={task} (ASE e2e) ===")
    calc = build_calculator(task, device)

    for _ in range(WARMUP_RUNS):
        run_single_measurement(task, device, calc)

    latencies_ms = []
    peak_memories_mb = []
    for idx in range(MEASURE_RUNS):
        latency_ms, peak_mem_mb = run_single_measurement(task, device, calc)
        latencies_ms.append(latency_ms)
        peak_memories_mb.append(peak_mem_mb)
        print(
            f"run {idx + 1:02d}/{MEASURE_RUNS}: "
            f"latency={latency_ms:.3f} ms, peak_mem={peak_mem_mb:.2f} MB"
        )

    return {
        "task": task,
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "peak_mem_mb": max(peak_memories_mb) if peak_memories_mb else 0.0,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Benchmark: FP32 single-structure ASE end-to-end latency")
    print("Model:", MODEL_NAME)
    print("Structure: ASE bulk Cu, a=5.43, cubic=True")
    print("Device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Warmup runs:", WARMUP_RUNS)
    print("Measure runs:", MEASURE_RUNS)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA in torch:", torch.version.cuda)

    results = [benchmark_task(task, device) for task in TASKS]

    print("\n=== Summary ===")
    for result in results:
        print(
            f"task={result['task']}: "
            f"mean={result['mean_ms']:.3f} ms, "
            f"std={result['std_ms']:.3f} ms, "
            f"min={result['min_ms']:.3f} ms, "
            f"max={result['max_ms']:.3f} ms, "
            f"peak_mem={result['peak_mem_mb']:.2f} MB"
        )


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as log_fp:
        with redirect_stdout(log_fp):
            main()
