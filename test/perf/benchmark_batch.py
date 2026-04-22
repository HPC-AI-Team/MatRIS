import os
import statistics
import time
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from matris.model.model import MatRIS


MODEL_NAME = "matris_10m_oam"
LATTICE_A = 5.43
TASKS = ("e", "ef", "efs")
BATCH_SIZES = (1, 2, 4, 8)
WARMUP_RUNS = 3
MEASURE_RUNS = 10
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "benchmark_batch.log")


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


def build_graph_batch(model: MatRIS, batch_size: int, device: str):
    atoms_list = [build_atoms() for _ in range(batch_size)]
    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
    graphs = [model.graph_converter(structure).to(device) for structure in structures]
    return graphs


def run_single_measurement(
    model: MatRIS,
    task: str,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    graphs = build_graph_batch(model, batch_size, device)

    reset_peak_memory(device)
    sync_if_needed(device)
    start = time.perf_counter()

    prediction = model(
        graphs,
        task=task,
        is_training=False,
    )

    if task == "e":
        _ = prediction["e"]
    elif task == "ef":
        _ = prediction["e"]
        _ = prediction["f"]
    elif task == "efs":
        _ = prediction["e"]
        _ = prediction["f"]
        _ = prediction["s"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_mem_mb = get_peak_memory_mb(device)
    return elapsed_ms, peak_mem_mb


def benchmark_config(model: MatRIS, task: str, batch_size: int, device: str) -> dict:
    print(f"\n=== Benchmark task={task}, batch_size={batch_size} ===")

    for _ in range(WARMUP_RUNS):
        run_single_measurement(model, task, batch_size, device)

    latencies_ms = []
    peak_memories_mb = []
    throughputs = []
    per_structure_ms = []

    for idx in range(MEASURE_RUNS):
        latency_ms, peak_mem_mb = run_single_measurement(model, task, batch_size, device)
        structures_per_sec = batch_size / (latency_ms / 1000.0)
        avg_per_structure_ms = latency_ms / batch_size

        latencies_ms.append(latency_ms)
        peak_memories_mb.append(peak_mem_mb)
        throughputs.append(structures_per_sec)
        per_structure_ms.append(avg_per_structure_ms)

        print(
            f"run {idx + 1:02d}/{MEASURE_RUNS}: "
            f"total={latency_ms:.3f} ms, "
            f"throughput={structures_per_sec:.3f} structures/s, "
            f"per_structure={avg_per_structure_ms:.3f} ms, "
            f"peak_mem={peak_mem_mb:.2f} MB"
        )

    return {
        "task": task,
        "batch_size": batch_size,
        "mean_total_ms": statistics.mean(latencies_ms),
        "std_total_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "mean_throughput": statistics.mean(throughputs),
        "std_throughput": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "mean_per_structure_ms": statistics.mean(per_structure_ms),
        "peak_mem_mb": max(peak_memories_mb) if peak_memories_mb else 0.0,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Benchmark: FP32 batch throughput")
    print("Model:", MODEL_NAME)
    print("Structure: ASE bulk Cu, a=5.43, cubic=True")
    print("Device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Warmup runs:", WARMUP_RUNS)
    print("Measure runs:", MEASURE_RUNS)
    print("Batch sizes:", BATCH_SIZES)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA in torch:", torch.version.cuda)

    model = MatRIS.load(model_name=MODEL_NAME, device=device)
    model.eval()

    results = []
    for task in TASKS:
        for batch_size in BATCH_SIZES:
            results.append(benchmark_config(model, task, batch_size, device))

    print("\n=== Summary ===")
    for result in results:
        print(
            f"task={result['task']}, batch={result['batch_size']}: "
            f"mean_total={result['mean_total_ms']:.3f} ms, "
            f"std_total={result['std_total_ms']:.3f} ms, "
            f"mean_throughput={result['mean_throughput']:.3f} structures/s, "
            f"std_throughput={result['std_throughput']:.3f} structures/s, "
            f"mean_per_structure={result['mean_per_structure_ms']:.3f} ms, "
            f"peak_mem={result['peak_mem_mb']:.2f} MB"
        )


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as log_fp:
        with redirect_stdout(log_fp):
            main()
