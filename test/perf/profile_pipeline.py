import os
import statistics
import time
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from matris.model.model import MatRIS
from matris.model.processgraph import process_graphs


MODEL_NAME = "matris_10m_oam"
LATTICE_A = 5.43
TASKS = ("e", "ef", "efs")
WARMUP_RUNS = 3
MEASURE_RUNS = 10
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "profile_pipeline.log")


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def build_atoms():
    return bulk("Cu", a=LATTICE_A, cubic=True)


def timed_call(device: str, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def run_embedding_only(model: MatRIS, batch_graph):
    node_feat = model.atom_embedding(batch_graph["atomic_numbers"] - 1)
    edge_feat, smooth_weight = model.edge_embedding(graphs=batch_graph)

    threebody_feat = None
    if len(batch_graph["line_graph_dict"]["line_graph"]) != 0:
        threebody_feat = model.three_body_embedding(graphs=batch_graph)
    return node_feat, edge_feat, threebody_feat, smooth_weight


def run_interaction_only(model: MatRIS, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight):
    for mp_layer in model.interaction_block:
        node_feat, edge_feat, threebody_feat = mp_layer(
            batch_graph=batch_graph,
            node_feat=node_feat,
            edge_feat=edge_feat,
            threebody_feat=threebody_feat,
            smooth_weight=smooth_weight,
        )
    return node_feat, edge_feat, threebody_feat


def run_readout_only(model: MatRIS, batch_graph, node_feat):
    node_feat = model.readout_norm(node_feat)
    total_energy = model.energy_head(batch_graph=batch_graph, node_feat=node_feat)
    return total_energy


def profile_once(model: MatRIS, task: str, device: str) -> dict:
    result = {
        "atoms_to_structure_ms": 0.0,
        "graph_converter_ms": 0.0,
        "graph_to_device_ms": 0.0,
        "process_graphs_ms": 0.0,
        "embedding_ms": 0.0,
        "interaction_blocks_ms": 0.0,
        "energy_head_ms": 0.0,
        "force_autograd_ms": 0.0,
        "stress_autograd_ms": 0.0,
        "cpu_output_ms": 0.0,
    }

    atoms = build_atoms()

    structure, result["atoms_to_structure_ms"] = timed_call(
        device, lambda: AseAtomsAdaptor.get_structure(atoms)
    )

    graph_cpu, result["graph_converter_ms"] = timed_call(
        device, lambda: model.graph_converter(structure)
    )

    graph, result["graph_to_device_ms"] = timed_call(
        device, lambda: graph_cpu.to(device)
    )
    graphs = [graph]

    if task == "e":
        batch_graph, result["process_graphs_ms"] = timed_call(
            device, lambda: process_graphs(graphs, compute_stress=False)
        )
        (node_feat, edge_feat, threebody_feat, smooth_weight), result["embedding_ms"] = timed_call(
            device, lambda: run_embedding_only(model, batch_graph)
        )
        (node_feat, edge_feat, threebody_feat), result["interaction_blocks_ms"] = timed_call(
            device,
            lambda: run_interaction_only(
                model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
            ),
        )
        total_energy, result["energy_head_ms"] = timed_call(
            device, lambda: run_readout_only(model, batch_graph, node_feat)
        )

        prediction_for_cpu = {
            "e": total_energy,
        }

    elif task == "ef":
        batch_graph, result["process_graphs_ms"] = timed_call(
            device, lambda: process_graphs(graphs, compute_stress=False)
        )
        (node_feat, edge_feat, threebody_feat, smooth_weight), result["embedding_ms"] = timed_call(
            device, lambda: run_embedding_only(model, batch_graph)
        )
        (node_feat, edge_feat, threebody_feat), result["interaction_blocks_ms"] = timed_call(
            device,
            lambda: run_interaction_only(
                model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
            ),
        )
        total_energy, result["energy_head_ms"] = timed_call(
            device, lambda: run_readout_only(model, batch_graph, node_feat)
        )

        force_tensor, result["force_autograd_ms"] = timed_call(
            device,
            lambda: torch.autograd.grad(
                total_energy.sum(),
                [batch_graph["batch_cart_coords"]],
                create_graph=False,
                retain_graph=False,
            )[0],
        )

        prediction_for_cpu = {
            "e": total_energy,
            "f": force_tensor,
        }

    elif task == "efs":
        batch_graph, result["process_graphs_ms"] = timed_call(
            device, lambda: process_graphs(graphs, compute_stress=True)
        )
        (node_feat, edge_feat, threebody_feat, smooth_weight), result["embedding_ms"] = timed_call(
            device, lambda: run_embedding_only(model, batch_graph)
        )
        (node_feat, edge_feat, threebody_feat), result["interaction_blocks_ms"] = timed_call(
            device,
            lambda: run_interaction_only(
                model, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight
            ),
        )
        total_energy, result["energy_head_ms"] = timed_call(
            device, lambda: run_readout_only(model, batch_graph, node_feat)
        )

        force_tensor, result["force_autograd_ms"] = timed_call(
            device,
            lambda: torch.autograd.grad(
                total_energy.sum(),
                [batch_graph["batch_cart_coords"]],
                create_graph=False,
                retain_graph=False,
            )[0],
        )

        # Rebuild a fresh autograd graph to profile stress separately.
        batch_graph_s, stress_process_graphs_ms = timed_call(
            device, lambda: process_graphs(graphs, compute_stress=True)
        )
        (node_feat_s, edge_feat_s, threebody_feat_s, smooth_weight_s), stress_embedding_ms = timed_call(
            device, lambda: run_embedding_only(model, batch_graph_s)
        )
        (node_feat_s, edge_feat_s, threebody_feat_s), stress_interaction_ms = timed_call(
            device,
            lambda: run_interaction_only(
                model,
                batch_graph_s,
                node_feat_s,
                edge_feat_s,
                threebody_feat_s,
                smooth_weight_s,
            ),
        )
        total_energy_s, stress_energy_head_ms = timed_call(
            device, lambda: run_readout_only(model, batch_graph_s, node_feat_s)
        )
        result["process_graphs_ms"] += stress_process_graphs_ms
        result["embedding_ms"] += stress_embedding_ms
        result["interaction_blocks_ms"] += stress_interaction_ms
        result["energy_head_ms"] += stress_energy_head_ms

        stress_tensor, result["stress_autograd_ms"] = timed_call(
            device,
            lambda: torch.autograd.grad(
                total_energy_s.sum(),
                [batch_graph_s["batch_strains"]],
                create_graph=False,
                retain_graph=False,
            )[0],
        )

        prediction_for_cpu = {
            "e": total_energy_s,
            "f": force_tensor,
            "s": stress_tensor,
        }

    else:
        raise ValueError(f"Unsupported task: {task}")

    def cpu_export():
        exported = {}
        for key, value in prediction_for_cpu.items():
            exported[key] = value.detach().cpu().numpy()
        return exported

    _, result["cpu_output_ms"] = timed_call(device, cpu_export)
    return result


def summarize_records(records: list[dict]) -> dict:
    keys = records[0].keys()
    summary = {}
    for key in keys:
        values = [record[key] for record in records]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def mean_map_from_summary(summary: dict) -> dict:
    return {key: stats["mean"] for key, stats in summary.items()}


def print_category_and_ranking(summary: dict) -> None:
    means = mean_map_from_summary(summary)
    overhead_keys = [
        "atoms_to_structure_ms",
        "graph_converter_ms",
        "graph_to_device_ms",
        "process_graphs_ms",
        "cpu_output_ms",
    ]
    core_keys = [
        "embedding_ms",
        "interaction_blocks_ms",
        "energy_head_ms",
        "force_autograd_ms",
        "stress_autograd_ms",
    ]

    overhead_total = sum(means[key] for key in overhead_keys)
    core_total = sum(means[key] for key in core_keys)
    total = overhead_total + core_total

    print("\nCategory Summary:")
    print(
        f"overhead_total={overhead_total:.3f} ms "
        f"({(overhead_total / total * 100.0) if total else 0.0:.2f}%)"
    )
    print(
        f"model_core_total={core_total:.3f} ms "
        f"({(core_total / total * 100.0) if total else 0.0:.2f}%)"
    )

    ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
    print("\nTime Ranking:")
    for idx, (key, value) in enumerate(ranked, start=1):
        pct = (value / total * 100.0) if total else 0.0
        category = "overhead" if key in overhead_keys else "model_core"
        print(f"{idx}. {key}: {value:.3f} ms ({pct:.2f}%) [{category}]")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Profile: FP32 pipeline breakdown")
    print("Model:", MODEL_NAME)
    print("Structure: ASE bulk Cu, a=5.43, cubic=True")
    print("Device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Warmup runs:", WARMUP_RUNS)
    print("Measure runs:", MEASURE_RUNS)
    print("Note: efs profiles force/stress on separate fresh graphs, so process_graphs/embedding/interaction/readout are doubled for diagnostic splitting and do not sum to exact end-to-end efs.")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA in torch:", torch.version.cuda)

    model = MatRIS.load(model_name=MODEL_NAME, device=device)
    model.eval()

    for task in TASKS:
        print(f"\n=== Profile task={task} ===")

        for _ in range(WARMUP_RUNS):
            profile_once(model, task, device)

        records = []
        for idx in range(MEASURE_RUNS):
            record = profile_once(model, task, device)
            records.append(record)
            print(
                f"run {idx + 1:02d}/{MEASURE_RUNS}: "
                f"atoms->structure={record['atoms_to_structure_ms']:.3f} ms, "
                f"graph={record['graph_converter_ms']:.3f} ms, "
                f"to(device)={record['graph_to_device_ms']:.3f} ms, "
                f"process_graphs={record['process_graphs_ms']:.3f} ms, "
                f"embedding={record['embedding_ms']:.3f} ms, "
                f"interaction={record['interaction_blocks_ms']:.3f} ms, "
                f"energy_head={record['energy_head_ms']:.3f} ms, "
                f"force_grad={record['force_autograd_ms']:.3f} ms, "
                f"stress_grad={record['stress_autograd_ms']:.3f} ms, "
                f"cpu_output={record['cpu_output_ms']:.3f} ms"
            )

        summary = summarize_records(records)
        print("\nSummary:")
        for key, stats in summary.items():
            print(
                f"{key}: mean={stats['mean']:.3f} ms, "
                f"std={stats['std']:.3f} ms, "
                f"min={stats['min']:.3f} ms, "
                f"max={stats['max']:.3f} ms"
            )
        print_category_and_ranking(summary)


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as log_fp:
        with redirect_stdout(log_fp):
            main()
