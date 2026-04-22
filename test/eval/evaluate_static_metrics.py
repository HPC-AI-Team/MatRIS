import argparse
import contextlib
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from ase.io import read as ase_read


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator


SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16", "fp16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatRIS on a structure set and save per-structure predictions/timing."
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to metadata csv, e.g. mp_testset_augmented_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save run_config.json, per_structure_predictions.jsonl and summary_overall.json",
    )
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efs", choices=("e", "ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision-mode",
        default="fp32",
        choices=sorted(SUPPORTED_PRECISIONS),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the loaded model with torch.compile when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of structures to run; 0 means all.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of initial structures to use as warmup before measured runs.",
    )
    return parser.parse_args()


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def configure_precision(device: str, precision_mode: str) -> dict:
    tf32_enabled = precision_mode == "tf32"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    autocast_dtype = None
    if precision_mode == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision_mode == "fp16":
        autocast_dtype = torch.float16

    return {
        "tf32_enabled": tf32_enabled,
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
    }


def autocast_context(device: str, precision_mode: str):
    if device != "cuda":
        return contextlib.nullcontext()
    if precision_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_calculator(args: argparse.Namespace) -> MatRISCalculator:
    calc = MatRISCalculator(
        model=args.model,
        task=args.task,
        device=args.device,
    )
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in the current PyTorch.")
        calc.model = torch.compile(calc.model)
    return calc


def run_prediction(row: dict, calc: MatRISCalculator, args: argparse.Namespace) -> dict:
    atoms = ase_read(row["cif_path"])
    atoms.calc = calc

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sync_if_needed(args.device)
    start = time.perf_counter()
    with autocast_context(args.device, args.precision_mode):
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces() if "f" in args.task else None
        stress = atoms.get_stress() if "s" in args.task else None
    sync_if_needed(args.device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    peak_mem_mb = 0.0
    if args.device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result = {
        "material_id": row.get("material_id", ""),
        "formula_pretty": row.get("formula_pretty", ""),
        "structure_role": row.get("structure_role", ""),
        "parent_material_id": row.get("parent_material_id", row.get("material_id", "")),
        "nsites": int(row["nsites"]) if row.get("nsites") else None,
        "crystal_system": row.get("crystal_system", ""),
        "is_magnetic": row.get("is_magnetic", ""),
        "cif_path": row.get("cif_path", ""),
        "pred_energy_eV": energy,
        "latency_ms": latency_ms,
        "peak_mem_mb": peak_mem_mb,
    }
    if forces is not None:
        result["pred_forces_eVA"] = forces.tolist()
    if stress is not None:
        result["pred_stress_eVA3"] = stress.tolist()
    return result


def run_warmup(rows: list[dict], calc: MatRISCalculator, args: argparse.Namespace) -> int:
    warmup_rows = rows[: min(args.warmup_steps, len(rows))]
    for idx, row in enumerate(warmup_rows, start=1):
        _ = run_prediction(row, calc, args)
        print(
            f"[warmup {idx}/{len(warmup_rows)}] {row.get('material_id', '')} "
            f"role={row.get('structure_role', '')}"
        )
    return len(warmup_rows)


def summarize(records: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in records]
    peaks = [r["peak_mem_mb"] for r in records]
    return {
        "num_structures": len(records),
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_ms_min": min(latencies) if latencies else 0.0,
        "latency_ms_max": max(latencies) if latencies else 0.0,
        "peak_mem_mb_max": max(peaks) if peaks else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_info = configure_precision(args.device, args.precision_mode)
    calc = build_calculator(args)

    with open(args.metadata_csv, "r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if args.limit > 0:
        rows = rows[: args.limit]

    run_config = {
        "metadata_csv": str(Path(args.metadata_csv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "compile": args.compile,
        "limit": args.limit,
        "warmup_steps": args.warmup_steps,
        "git_commit": get_git_commit(),
        **precision_info,
    }

    warmup_used = run_warmup(rows, calc, args) if args.warmup_steps > 0 else 0

    records = []
    jsonl_path = output_dir / "per_structure_predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as out_fp:
        for idx, row in enumerate(rows, start=1):
            record = run_prediction(row, calc, args)
            records.append(record)
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(rows)}] {record['material_id']} "
                f"role={record['structure_role']} latency={record['latency_ms']:.3f} ms"
            )

    summary = summarize(records)
    summary["warmup_steps_used"] = warmup_used

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(run_config, fp, ensure_ascii=False, indent=2)

    with open(output_dir / "summary_overall.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
