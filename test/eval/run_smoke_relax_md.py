import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from pymatgen.core.structure import Structure


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.md import MolecularDynamics
from matris.applications.relax import StructOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight relaxation + MD smoke test.")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=("fp32", "tf32", "bf16", "fp16"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--relax-structure", default="/home/lht/lab/MatRIS/example/cif_file/demo.cif")
    parser.add_argument("--relax-task", default="efs")
    parser.add_argument("--relax-steps", type=int, default=20)
    parser.add_argument("--relax-fmax", type=float, default=0.1)
    parser.add_argument("--optimizer", default="FIRE")
    parser.add_argument("--relax-cell", action="store_true", default=True)
    parser.add_argument("--ase-filter", default="FrechetCellFilter")
    parser.add_argument("--md-task", default="efs")
    parser.add_argument("--md-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def configure_precision(device: str, precision_mode: str) -> None:
    tf32_enabled = precision_mode == "tf32"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled


def maybe_compile(calculator_like, use_compile: bool):
    if not use_compile:
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in the current PyTorch.")
    if hasattr(calculator_like, "calculator") and hasattr(calculator_like.calculator, "model"):
        calculator_like.calculator.model = torch.compile(calculator_like.calculator.model)


def build_md_atoms():
    return bulk("Cu", a=5.43, cubic=True)


def run_relaxation(args: argparse.Namespace) -> dict:
    atom = Structure.from_file(args.relax_structure)
    optimizer = StructOptimizer(
        model=args.model,
        task=args.relax_task,
        optimizer=args.optimizer,
        device=args.device,
    )
    maybe_compile(optimizer, args.compile)

    sync_if_needed(args.device)
    start = time.perf_counter()
    result = optimizer.relax(
        atoms=atom,
        verbose=False,
        steps=args.relax_steps,
        fmax=args.relax_fmax,
        relax_cell=args.relax_cell,
        ase_filter=args.ase_filter,
    )
    sync_if_needed(args.device)

    traj = result["trajectory"]
    final_forces = np.asarray(traj.forces[-1], dtype=float)
    final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
    return {
        "total_time_s": time.perf_counter() - start,
        "recorded_steps": len(traj),
        "final_energy_eV": float(traj.energies[-1]),
        "final_fmax_eVA": final_fmax,
        "converged": bool(final_fmax <= args.relax_fmax),
    }


def run_md(args: argparse.Namespace) -> dict:
    atoms = build_md_atoms()
    md = MolecularDynamics(
        atoms=atoms,
        model=args.model,
        ensemble="nvt",
        thermostat="Berendsen",
        temperature=args.temperature,
        starting_temperature=args.temperature,
        timestep=args.timestep_fs,
        trajectory=None,
        logfile=None,
        loginterval=1,
        task=args.md_task,
        device=args.device,
    )
    maybe_compile(md, args.compile)

    sync_if_needed(args.device)
    start = time.perf_counter()
    md.run(args.md_steps)
    sync_if_needed(args.device)
    total_time_s = time.perf_counter() - start
    return {
        "steps": args.md_steps,
        "total_time_s": total_time_s,
        "steps_per_s": args.md_steps / total_time_s if total_time_s > 0 else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_precision(args.device, args.precision_mode)
    relax_result = run_relaxation(args)
    md_result = run_md(args)
    summary = {
        "model": args.model,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "compile": args.compile,
        "relax": relax_result,
        "md": md_result,
    }

    with open(output_dir / "smoke_relax_md_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("=== Relaxation ===")
    for key, value in relax_result.items():
        print(f"{key}: {value}")
    print("\n=== MD ===")
    for key, value in md_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
