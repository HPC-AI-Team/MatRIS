import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight force-consistency checks: random perturbation + finite difference."
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="ef", choices=("ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=("fp32", "tf32", "bf16", "fp16"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fd-step-ang", type=float, default=1e-3)
    parser.add_argument("--perturb-std-ang", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_precision(device: str, precision_mode: str) -> None:
    tf32_enabled = precision_mode == "tf32"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled


def autocast_context(device: str, precision_mode: str):
    if device != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    if precision_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def build_calculator(args: argparse.Namespace) -> MatRISCalculator:
    return MatRISCalculator(model=args.model, task=args.task, device=args.device)


def predict(atoms, calc: MatRISCalculator, args: argparse.Namespace) -> tuple[float, np.ndarray]:
    atoms = atoms.copy()
    atoms.calc = calc
    with autocast_context(args.device, args.precision_mode):
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces


def finite_difference_check(atoms, calc: MatRISCalculator, args: argparse.Namespace, rng: random.Random) -> dict:
    atom_idx = rng.randrange(len(atoms))
    axis = rng.randrange(3)
    h = args.fd_step_ang

    plus = atoms.copy()
    minus = atoms.copy()
    pos_plus = plus.get_positions()
    pos_minus = minus.get_positions()
    pos_plus[atom_idx, axis] += h
    pos_minus[atom_idx, axis] -= h
    plus.set_positions(pos_plus)
    minus.set_positions(pos_minus)

    e_plus, _ = predict(plus, calc, args)
    e_minus, _ = predict(minus, calc, args)
    _, forces = predict(atoms, calc, args)

    fd_force = -(e_plus - e_minus) / (2.0 * h)
    direct_force = float(forces[atom_idx, axis])
    return {
        "fd_atom_index": atom_idx,
        "fd_axis": axis,
        "fd_force_eVA": fd_force,
        "direct_force_eVA": direct_force,
        "fd_abs_error_eVA": abs(fd_force - direct_force),
    }


def random_perturbation_check(atoms, calc: MatRISCalculator, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    e0, f0 = predict(atoms, calc, args)

    perturbed = atoms.copy()
    disp = rng.normal(loc=0.0, scale=args.perturb_std_ang, size=perturbed.positions.shape)
    perturbed.set_positions(perturbed.positions + disp)
    e1, f1 = predict(perturbed, calc, args)

    return {
        "energy_change_eV": abs(e1 - e0),
        "force_change_mae_eVA": float(np.abs(f1 - f0).mean()),
        "force_change_max_eVA": float(np.linalg.norm(f1 - f0, axis=1).max()),
        "perturb_rms_ang": float(np.sqrt((disp ** 2).mean())),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_precision(args.device, args.precision_mode)
    calc = build_calculator(args)

    with open(args.metadata_csv, "r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))[: args.limit]

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    records = []
    for idx, row in enumerate(rows, start=1):
        atoms = ase_read(row["cif_path"])
        fd = finite_difference_check(atoms, calc, args, rng)
        pert = random_perturbation_check(atoms, calc, args, np_rng)
        record = {
            "material_id": row.get("material_id", ""),
            "structure_role": row.get("structure_role", ""),
            "cif_path": row.get("cif_path", ""),
            **fd,
            **pert,
        }
        records.append(record)
        print(
            f"[{idx}/{len(rows)}] {record['material_id']} "
            f"fd_abs_error={record['fd_abs_error_eVA']:.6e} eV/A "
            f"force_change_mae={record['force_change_mae_eVA']:.6e} eV/A"
        )

    summary = {
        "num_structures": len(records),
        "fd_abs_error_eVA_mean": float(np.mean([r["fd_abs_error_eVA"] for r in records])) if records else 0.0,
        "fd_abs_error_eVA_max": float(np.max([r["fd_abs_error_eVA"] for r in records])) if records else 0.0,
        "force_change_mae_eVA_mean": float(np.mean([r["force_change_mae_eVA"] for r in records])) if records else 0.0,
        "force_change_max_eVA_max": float(np.max([r["force_change_max_eVA"] for r in records])) if records else 0.0,
    }

    with open(output_dir / "force_consistency_records.jsonl", "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "force_consistency_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6e}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
