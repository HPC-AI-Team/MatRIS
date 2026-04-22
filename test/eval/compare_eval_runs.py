import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate static-eval run against an FP32 baseline run."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def record_key(row: dict) -> tuple[str, str, str]:
    return (
        row.get("material_id", ""),
        row.get("structure_role", ""),
        row.get("cif_path", ""),
    )


def force_metrics(base: np.ndarray, cand: np.ndarray) -> dict:
    diff = cand - base
    comp_abs = np.abs(diff).reshape(-1)
    vec_abs = np.linalg.norm(diff, axis=1)
    return {
        "force_mae": float(comp_abs.mean()) if comp_abs.size else 0.0,
        "force_rmse": float(np.sqrt((diff ** 2).mean())) if diff.size else 0.0,
        "max_force_error": float(vec_abs.max()) if vec_abs.size else 0.0,
        "p95_force_error": float(np.percentile(vec_abs, 95)) if vec_abs.size else 0.0,
        "p99_force_error": float(np.percentile(vec_abs, 99)) if vec_abs.size else 0.0,
    }


def stress_mae(base: np.ndarray, cand: np.ndarray) -> float:
    return float(np.abs(cand - base).mean()) if base.size else 0.0


def build_comparison_rows(baseline_rows: list[dict], candidate_rows: list[dict]) -> list[dict]:
    base_map = {record_key(row): row for row in baseline_rows}
    cand_map = {record_key(row): row for row in candidate_rows}

    shared_keys = sorted(set(base_map) & set(cand_map))
    rows = []
    for key in shared_keys:
        base = base_map[key]
        cand = cand_map[key]

        nsites = int(base.get("nsites") or 1)
        row = {
            "material_id": base.get("material_id", ""),
            "structure_role": base.get("structure_role", ""),
            "nsites": nsites,
            "crystal_system": base.get("crystal_system", ""),
            "is_magnetic": base.get("is_magnetic", ""),
            "latency_ms_baseline": float(base.get("latency_ms", 0.0)),
            "latency_ms_candidate": float(cand.get("latency_ms", 0.0)),
            "energy_abs_delta_eV": abs(float(cand["pred_energy_eV"]) - float(base["pred_energy_eV"])),
        }
        row["energy_abs_delta_meV_per_atom"] = row["energy_abs_delta_eV"] * 1000.0 / max(nsites, 1)

        if "pred_forces_eVA" in base and "pred_forces_eVA" in cand:
            fm = force_metrics(
                np.asarray(base["pred_forces_eVA"], dtype=float),
                np.asarray(cand["pred_forces_eVA"], dtype=float),
            )
            row.update(fm)

        if "pred_stress_eVA3" in base and "pred_stress_eVA3" in cand:
            row["stress_mae_eVA3"] = stress_mae(
                np.asarray(base["pred_stress_eVA3"], dtype=float),
                np.asarray(cand["pred_stress_eVA3"], dtype=float),
            )

        rows.append(row)
    return rows


def nsites_bin(nsites: int) -> str:
    if nsites <= 4:
        return "small(<=4)"
    if nsites <= 16:
        return "medium(5-16)"
    if nsites <= 32:
        return "large(17-32)"
    return "xlarge(>32)"


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}

    baseline_latency = np.array([r["latency_ms_baseline"] for r in rows], dtype=float)
    candidate_latency = np.array([r["latency_ms_candidate"] for r in rows], dtype=float)
    out = {
        "num_structures": len(rows),
        "latency_ms_baseline_mean": float(baseline_latency.mean()),
        "latency_ms_candidate_mean": float(candidate_latency.mean()),
        "speedup_vs_baseline": float(baseline_latency.mean() / candidate_latency.mean())
        if candidate_latency.mean() > 0
        else None,
        "energy_mae_delta_meV_per_atom": float(
            np.mean([r["energy_abs_delta_meV_per_atom"] for r in rows])
        ),
    }

    if "force_mae" in rows[0]:
        out["force_mae_delta_eVA"] = float(np.mean([r["force_mae"] for r in rows]))
        out["force_rmse_delta_eVA"] = float(np.mean([r["force_rmse"] for r in rows]))
        out["max_force_error_eVA"] = float(np.max([r["max_force_error"] for r in rows]))
        out["p95_force_error_eVA"] = float(np.mean([r["p95_force_error"] for r in rows]))
        out["p99_force_error_eVA"] = float(np.mean([r["p99_force_error"] for r in rows]))

    if "stress_mae_eVA3" in rows[0]:
        out["stress_mae_delta_eVA3"] = float(np.mean([r["stress_mae_eVA3"] for r in rows]))

    return out


def summarize_by_group(rows: list[dict]) -> dict:
    group_maps = {
        "structure_role": defaultdict(list),
        "nsites_bin": defaultdict(list),
        "crystal_system": defaultdict(list),
        "is_magnetic": defaultdict(list),
    }
    for row in rows:
        group_maps["structure_role"][row["structure_role"]].append(row)
        group_maps["nsites_bin"][nsites_bin(int(row["nsites"]))].append(row)
        group_maps["crystal_system"][row["crystal_system"]].append(row)
        group_maps["is_magnetic"][str(row["is_magnetic"])].append(row)

    summary = {}
    for group_name, mapping in group_maps.items():
        summary[group_name] = {name: summarize_rows(group_rows) for name, group_rows in sorted(mapping.items())}
    return summary


def write_markdown_report(path: Path, baseline_cfg: dict, candidate_cfg: dict, overall: dict) -> None:
    lines = [
        "# Comparison Report",
        "",
        f"- baseline_dir: `{path.parent.parent / baseline_cfg.get('output_dir', '')}`" if baseline_cfg.get("output_dir") else "",
        f"- candidate_dir: `{path.parent.parent / candidate_cfg.get('output_dir', '')}`" if candidate_cfg.get("output_dir") else "",
        f"- baseline_precision: `{baseline_cfg.get('precision_mode', 'unknown')}`",
        f"- candidate_precision: `{candidate_cfg.get('precision_mode', 'unknown')}`",
        f"- candidate_compile: `{candidate_cfg.get('compile', False)}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in overall.items():
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.6f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(line for line in lines if line != ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = load_json(baseline_dir / "run_config.json")
    candidate_cfg = load_json(candidate_dir / "run_config.json")
    baseline_rows = load_jsonl(baseline_dir / "per_structure_predictions.jsonl")
    candidate_rows = load_jsonl(candidate_dir / "per_structure_predictions.jsonl")

    comparison_rows = build_comparison_rows(baseline_rows, candidate_rows)
    overall = summarize_rows(comparison_rows)
    by_group = summarize_by_group(comparison_rows)

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as fp:
        json.dump(overall, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "comparison_by_group.json", "w", encoding="utf-8") as fp:
        json.dump(by_group, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "comparison_rows.jsonl", "w", encoding="utf-8") as fp:
        for row in comparison_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_markdown_report(output_dir / "comparison_report.md", baseline_cfg, candidate_cfg, overall)

    print("=== Overall Comparison ===")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
