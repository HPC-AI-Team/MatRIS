import csv
import os
from collections import defaultdict
from pathlib import Path

from mp_api.client import MPRester


OUT_ROOT = Path("/home/lht/lab/mp_testset")
STRUCT_DIR = OUT_ROOT / "raw_structures"
META_DIR = OUT_ROOT / "metadata"
META_CSV = META_DIR / "mp_testset_metadata.csv"

# Current target: build a balanced-structure parent set for task 7.
MAX_TOTAL = 40
MAX_PER_GROUP = 5
ENERGY_ABOVE_HULL_MAX = 0.02
NUM_SITES_MIN = 1
NUM_SITES_MAX = 40

# A small, diverse set of chemical systems to seed the parent test set.
CHEMSYS_LIST = [
    "Si",
    "C",
    "Al-O",
    "Ti-O",
    "Fe-O",
    "Ni-O",
    "Li-Fe-O",
    "Na-Cl",
    "Mo-S",
    "B-N",
]


def ensure_dirs() -> None:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def crystal_system_name(value) -> str:
    if value is None:
        return "unknown"
    # mp-api may return an enum-like object or a string.
    return getattr(value, "value", str(value)).lower()


def doc_crystal_system(doc) -> str:
    symmetry = getattr(doc, "symmetry", None)
    if symmetry is None:
        return "unknown"
    return crystal_system_name(getattr(symmetry, "crystal_system", None))


def safe_formula(value: str) -> str:
    return value.replace("/", "_").replace(" ", "")


def select_docs(docs):
    selected = []
    counts = defaultdict(int)

    for doc in docs:
        crystal = doc_crystal_system(doc)
        magnetic = bool(getattr(doc, "is_magnetic", False))
        group_key = (crystal, magnetic)

        if counts[group_key] >= MAX_PER_GROUP:
            continue

        selected.append(doc)
        counts[group_key] += 1

        if len(selected) >= MAX_TOTAL:
            break

    return selected


def query_parent_structures():
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "nsites",
        "is_magnetic",
        "energy_above_hull",
        "band_gap",
        "elements",
        "symmetry",
    ]

    all_docs = []
    seen_ids = set()

    with MPRester() as mpr:
        for chemsys in CHEMSYS_LIST:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                energy_above_hull=(0, ENERGY_ABOVE_HULL_MAX),
                num_sites=(NUM_SITES_MIN, NUM_SITES_MAX),
                fields=fields,
            )

            for doc in docs:
                mid = str(doc.material_id)
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                all_docs.append(doc)

    # Sort for reproducibility: lower hull energy first, then smaller cells.
    all_docs.sort(
        key=lambda d: (
            float(getattr(d, "energy_above_hull", 999.0) or 999.0),
            int(getattr(d, "nsites", 10**9) or 10**9),
            str(getattr(d, "material_id", "")),
        )
    )
    return select_docs(all_docs)


def write_outputs(docs) -> None:
    rows = []

    for doc in docs:
        material_id = str(doc.material_id)
        formula = str(doc.formula_pretty)
        cif_name = f"{material_id}_{safe_formula(formula)}.cif"
        cif_path = STRUCT_DIR / cif_name
        doc.structure.to(filename=str(cif_path))

        crystal = doc_crystal_system(doc)
        symmetry = getattr(doc, "symmetry", None)
        spacegroup = getattr(symmetry, "symbol", "") if symmetry is not None else ""

        rows.append(
            {
                "material_id": material_id,
                "formula_pretty": formula,
                "nsites": int(doc.nsites),
                "crystal_system": crystal,
                "spacegroup_symbol": spacegroup,
                "is_magnetic": bool(getattr(doc, "is_magnetic", False)),
                "energy_above_hull": float(getattr(doc, "energy_above_hull", 0.0) or 0.0),
                "band_gap": float(getattr(doc, "band_gap", 0.0) or 0.0),
                "elements": ",".join(str(e) for e in getattr(doc, "elements", [])),
                "source": "materials_project",
                "structure_role": "equilibrium_parent",
                "cif_path": str(cif_path),
            }
        )

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "material_id",
                "formula_pretty",
                "nsites",
                "crystal_system",
                "spacegroup_symbol",
                "is_magnetic",
                "energy_above_hull",
                "band_gap",
                "elements",
                "source",
                "structure_role",
                "cif_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(docs) -> None:
    print(f"saved_structures: {len(docs)}")
    print(f"structure_dir: {STRUCT_DIR}")
    print(f"metadata_csv: {META_CSV}")

    crystal_counts = defaultdict(int)
    magnetic_counts = defaultdict(int)
    for doc in docs:
        crystal_counts[doc_crystal_system(doc)] += 1
        magnetic_counts[bool(getattr(doc, "is_magnetic", False))] += 1

    print("crystal_system_counts:")
    for key in sorted(crystal_counts):
        print(f"  {key}: {crystal_counts[key]}")

    print("magnetic_counts:")
    for key in sorted(magnetic_counts):
        print(f"  {key}: {magnetic_counts[key]}")


def main() -> None:
    if not os.environ.get("MP_API_KEY"):
        raise RuntimeError("MP_API_KEY is not set in the environment.")

    ensure_dirs()
    docs = query_parent_structures()
    write_outputs(docs)
    print_summary(docs)


if __name__ == "__main__":
    main()
