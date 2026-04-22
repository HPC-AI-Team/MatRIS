import csv
import random
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


ROOT = Path("/home/lht/lab/mp_testset")
RAW_DIR = ROOT / "raw_structures"
META_DIR = ROOT / "metadata"
INPUT_META_CSV = META_DIR / "mp_testset_metadata.csv"
OUTPUT_META_CSV = META_DIR / "mp_testset_augmented_metadata.csv"

OUT_EQ_DIR = ROOT / "equilibrium"
OUT_PERTURB_DIR = ROOT / "perturbed_small"
OUT_STRAIN_DIR = ROOT / "strained_high"
OUT_MD_INIT_DIR = ROOT / "md_initial"

RNG_SEED = 42
SMALL_PERTURB_STD_ANG = 0.03
SMALL_PERTURB_MAX_ANG = 0.06
STRAIN_FACTORS = (0.94, 1.06)
MD_INIT_COUNT = 10


def ensure_dirs():
    for path in [OUT_EQ_DIR, OUT_PERTURB_DIR, OUT_STRAIN_DIR, OUT_MD_INIT_DIR, META_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_parent_rows():
    with open(INPUT_META_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def copy_equilibrium_structure(src: Path, dst: Path) -> None:
    structure = Structure.from_file(src)
    structure.to(filename=str(dst))


def make_small_perturbation(src: Path, dst: Path, rng: np.random.Generator) -> None:
    structure = Structure.from_file(src)
    cart = structure.cart_coords.copy()
    disp = rng.normal(loc=0.0, scale=SMALL_PERTURB_STD_ANG, size=cart.shape)
    disp = np.clip(disp, -SMALL_PERTURB_MAX_ANG, SMALL_PERTURB_MAX_ANG)
    new_cart = cart + disp
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=new_cart,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )
    new_structure.to(filename=str(dst))


def make_strained_structures(src: Path, dst_low: Path, dst_high: Path) -> None:
    structure = Structure.from_file(src)

    low = structure.copy()
    low.scale_lattice(structure.volume * (STRAIN_FACTORS[0] ** 3))
    low.to(filename=str(dst_low))

    high = structure.copy()
    high.scale_lattice(structure.volume * (STRAIN_FACTORS[1] ** 3))
    high.to(filename=str(dst_high))


def row_with_updates(row: dict, **updates):
    new_row = dict(row)
    new_row.update(updates)
    return new_row


def main():
    ensure_dirs()
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    parent_rows = read_parent_rows()
    augmented_rows = []

    parent_ids = [row["material_id"] for row in parent_rows]
    md_init_ids = set(random.sample(parent_ids, min(MD_INIT_COUNT, len(parent_ids))))

    for row in parent_rows:
        src = Path(row["cif_path"])
        base_name = src.name
        stem = src.stem

        eq_dst = OUT_EQ_DIR / base_name
        copy_equilibrium_structure(src, eq_dst)
        augmented_rows.append(
            row_with_updates(
                row,
                structure_role="equilibrium",
                cif_path=str(eq_dst),
                parent_material_id=row["material_id"],
                perturbation_type="none",
            )
        )

        perturb_dst = OUT_PERTURB_DIR / f"{stem}_perturb_small.cif"
        make_small_perturbation(src, perturb_dst, rng)
        augmented_rows.append(
            row_with_updates(
                row,
                structure_role="perturbed_small",
                cif_path=str(perturb_dst),
                parent_material_id=row["material_id"],
                perturbation_type=f"gaussian_std_{SMALL_PERTURB_STD_ANG}_A",
            )
        )

        strain_low_dst = OUT_STRAIN_DIR / f"{stem}_strain_{STRAIN_FACTORS[0]:.2f}.cif"
        strain_high_dst = OUT_STRAIN_DIR / f"{stem}_strain_{STRAIN_FACTORS[1]:.2f}.cif"
        make_strained_structures(src, strain_low_dst, strain_high_dst)

        augmented_rows.append(
            row_with_updates(
                row,
                structure_role="strained_high",
                cif_path=str(strain_low_dst),
                parent_material_id=row["material_id"],
                perturbation_type=f"isotropic_strain_{STRAIN_FACTORS[0]:.2f}",
            )
        )
        augmented_rows.append(
            row_with_updates(
                row,
                structure_role="strained_high",
                cif_path=str(strain_high_dst),
                parent_material_id=row["material_id"],
                perturbation_type=f"isotropic_strain_{STRAIN_FACTORS[1]:.2f}",
            )
        )

        if row["material_id"] in md_init_ids:
            md_dst = OUT_MD_INIT_DIR / base_name
            copy_equilibrium_structure(src, md_dst)
            augmented_rows.append(
                row_with_updates(
                    row,
                    structure_role="md_initial",
                    cif_path=str(md_dst),
                    parent_material_id=row["material_id"],
                    perturbation_type="none_md_seed",
                )
            )

    with open(OUTPUT_META_CSV, "w", newline="", encoding="utf-8") as f:
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
                "parent_material_id",
                "perturbation_type",
            ],
        )
        writer.writeheader()
        writer.writerows(augmented_rows)

    role_counts = {}
    for row in augmented_rows:
        role = row["structure_role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"input_equilibrium_parents: {len(parent_rows)}")
    print(f"output_metadata_csv: {OUTPUT_META_CSV}")
    print("generated_counts:")
    for key in sorted(role_counts):
        print(f"  {key}: {role_counts[key]}")


if __name__ == "__main__":
    main()
