"""Relaxation must preserve molecular and periodic structure representations."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from ase.build import bulk, molecule
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matris.applications.relax import StructOptimizer
from matris.model import MatRIS


def make_optimizer(path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = MatRIS(
            num_layers=1, node_feat_dim=32, edge_feat_dim=32,
            three_body_feat_dim=32, mlp_hidden_dims=(32, 32),
        )
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
    return StructOptimizer(path, task="efsm", device="cpu", mode="torch")


def check_relax_structure_type(optimizer, case):
    if case in ("partial_pbc", "crystal"):
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        atoms.positions[0] += [0.03, -0.02, 0.01]
        if case == "partial_pbc":
            atoms.pbc = [True, True, False]
        expected_type = Structure
    else:
        atoms = molecule("H2O")
        if case == "vacuum":
            atoms.center(vacuum=5)
        expected_type = Molecule

    cell, pbc = atoms.cell.copy(), atoms.pbc.copy()
    original_positions = atoms.positions.copy()
    input_structure = AseAtomsAdaptor.get_molecule(atoms) if case == "molecule" else atoms
    result = optimizer.relax(
        input_structure, relax_cell=case == "crystal", fmax=0, steps=1, verbose=False,
    )
    final = result["final_structure"]
    trajectory = result["trajectory"]
    assert isinstance(final, expected_type)
    assert len(final) == len(atoms)
    assert np.isfinite(trajectory.energies).all()
    np.testing.assert_allclose(final.cart_coords, trajectory.atom_positions[-1])
    np.testing.assert_allclose(final.site_properties["magmom"], trajectory.magmoms[-1])
    assert not np.array_equal(final.cart_coords, original_positions)
    if case != "crystal":
        np.testing.assert_array_equal(trajectory.atoms.cell, cell)
    np.testing.assert_array_equal(trajectory.atoms.pbc, pbc)
    if expected_type is Structure:
        np.testing.assert_allclose(final.lattice.matrix, trajectory.cells[-1])
        assert final.lattice.pbc == tuple(pbc)


class TestRelax(unittest.TestCase):
    def test_structure_type(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = make_optimizer(Path(directory) / "model.pth.tar")
            for case in ("zero_cell", "vacuum", "molecule", "partial_pbc", "crystal"):
                with self.subTest(case=case):
                    check_relax_structure_type(optimizer, case)


if __name__ == "__main__":
    unittest.main()
