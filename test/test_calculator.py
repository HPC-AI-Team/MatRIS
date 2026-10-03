"""Freezing model parameters must preserve ASE energy, forces and stress."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

from matris.applications.base import MatRISCalculator
from matris.model import MatRIS


class TestCalculator(unittest.TestCase):
    def test_nonperiodic_cell_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth.tar"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                model = MatRIS(
                    num_layers=1, node_feat_dim=32, edge_feat_dim=32,
                    three_body_feat_dim=32, mlp_hidden_dims=(32, 32),
                )
            torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
            rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
            structures = [molecule("H2O")]
            for pbc in ([True, False, False], [True, True, False]):
                structures.append(Atoms(
                    "Si2", positions=[[0., 0., 0.], [1.4, 1.4, 1.4]],
                    cell=np.diag(np.array(pbc) * 5.43), pbc=pbc,
                ))
            devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
            for device in devices:
                mode = "fast" if device == "cuda" else "torch"
                calculator = MatRISCalculator(path, task="ef", device=device, mode=mode)
                for original in structures:
                    with self.subTest(device=device, pbc=original.pbc.tolist()):
                        original.calc = calculator
                        energy, forces = original.get_potential_energy(), original.get_forces()
                        atoms = original.copy()
                        atoms.positions[:] = original.positions @ rotation
                        atoms.set_cell(original.cell.array @ rotation)
                        positions, cell, pbc = atoms.positions.copy(), atoms.cell.copy(), atoms.pbc.copy()
                        atoms.calc = calculator
                        np.testing.assert_allclose(atoms.get_potential_energy(), energy, atol=2e-5, rtol=1e-4)
                        np.testing.assert_allclose(atoms.get_forces(), forces @ rotation, atol=2e-5, rtol=1e-4)
                        np.testing.assert_array_equal(atoms.positions, positions)
                        np.testing.assert_array_equal(atoms.cell, cell)
                        np.testing.assert_array_equal(atoms.pbc, pbc)

    def test_frozen_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth.tar"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                model = MatRIS(
                    num_layers=1, node_feat_dim=32, edge_feat_dim=32,
                    three_body_feat_dim=32, mlp_hidden_dims=(32, 32),
                )
            torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
            atoms = bulk("Si", "diamond", a=5.43, cubic=True)
            atoms.positions[0] += [0.03, -0.02, 0.01]
            devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
            for device in devices:
                modes = ("torch", "fast") if device == "cuda" else ("torch",)
                for mode in modes:
                    for checkpoint in (False, True):
                        with self.subTest(device=device, mode=mode, checkpoint=checkpoint):
                            calculator = MatRISCalculator(
                                path, task="efsm", device=device, mode=mode,
                                activation_checkpoint=checkpoint,
                            )
                            self.assertFalse(any(p.requires_grad for p in calculator.model.parameters()))
                            calculator.model.requires_grad_(True)
                            calculator.calculate(atoms)
                            expected = calculator.results.copy()
                            calculator.model.requires_grad_(False)
                            calculator.reset()
                            atoms.calc = calculator
                            atoms.get_forces()
                            for key, value in expected.items():
                                self.assertTrue(np.isfinite(calculator.results[key]).all())
                                np.testing.assert_allclose(
                                    calculator.results[key], value, rtol=1e-5, atol=1e-6,
                                )
                            self.assertTrue(all(p.grad is None for p in calculator.model.parameters()))


if __name__ == "__main__":
    unittest.main()
