"""Optional TorchSim regression: python -m unittest discover -s test."""

import importlib.util
import unittest

import torch
from ase.build import bulk

from matris.model import MatRIS


@unittest.skipUnless(importlib.util.find_spec("torch_sim"), "TorchSim is optional")
class TestTorchSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch_sim as ts
        from matris.applications.torchsim import MatRISTorchSimModel

        torch.manual_seed(17)
        model = MatRIS(num_layers=1, node_feat_dim=32, edge_feat_dim=32,
                       three_body_feat_dim=32, mlp_hidden_dims=(32, 32))
        cls.model = MatRISTorchSimModel(model, mode="torch")
        cls.state = ts.io.atoms_to_state(
            [bulk("Si", "diamond", a=5.43, cubic=True)] * 2,
            device=torch.device("cpu"), dtype=torch.float32,
        )

    def test_interface(self):
        from torch_sim.models.interface import validate_model_outputs

        validate_model_outputs(self.model, self.model.device, self.model.dtype)

    def test_targets_and_units(self):
        from matris.applications.torchsim import MatRISTorchSimModel
        from pymatgen.io.ase import AseAtomsAdaptor

        model = self.model
        graph = model.model.graph_converter(
            AseAtomsAdaptor.get_structure(bulk("Si", "diamond", a=5.43, cubic=True)),
            mode="torch",
        )
        expected = model.model([graph, graph], task="efsm", mode="torch")
        for target in ("e", "em", "ef", "efs", "efsm"):
            model = MatRISTorchSimModel(self.model.model, mode="torch", target=target)
            with torch.no_grad():
                actual = model(self.state)
            self.assertEqual(set(actual), {name for key, name in zip(
                "efsm", ("energy", "forces", "stress", "magmoms")) if key in target})
            self.assertEqual(model.compute_forces, "f" in target)
            self.assertEqual(model.compute_stress, "s" in target)
            self.assertTrue(all(not value.requires_grad for value in actual.values()))
            torch.testing.assert_close(actual["energy"], expected["e"] * 8)
            if "f" in target:
                torch.testing.assert_close(actual["forces"], torch.cat(expected["f"]), atol=2e-5, rtol=1e-4)
            if "s" in target:
                torch.testing.assert_close(actual["stress"], torch.stack(expected["s"]) / 160.21766208,
                                           atol=1e-6, rtol=1e-4)
            if "m" in target:
                self.assertEqual(actual["magmoms"].shape, (16,))
                torch.testing.assert_close(actual["magmoms"], torch.cat(expected["m"]))
        for target in ("", "es", "fsm", "unknown"):
            with self.assertRaisesRegex(ValueError, "target must be"):
                MatRISTorchSimModel(self.model.model, target=target)

    def test_magnetic_batch(self):
        import torch_sim as ts
        from ase.build import molecule
        from matris.applications.torchsim import MatRISTorchSimModel

        model = MatRISTorchSimModel(self.model.model, mode="torch", target="efsm")
        atoms = [molecule("H2O"), molecule("H2")]
        singles = [model(ts.io.atoms_to_state(
            a, device=model.device, dtype=model.dtype,
        )) for a in atoms]
        for order in ([0, 1], [1, 0]):
            actual = model(ts.io.atoms_to_state(
                [atoms[i] for i in order], device=model.device, dtype=model.dtype,
            ))
            self.assertEqual(actual["magmoms"].shape, (5,))
            for key in actual:
                torch.testing.assert_close(
                    actual[key], torch.cat([singles[i][key] for i in order]),
                    atol=2e-5, rtol=1e-4,
                )

    def test_inference_mode(self):
        import torch_sim as ts

        expected = self.model(self.state)
        with torch.inference_mode():
            state = ts.io.atoms_to_state(
                [bulk("Si", "diamond", a=5.43, cubic=True)] * 2,
                device=self.model.device, dtype=self.model.dtype,
            )
            state.atomic_numbers = state.atomic_numbers.to(torch.int32)
            actual = self.model(state)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_missing_nonperiodic_cell(self):
        import torch_sim as ts
        from ase import Atoms

        for cell, pbc in [([5.43, 5.43, 0], [True, True, False]),
                          ([5.43, 0, 0], [True, False, False])]:
            atoms = Atoms("Si2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]],
                          cell=cell, pbc=pbc)
            state = ts.io.atoms_to_state(atoms, device=self.model.device, dtype=self.model.dtype)
            actual = self.model(state)
            atoms.set_cell([5.43 if periodic else 30 for periodic in pbc])
            expected = self.model(ts.io.atoms_to_state(
                atoms, device=self.model.device, dtype=self.model.dtype,
            ))
            for key in ("energy", "forces"):
                torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(state.row_vector_cell[0].diag(), torch.tensor(cell, dtype=self.model.dtype))

    def test_nonperiodic_mixed_batch(self):
        import torch_sim as ts
        from ase.build import molecule

        atoms = [molecule(name) for name in ("H2O", "H2", "CH4")]
        singles = [self.model(ts.io.atoms_to_state(
            a, device=self.model.device, dtype=self.model.dtype,
        )) for a in atoms]
        for order in ([0, 1, 2], [1, 2, 0]):
            state = ts.io.atoms_to_state(
                [atoms[i] for i in order], device=self.model.device, dtype=self.model.dtype,
            )
            actual = self.model(state)
            for key in actual:
                expected = torch.cat([singles[i][key] for i in order])
                torch.testing.assert_close(actual[key], expected, atol=1e-5, rtol=1e-4)
            self.assertFalse(state.pbc.any())
            self.assertEqual(torch.count_nonzero(actual["stress"]).item(), 0)

    def test_disconnected_molecule_additivity(self):
        import torch_sim as ts
        from ase.build import molecule

        water, hydrogen = molecule("H2O"), molecule("H2")
        hydrogen.translate([32, 0, 0])
        outputs = [self.model(ts.io.atoms_to_state(
            atoms, device=self.model.device, dtype=self.model.dtype,
        )) for atoms in (water, hydrogen, water + hydrogen)]
        torch.testing.assert_close(outputs[2]["energy"],
                                   outputs[0]["energy"] + outputs[1]["energy"],
                                   atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(outputs[2]["forces"],
                                   torch.cat([out["forces"] for out in outputs[:2]]),
                                   atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
