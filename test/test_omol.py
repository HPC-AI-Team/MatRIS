from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from ase.build import molecule

from matris.applications.base import MatRISCalculator
from matris.model import MatRIS


class TestOMol(unittest.TestCase):
    def test_checkpoint_inference_uses_electronic_state(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            model = MatRIS(
                num_layers=1, node_feat_dim=16, edge_feat_dim=16,
                three_body_feat_dim=16, mlp_hidden_dims=(16, 16),
                norm_type="layer", three_body_expansion="legendre", smooth_attention=True,
                use_charge_spin_embedding=True, use_long_range=True,
            )
        config = dict(model.config)
        del config["use_charge_spin_embedding"]
        config.update(
            smooth=True, smooth_attention=False, use_long_range=True,
            long_range_dim=8, long_range_centers=4, long_range_length=24.0,
            use_charge_embedding=True, use_spin_embedding=True,
            diffusion_pretraining=False, line_graph_amp=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "omol.pth.tar"
            torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
            calc = MatRISCalculator(checkpoint, task="ef", device="cpu", mode="torch")
            self.assertEqual(calc.model.config, model.config)
            atoms = molecule("H2O")
            atoms.calc = calc
            neutral = atoms.get_potential_energy()
            self.assertNotIn("charge", atoms.info)
            self.assertNotIn("spin", atoms.info)
            self.assertTrue(np.isfinite(atoms.get_forces()).all())
            atoms.info.update(charge=None, spin=None, total_charge=1, spin_multiplicity=2)
            calc.reset()
            self.assertEqual(atoms.get_potential_energy(), neutral)
            atoms.info.update(charge=0, spin=1)
            self.assertEqual(calc.check_state(atoms), [])
            atoms.info["spin"] = 0
            self.assertNotAlmostEqual(atoms.get_potential_energy(), neutral, places=6)
            atoms.info.update(charge=1, spin=2)
            self.assertNotAlmostEqual(atoms.get_potential_energy(), neutral, places=6)
            atoms.info.update(charge=0, spin=1)
            self.assertEqual(atoms.get_potential_energy(), neutral)


if __name__ == "__main__":
    unittest.main()
