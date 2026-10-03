"""GPU neighbor capacity regression; requires the optional fast dependencies."""

import importlib.util
import unittest

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("nvalchemiops"),
    "CUDA and nvalchemiops are required",
)
class TestGPUGraph(unittest.TestCase):
    def test_empty_directed_average(self):
        from matris.model.functions import directed_average

        features = torch.empty((0, 32), device="cuda", requires_grad=True)
        indices = torch.empty(0, dtype=torch.int64, device="cuda")
        result = directed_average(features, indices, 0, fast=True)
        self.assertEqual(result.shape, (0, 32))
        result.sum().backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_partial_periodic_converter(self):
        from pymatgen.core import Lattice, Structure
        from matris.graph import GraphConverter

        converter = GraphConverter().eval().cuda()
        for pbc in ((True, False, False), (True, True, False), (False, False, False)):
            with self.subTest(pbc=pbc):
                structure = Structure(
                    Lattice([[4, 0, 0], [0.5, 4, 0], [0.25, 0.5, 4]], pbc=pbc),
                    ["Si", "Si"], [[0, 0, 0], [0.4, 0.4, 0.4]],
                )
                expected = converter(structure, mode="torch")
                actual = converter(structure, mode="fast")
                rows = []
                for graph in (expected, actual):
                    edges = np.column_stack((
                        graph.atom_graph.cpu().numpy(), graph.neighbor_image.cpu().numpy(),
                    ))
                    rows.append(edges[np.lexsort(edges[:, ::-1].T)])
                np.testing.assert_array_equal(rows[1], rows[0])
                self.assertEqual(torch.count_nonzero(
                    actual.neighbor_image[:, np.logical_not(pbc)]
                ).item(), 0)

    def test_boundary_inference(self):
        from pymatgen.core import Lattice, Structure
        from matris.model import MatRIS

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = MatRIS(
                num_layers=1, node_feat_dim=32, edge_feat_dim=32,
                three_body_feat_dim=32, mlp_hidden_dims=(32, 32), norm_type="layer",
            ).eval().cuda().requires_grad_(False)
        structures = [
            Structure(Lattice.cubic(40), ["Si"], [[0, 0, 0]]),
            Structure(Lattice.cubic(40), ["Si", "O"], [[0, 0, 0], [0.5, 0, 0]]),
        ]
        for pbc in ((True, False, False), (True, True, False)):
            structures.append(Structure(
                Lattice([[4, 0, 0], [0.5, 4, 0], [0.25, 0.5, 4]], pbc=pbc),
                ["Si", "Si"], [[0, 0, 0], [0.4, 0.4, 0.4]],
            ))
        for index, structure in enumerate(structures):
            with self.subTest(index=index, pbc=structure.lattice.pbc):
                expected_graph = model.graph_converter(structure, mode="torch").to("cuda")
                actual_graph = model.graph_converter(structure, mode="fast")
                expected = model([expected_graph], task="efs", mode="torch")
                actual = model([actual_graph], task="efs", mode="fast")
                for key in ("e", "f", "s"):
                    reference = expected[key] if key == "e" else expected[key][0]
                    value = actual[key] if key == "e" else actual[key][0]
                    self.assertTrue(torch.isfinite(value).all().item())
                    torch.testing.assert_close(value, reference, atol=2e-5, rtol=1e-4)
                if index < 2:
                    self.assertEqual(actual_graph.atom_graph.shape[0], 0)
                    torch.testing.assert_close(actual["f"][0], torch.zeros_like(actual["f"][0]))
                    torch.testing.assert_close(actual["s"][0], torch.zeros_like(actual["s"][0]))

    def test_isolated_and_periodic_atoms(self):
        from matris.graph.converter import build_graph_tensors
        from matris.model.op.graph import build_graph_tensors_gpu, gpu_neighbor_list

        structures = [
            Atoms("Si", positions=[[0, 0, 0]], cell=[40, 40, 40]),
            Atoms("Si3", positions=[[0, 0, 0], [2.3, 0, 0], [20, 0, 0]], cell=[40, 40, 40]),
            bulk("Al", "fcc", a=4.05),  # One atom with periodic neighbors is not isolated.
        ]
        for atoms in structures:
            with self.subTest(natoms=len(atoms), pbc=atoms.pbc.tolist()):
                edges, ptr, images, distances = gpu_neighbor_list(
                    atoms.positions, atoms.cell.array, 6.0, device="cuda", pbc=atoms.pbc,
                )
                expected = build_graph_tensors(
                    len(atoms), *edges.cpu().numpy(), images.cpu().numpy(),
                    distances.cpu().numpy(), line_cutoff=4.0,
                )
                actual = build_graph_tensors_gpu(edges, ptr, images, distances, line_cutoff=4.0)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result.cpu(), reference)
                if atoms.pbc.all():
                    self.assertGreater(edges.shape[1], 0)

    def test_dense_periodic_neighbors(self):
        from matris.model.op.graph import gpu_neighbor_list

        for lattice in (2.4, 1.5):
            with self.subTest(lattice=lattice):
                atoms = bulk("Al", "fcc", a=lattice, cubic=True)
                i, j, shifts, distance = neighbor_list("ijSd", atoms, 6.0 + 1e-8)
                self.assertGreater(np.bincount(i).max(), 192)
                edges, ptr, images, actual_distance = gpu_neighbor_list(
                    atoms.positions, atoms.cell.array, 6.0, device="cuda"
                )
                expected = np.column_stack((i, j, shifts))
                actual = np.column_stack((edges.cpu().numpy().T, images.cpu().numpy()))
                expected_order = np.lexsort(expected[:, ::-1].T)
                actual_order = np.lexsort(actual[:, ::-1].T)
                np.testing.assert_array_equal(actual[actual_order], expected[expected_order])
                np.testing.assert_array_equal(np.diff(ptr.cpu().numpy()), np.bincount(i))
                np.testing.assert_allclose(
                    actual_distance.cpu().numpy()[actual_order], distance[expected_order],
                    atol=1e-12, rtol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
