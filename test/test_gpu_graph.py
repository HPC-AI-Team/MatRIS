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
