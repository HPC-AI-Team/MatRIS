"""Independent tensor layouts must share one dynamic compiled graph."""

import unittest

import torch
import matris.model.compile  # Apply the production compilation defaults.


class TestCompile(unittest.TestCase):
    def test_independent_dimensions(self):
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        def reduce_features(volumes, vectors):
            return volumes.sum() + vectors.sum()

        compiled = torch.compile(reduce_features, dynamic=True, backend=backend)
        for batch_size in [3, 2, 4, 3]:
            volumes = torch.ones(batch_size)
            vectors = torch.ones(7, 3)
            torch.testing.assert_close(compiled(volumes, vectors), torch.tensor(float(batch_size + 21)))
        self.assertEqual(len(graphs), 1)

    def test_independent_offsets(self):
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        def reduce_indices(atom_index, line_index):
            return atom_index.sum() + line_index.sum()

        compiled = torch.compile(reduce_indices, dynamic=True, backend=backend)
        for atom_offset, line_offset in [(24, 24), (24, 48), (32, 48), (48, 48)]:
            atom_index = torch.ones(128)[atom_offset:atom_offset + 16]
            line_index = torch.ones(128)[line_offset:line_offset + 16]
            torch.testing.assert_close(compiled(atom_index, line_index), torch.tensor(32.))
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
