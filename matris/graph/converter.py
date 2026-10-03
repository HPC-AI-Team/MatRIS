"""CPU training graphs and CUDA inference graphs with shared topology semantics."""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Structure
from torch import nn

from .radiusgraph import RadiusGraph


def build_graph_tensors(natoms, centers, neighbors, images, distances, *, line_cutoff):
    """Pair periodic reverse edges and enumerate directed angles with NumPy."""
    centers = np.asarray(centers, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    images = np.asarray(images, dtype=np.int64)
    distances = np.asarray(distances)
    keys = np.column_stack((centers, neighbors, images))
    reverse_keys = np.column_stack((neighbors, centers, -images))
    order = np.lexsort(keys[:, ::-1].T)
    reverse_order = np.lexsort(reverse_keys[:, ::-1].T)
    reverse = np.empty(len(centers), dtype=np.int64)
    reverse[order] = reverse_order
    edges = np.arange(len(centers))
    if not np.array_equal(keys[order], reverse_keys[reverse_order]) or np.any(
        reverse == edges
    ):
        raise ValueError("Periodic neighbor list is missing a reverse edge.")
    u2d, d2u = np.unique(np.minimum(edges, reverse), return_inverse=True)

    # Center-major ordering matches the CUDA builder. Only angle-sized arrays
    # are allocated; no dense atom-by-atom or edge-by-edge adjacency is built.
    by_center = np.argsort(centers, kind="stable")
    left = by_center[distances[by_center] <= line_cutoff]
    right = by_center[distances[by_center] < line_cutoff]
    counts = np.bincount(centers[right], minlength=natoms)
    starts = np.cumsum(counts) - counts
    repeats = counts[centers[left]]
    row_starts = np.cumsum(repeats) - repeats
    source = np.repeat(left, repeats)
    target = right[
        np.arange(repeats.sum())
        + np.repeat(starts[centers[left]] - row_starts, repeats)
    ]
    distinct = source != target
    source, target = source[distinct], target[distinct]
    atom_graph = np.column_stack((centers, neighbors)).astype(np.int32)
    line_graph = np.column_stack(
        (centers[source], d2u[source], source, d2u[target], target)
    ).astype(np.int32)
    return tuple(
        torch.from_numpy(array)
        for array in (
            atom_graph,
            d2u.astype(np.int32),
            u2d.astype(np.int32),
            line_graph,
        )
    )


class GraphConverter(nn.Module):
    """Convert structures to graph tensors on CPU, or CUDA during fast inference."""

    def __init__(
        self,
        atom_graph_cutoff: float = 6,
        line_graph_cutoff: float = 4,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.atom_graph_cutoff = atom_graph_cutoff
        self.line_graph_cutoff = (
            atom_graph_cutoff if line_graph_cutoff is None else line_graph_cutoff
        )
        if verbose:
            print(self)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(atom_graph_cutoff={self.atom_graph_cutoff}, "
            f"line_graph_cutoff={self.line_graph_cutoff})"
        )

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
        atomic_numbers: (
            np.ndarray | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        frac_coords: np.ndarray | torch.Tensor | None = None,
        lattice_matrix: np.ndarray | torch.Tensor | None = None,
        cart_coords: np.ndarray | torch.Tensor | None = None,
        *,
        mode: str = "fast",
        device: str | torch.device | None = None,
    ) -> RadiusGraph:
        """Training and DataLoader workers always build on CPU.

        ``eval()`` with a CUDA device and ``mode="fast"`` uses GPU neighbor
        search and CUDA graph assembly. Torch mode and CPU inference use NumPy.
        """
        if mode not in ("fast", "torch"):
            raise ValueError(f"Unknown mode {mode!r}; choose 'fast' or 'torch'.")
        target_device = (
            torch.device(device) if device is not None else self._device_anchor.device
        )
        use_cuda = (
            not self.training
            and mode == "fast"
            and target_device.type == "cuda"
            and torch.utils.data.get_worker_info() is None
        )
        graph_device = target_device if use_cuda else torch.device("cpu")
        if atomic_numbers is None:
            atomic_numbers = np.fromiter(
                (site.specie.Z for site in structure),
                dtype=np.int32,
                count=len(structure),
            )
        if frac_coords is None:
            frac_coords = structure.frac_coords
        if lattice_matrix is None:
            lattice_matrix = structure.lattice.matrix
        atomic_number = torch.asarray(
            atomic_numbers, dtype=torch.int32, device=graph_device, copy=True
        )
        atom_frac_coord = torch.asarray(
            frac_coords, dtype=torch.float32, device=graph_device, copy=True
        )
        lattice = torch.asarray(
            lattice_matrix, dtype=torch.float32, device=graph_device, copy=True
        )

        if use_cuda:
            from ..model.op.graph import build_graph_tensors_gpu, gpu_neighbor_list

            edges, ptr, image, distance = gpu_neighbor_list(
                structure.cart_coords if cart_coords is None else cart_coords,
                lattice_matrix,
                self.atom_graph_cutoff,
                device=target_device,
                pbc=structure.lattice.pbc,
            )
            tensors = build_graph_tensors_gpu(
                edges, ptr, image, distance, line_cutoff=self.line_graph_cutoff
            )
        else:
            centers, neighbors, image, distance = structure.get_neighbor_list(
                r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
            )
            tensors = build_graph_tensors(
                len(structure),
                centers,
                neighbors,
                image,
                distance,
                line_cutoff=self.line_graph_cutoff,
            )
        atom_graph, directed2undirected, undirected2directed, line_graph = tensors
        return RadiusGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.as_tensor(
                image, dtype=torch.float32, device=graph_device
            ),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            line_graph=line_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "line_graph_cutoff": self.line_graph_cutoff,
        }

    @classmethod
    def from_dict(cls, config: dict) -> GraphConverter:
        return cls(**config)
