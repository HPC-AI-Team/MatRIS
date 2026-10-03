"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

datatype = torch.float32

class RadiusGraph:
    def __init__(
        self,
        graph_id: str | None,
        mp_id: str | None,
        composition: str | None,
        atomic_number: Tensor,
        atom_frac_coord: Tensor,
        lattice: Tensor,
        neighbor_image: Tensor,
        atom_graph: Tensor,
        atom_graph_cutoff: float,
        line_graph: Tensor,
        line_graph_cutoff: float,
        directed2undirected: Tensor,
        undirected2directed: Tensor,
    ):
        """Initialize a RadiusGraph object representing a material graph.
        
        Args:
            graph_id (str | None): Unique identifier for this graph instance.
            mp_id (str | None): Materials Project ID associated with this structure.
            composition (str | None): Chemical composition of the compound.

            atomic_number (Tensor): Atomic numbers of all atoms in the structure. 
                Shape: [n_atom]

            atom_frac_coord (Tensor): Fractional atomic coordinates within the unit cell.
                Shape: [n_atom, 3]

            lattice (Tensor): Lattice vectors defining the periodic cell.
                Shape: [3, 3]

            neighbor_image (Tensor): Periodic image offsets for each directed edge, 
                indicating how many unit cells away the neighbor lies along each axis.
                Shape: [n_directed_edges, 3]

            atom_graph (Tensor): Directed atomic graph, where each row stores 
                (center_atom_index, neighbor_atom_index).
                Shape: [n_directed_edges, 2]

            atom_graph_cutoff (float): Cutoff radius (in Å) used to build the atom graph.

            line_graph (Tensor): Line graph describing angular connections between bonds.
                Each row stores 
                (central_atom, undirected_bond_1, directed_bond_1, 
                 undirected_bond_2, directed_bond_2).
                Shape: [n_angle, 5]

            line_graph_cutoff (float): Cutoff radius (in Å) used to build the line graph.

            directed2undirected (Tensor): Mapping from each directed edge index 
                to its corresponding undirected edge index.
                Shape: [n_directed_edges]

            undirected2directed (Tensor): Mapping from each undirected edge index 
                to one of its directed edge indices.
                Shape: [n_undirected_edges]
        """
        super().__init__()
        self.graph_id = graph_id
        self.mp_id = mp_id
        self.composition = composition
        
        self.atomic_number = atomic_number
        self.atom_frac_coord = atom_frac_coord
        self.lattice = lattice
        self.neighbor_image = neighbor_image
        self.line_graph = line_graph
        self.line_graph_cutoff = line_graph_cutoff
        self.atom_graph = atom_graph
        self.atom_graph_cutoff = atom_graph_cutoff
        self.directed2undirected = directed2undirected
        self.undirected2directed = undirected2directed
        
        assert len(directed2undirected) == 2 * len(undirected2directed), (
            f"Number of directed indices ({len(directed2undirected)}) != "
            f"2 * number of undirected indices ({2 * len(undirected2directed)})!"
        )
        
    def to(self, device: str = "cpu") -> RadiusGraph:
        """Move the graph to a device."""
        return RadiusGraph(
            graph_id=self.graph_id,
            mp_id=self.mp_id,
            composition=self.composition,
            atomic_number=self.atomic_number.to(device),
            atom_frac_coord=self.atom_frac_coord.to(device),
            lattice=self.lattice.to(device),
            neighbor_image=self.neighbor_image.to(device),
            atom_graph=self.atom_graph.to(device),
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph=self.line_graph.to(device),
            line_graph_cutoff=self.line_graph_cutoff,
            directed2undirected=self.directed2undirected.to(device),
            undirected2directed=self.undirected2directed.to(device),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "mp_id": self.mp_id,
            "composition": self.composition,
            "atomic_number": self.atomic_number,
            "atom_frac_coord": self.atom_frac_coord,
            "lattice": self.lattice,
            "neighbor_image": self.neighbor_image,
            "atom_graph": self.atom_graph,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "line_graph": self.line_graph,
            "line_graph_cutoff": self.line_graph_cutoff,
            "directed2undirected": self.directed2undirected,
            "undirected2directed": self.undirected2directed,
        }

    def save(self, fname: str | None = None, save_dir: str = ".") -> str:
        """Save the Radiusgraph to a file.

        Args:
            fname (str, optional): File name. Defaults to None.
            save_dir (str, optional): Directory to save the file. Defaults to ".".
        """
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        torch.save(self.to_dict(), f=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name: str) -> RadiusGraph:
        """Load a Radiusgraph from a file.

        Args:
            file_name (str): The path to the file.
        """
        return cls(**torch.load(file_name, weights_only=True))

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> RadiusGraph:
        """Load a RadiusGraph from a dictionary."""
        return RadiusGraph(**dic)

    def __repr__(self) -> str:
        """String representation of the graph."""
        composition = self.composition
        atom_graph_cutoff = self.atom_graph_cutoff
        line_graph_cutoff = self.line_graph_cutoff
        atom_graph_len = self.atom_graph
        n_atoms = len(self.atomic_number)
        atom_graph_len = len(self.atom_graph)
        line_graph_len = len(self.line_graph)
        return (
            f"RadiusGraph({composition=}, {atom_graph_cutoff=}, {line_graph_cutoff=}, "
            f"{n_atoms=}, {atom_graph_len=}, {line_graph_len=})"
        )
