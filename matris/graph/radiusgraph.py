"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor
import sys
from abc import ABC, abstractmethod

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
        return torch.load(file_name)

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
    

class Node:
    """A node in graph."""

    def __init__(self, index: int, info: dict | None = None) -> None:
        """Initialize a Node.

        Args:
            index (int): the index of this node
            info (dict, optional): any additional information about this node.
        """
        self.index = index
        self.info = info
        self.neighbors: dict[int, list[DirectedEdge | UndirectedEdge]] = {}

    def add_neighbor(self, index, edge):
        """Draw an directed edge between self and the node specified by index.

        Args:
            index (int): the index of neighboring node
            edge (DirectedEdge): an DirectedEdge object pointing from self to the node.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class Edge(ABC):
    """Abstract base class for edges in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        """Initialize an Edge."""
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self):
        """String representation of this edge."""
        nodes, index, info = self.nodes, self.index, self.info
        return f"{type(self).__name__}({nodes=}, {index=}, {info=})"

    def __hash__(self) -> int:
        """Hash this edge."""
        img = (self.info or {}).get("image")
        img_str = "" if img is None else img.tobytes()
        return hash((self.nodes[0], self.nodes[1], img_str))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two edges are equal."""
        raise NotImplementedError


class UndirectedEdge(Edge):
    """An undirected/bi-directed edge in a graph."""

    __hash__ = Edge.__hash__

    def __eq__(self, other):
        """Check if two undirected edges are equal."""
        return set(self.nodes) == set(other.nodes) and self.info == other.info


class DirectedEdge(Edge):
    """A directed edge in a graph."""

    __hash__ = Edge.__hash__

    def make_undirected(self, index: int, info: dict | None = None) -> UndirectedEdge:
        """Make a directed edge undirected."""
        info = info or {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def __eq__(self, other: object) -> bool:
        """Check if the two directed edges are equal.

        Args:
            other (DirectedEdge): another DirectedEdge to compare to

        Returns:
            bool: True if other is the same directed edge, or if other is the directed
                edge with reverse direction of self, else False.
        """
        self_img = (self.info or {}).get("image")
        other_img = (other.info or {}).get("image")
        none_img = self_img is other_img is None
        if self.nodes == other.nodes and (none_img or all(self_img == other_img)):
            # the image key here is provided by Pymatgen, which refers to the periodic
            # cell the neighbor node comes from

            # In this case the two directed edges are exactly the same, but this is not
            # supposed tp happen unless there's a bug in Pymatgen. (we will never add
            # the same edge twice in creating a crystal graph.)
            print(
                "!!!!!! the two directed edges are equal but this operation is "
                "not supposed to happen",
                file=sys.stderr,
            )
            return True

        # In this case the first edge is from node i to j and the second edge is
        # from node j to i
        return self.nodes == other.nodes[::-1] and (
            none_img or all(self_img == -1 * other_img)
        )


class Graph:
    """A graph for storing the neighbor information of atoms."""

    def __init__(self, nodes: list[Node]) -> None:
        """Initialize a Graph from a list of nodes."""
        self.nodes = nodes
        self.directed_edges: dict[frozenset[int], list[DirectedEdge]] = {}
        self.directed_edges_list: list[DirectedEdge] = []
        self.undirected_edges: dict[frozenset[int], list[UndirectedEdge]] = {}
        self.undirected_edges_list: list[UndirectedEdge] = []

    def add_edge(self, center_index, neighbor_index, image, distance) -> None:
        """Add an directed edge to the graph.

        Args:
            center_index (int): center node index
            neighbor_index (int): neighbor node index
            image (np.array): the periodic cell image the neighbor is from
            distance (float): distance between center and neighbor.
        """
        # Create directed_edge (DE) index using the length of added DEs
        directed_edge_index = len(self.directed_edges_list)

        # Create the DE with info
        this_directed_edge = DirectedEdge(
            [center_index, neighbor_index],
            index=directed_edge_index,
            info={"image": image, "distance": distance},
        )
        # Record the two ends of the DE and see if they have been added
        # Here the sequence of the ends doesn't matter since we want to
        #  search for associated undirected_edge (UDE)
        tmp = frozenset([center_index, neighbor_index])
        # This combination of two ends hasn't been added
        if tmp not in self.undirected_edges:
            # Create UDE index
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            # Create UDE
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp] = [this_undirected_edge]
            
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)
        else:
            # This pair of nodes has been added before, we need to see if this time,
            # 1. it's the other directed edge of the same undirected edge or,
            # 2. it's another totally different undirected edge that has
            # different image and distance (this is possible consider periodicity)
            for undirected_edge in self.undirected_edges[tmp]:
                if (
                    abs(undirected_edge.info["distance"] - distance) < 1e-6
                    and len(undirected_edge.info["directed_edge_index"]) == 1
                ):
                    # There is an undirected edge with similar length and only one of
                    # the directed edges associated has been added
                    added_DE = self.directed_edges_list[
                        undirected_edge.info["directed_edge_index"][0]
                    ]

                    # See if the DE that's associated to this UDE
                    # is the reverse of our DE
                    if added_DE == this_directed_edge:
                        # Add UDE index to this DE
                        this_directed_edge.info[
                            "undirected_edge_index"
                        ] = added_DE.info["undirected_edge_index"]

                        # At the center node, draw edge with this DE
                        self.nodes[center_index].add_neighbor(
                            neighbor_index, this_directed_edge
                        )

                        # Add this DE to DE list
                        self.directed_edges_list.append(this_directed_edge)

                        # Add this DE to the UDE
                        undirected_edge.info["directed_edge_index"].append(
                            directed_edge_index
                        )
                        return

            # no undirected_edge matches to this directed edge
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp].append(this_undirected_edge)
            
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)

    def adjacency_list(self):
        """Get the adjacency list
        Return:
            graph: the adjacency list
                [[0, 1],
                 [0, 2],
                 ...
                 [5, 2]
                 ...  ]]
                the fist column specifies center/source node,
                the second column specifies neighbor/destination node
            directed2undirected:
                [0, 1, ...]
                a list of length = num_directed_edge that specifies
                the undirected edge index corresponding to the directed edges
                represented in each row in the graph adjacency list.
        """
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]

        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff):
        """Get the line graph adjacency list.

        Args:
            cutoff (float): a float to indicate the maximum edge length to be included
                in constructing the line graph, this is used to decrease computation
                complexity

        Return:
            line_graph:
                [[0, 1, 1, 2, 2],
                [0, 1, 1, 4, 23],
                [1, 4, 23, 5, 66],
                ... ...  ]
                the fist column specifies node(atom) index at this angle,
                the second column specifies 1st undirected edge(left bond) index,
                the third column specifies 1st directed edge(left bond) index,
                the fourth column specifies 2nd undirected edge(right bond) index,
                the fifth column specifies 2nd directed edge(right bond) index,.
            undirected2directed:
                [32, 45, ...]
                a list of length = num_undirected_edge that
                maps the undirected edge index to one of its directed edges indices
        """
        assert len(self.directed_edges_list) == 2 * len(self.undirected_edges_list), (
            f"Error: number of directed edges={len(self.directed_edges_list)} != 2 * "
            f"number of undirected edges={len(self.directed_edges_list)}!"
            f"This indicates directed edges are not complete"
        )
        line_graph = []
        #undirected2directed = []

        # Loop through all the undirected edges(UDE)
        for u_edge in self.undirected_edges_list:
            # Create the mapping from the UDE index to one of its DE index
            #undirected2directed.append(u_edge.info["directed_edge_index"][0])

            # Ignore the UDE if its length is larger than cutoff (3A for default)
            if u_edge.info["distance"] > cutoff:
                continue

            # Assert if the UDE is valid
            # if encountered exception,
            # it means after Atom_Graph creation, the UDE has only 1 DE associated
            # This exception is not encountered from the develop team's experience
            assert len(u_edge.info["directed_edge_index"]) == 2, (
                "Did not find 2 Directed_edges !!!"
                f"undirected edge {u_edge} has:"
                f"edge.info['directed_edge_index'] = "
                f"{u_edge.info['directed_edge_index']}"
                f"len directed_edges_list = {len(self.directed_edges_list)}"
                f"len undirected_edges_list = {len(self.undirected_edges_list)}"
            )

            # This UDE is valid to be considered as a node in Bond_Graph

            # Get the two ends (centers) and the two DE associated with this UDE
            # DE1 should have center=center1 and DE2 should have center=center2
            # We will need to find directed edges with center = center1
            # and create angles with DE1, then do the same for center2 and DE2
            for center, DE in zip(u_edge.nodes, u_edge.info["directed_edge_index"]):
                for directed_edges in self.nodes[center].neighbors.values():
                    for directed_edge in directed_edges:
                        if directed_edge.index == DE:
                            continue
                        if directed_edge.info["distance"] < cutoff:
                            line_graph.append(
                                [
                                    center,
                                    u_edge.index,
                                    DE,
                                    directed_edge.info["undirected_edge_index"],
                                    directed_edge.index,
                                ]
                            )
        return line_graph#, undirected2directed

    def undirected2directed(self):
        """The index map from undirected_edge index to one of its directed_edge
        index.
        """
        return [
            undirected_edge.info["directed_edge_index"][0]
            for undirected_edge in self.undirected_edges_list
        ]

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return f"Graph({num_nodes=}, {num_directed_edges=}, {num_undirected_edges=})"
    