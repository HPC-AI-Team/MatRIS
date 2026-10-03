import torch
from matris.graph import RadiusGraph

from collections.abc import Sequence


def process_graphs(
    graphs: Sequence[RadiusGraph], compute_stress: bool = True, *, fast: bool = False
):
    """
    Process a sequence of graphs and batch them into a unified graph.

    Args:
        graphs: Sequence of RadiusGraph objects containing multiple crystal graph data.
        compute_stress: Whether to compute stress.

    Returns:
        batched_graph:
            atomic_numbers: Atomic numbers of all atoms [total_atoms]
            edge_lengths: Lengths of all edges [total_direct_edges]
            unit_edge_vectors: Unit edge vectors [total_direct_edges, 3]
            atom_segment: Graph(Batch) indices for each atom [total_atoms]
            atoms_per_graph: List of atom counts per graph
            directed2undirected: Mapping from directed to undirected edges [total_direct_edges]
            undirected2directed: Mapping from undirected to directed edges [total_undirected_edges]
            volumes: Volume of each graph [num_graphs, 1, 1]
            lattice: Lattice vectors [num_graphs*3, 3]

            batch_cart_coords: Cartesian coordinates of all atoms [total_atoms, 3]
            batch_strains: Strain tensors [num_graphs, 3, 3] (only when compute_stress=True)

            atom_graph_dict: Dict containing atom graph information
                atom_graph: Atom connectivity [total_direct_edges, 2]
                target_index/source_index: Target/source node indices for edges [total_direct_edges]
                num_segment: Number of segments
                source_bincount: The frequency of each source node in atom graph
                target_bincount: The frequency of each target node in atom graph
            line_graph_dict: Dict containing line graph information
                line_graph: Compressed line graph [num_angles, 3] # remove directed edge index
                atom_list: Central atom indices for angles [num_angles]
                target_index/source_index: Target/source edge indices for line graph [num_angles]
                target_DE_index/source_DE_index: Directed edge indices [num_angles]
                source_bincount: The frequency of each source node in line graph
                target_bincount: The frequency of each target node in line graph
    """

    batched_graph = {}
    num_graphs = len(graphs)
    device = graphs[0].lattice.device
    dtype = graphs[0].lattice.dtype

    atomic_numbers = []
    batched_atom_graph, batched_line_graph = [], []
    atom_segment, atoms_per_graph = [], []

    batch_lattice, batch_image, batch_cart_coords = [], [], []
    undirected2directed, directed2undirected = [], []

    atom_graph_len, num_undirected2directed, num_directed2undirected = 0, 0, 0
    for graph_idx, graph in enumerate(graphs):
        direct_edge_num = len(graph.atom_graph)
        undirected2directed.append(graph.undirected2directed)
        num_undirected2directed += int(direct_edge_num / 2)  # undirect edge num

        directed2undirected.append(graph.directed2undirected)
        num_directed2undirected += direct_edge_num  # direct edge num

        batched_atom_graph.append(graph.atom_graph)
        atom_graph_len += direct_edge_num  # direct edge num

    atom_offset_idx_list = torch.empty(
        (atom_graph_len, 2), dtype=torch.int32, device=device
    )
    edge_offses_idx_list = torch.empty(
        num_undirected2directed, dtype=torch.int32, device=device
    )
    n_undirected_list = torch.empty(
        num_directed2undirected, dtype=torch.int32, device=device
    )

    atom_idx_offset, edge_idx_offset, undirected_list_offset, directed_list_offset = (
        0,
        0,
        0,
        0,
    )
    start_atom_offset, start_edge_offset, start_n_undirected = 0, 0, 0

    for graph_idx, graph in enumerate(graphs):
        atomic_numbers.append(graph.atomic_number)  # atom type
        n_atom = graph.atomic_number.shape[0]
        atoms_per_graph.append(n_atom)
        if graph.atom_frac_coord.requires_grad:
            # graph.atom_frac_coord.requires_grad = False
            graph.atom_frac_coord = graph.atom_frac_coord.detach()
        if graph.lattice.requires_grad:
            # graph.lattice.requires_grad = False
            graph.lattice = graph.lattice.detach()

        atom_cart_coords = graph.atom_frac_coord @ graph.lattice  # [n_atom, 3]

        batch_cart_coords.append(atom_cart_coords)
        batch_lattice.append(graph.lattice)
        batch_image.append(graph.neighbor_image)

        edge_offses_idx_list[
            start_edge_offset : start_edge_offset + len(graph.undirected2directed)
        ] = edge_idx_offset
        atom_offset_idx_list[
            start_atom_offset : start_atom_offset + len(graph.atom_graph), :
        ] = atom_idx_offset
        n_undirected_list[
            start_n_undirected : start_n_undirected + len(graph.directed2undirected)
        ] = undirected_list_offset

        if len(graph.line_graph) != 0:
            this_line_graph = graph.line_graph.new_zeros([graph.line_graph.shape[0], 5])
            this_line_graph[:, 0] = (
                graph.line_graph[:, 0] + atom_idx_offset
            )  # atom type at this angle
            this_line_graph[:, 1] = (
                graph.line_graph[:, 1] + undirected_list_offset
            )  # left undirect edge index
            this_line_graph[:, 3] = (
                graph.line_graph[:, 3] + undirected_list_offset
            )  # right undirect edge index

            this_line_graph[:, 2] = (
                graph.line_graph[:, 2] + directed_list_offset
            )  # left direct edge index
            this_line_graph[:, 4] = (
                graph.line_graph[:, 4] + directed_list_offset
            )  # right direct edge index

            batched_line_graph.append(this_line_graph)

        atom_segment.append(
            torch.full((n_atom,), graph_idx, dtype=torch.int64, device=device)
        )
        atom_idx_offset += n_atom
        edge_idx_offset += len(graph.atom_graph)
        directed_list_offset += len(graph.directed2undirected)
        undirected_list_offset += len(graph.undirected2directed)

        start_edge_offset += len(graph.undirected2directed)
        start_atom_offset += len(graph.atom_graph)
        start_n_undirected += len(graph.directed2undirected)

    # ======= packing features =======
    atomic_numbers = torch.cat(atomic_numbers, dim=0)
    batched_atom_graph = torch.cat(batched_atom_graph, dim=0)  # [edge, 2]
    batched_atom_graph += atom_offset_idx_list  # add offset
    batched_atom_graph = batched_atom_graph.to(torch.int64)
    if fast and device.type == "cuda":
        batched_atom_graph = batched_atom_graph.t().contiguous().t()

    if batched_line_graph != []:
        batched_line_graph = torch.cat(batched_line_graph, dim=0)  # line_graph
        batched_line_graph = batched_line_graph.to(torch.int64)
    else:  # when line graph is empty
        batched_line_graph = torch.empty((0, 5), dtype=torch.int64, device=device)

    atom_segment = torch.cat(atom_segment, dim=0).type(torch.int64)

    directed2undirected = torch.cat(directed2undirected, dim=0)
    directed2undirected += n_undirected_list  # add offset
    undirected2directed = torch.cat(undirected2directed, dim=0)
    undirected2directed += edge_offses_idx_list  # add offset

    batch_image = torch.block_diag(
        *batch_image
    )  # [direct_edge_num, num_graphs * 3]. Note: Only the diagonal blocks are valid.
    batch_lattice = torch.cat(batch_lattice, dim=0)  # [num_graphs * 3, 3]
    batch_cart_coords = torch.cat(batch_cart_coords, dim=0)

    # Reshape [n_graphs*3, 3] -> [n_graphs, 3, 3] for volumes calculation
    batch_lattices = batch_lattice.reshape(num_graphs, 3, 3)
    cross_product = torch.linalg.cross(batch_lattices[:, 1], batch_lattices[:, 2])
    volumes = torch.einsum("ni,ni->n", batch_lattices[:, 0], cross_product).abs()
    volumes = volumes.unsqueeze(1).unsqueeze(2)

    batch_line_graph_compress = torch.empty((0, 5), dtype=torch.int64, device=device)
    if len(batched_line_graph) != 0:
        if fast and device.type == "cuda":
            # One column-major allocation backs compressed and directed indices.
            packed = torch.stack(
                tuple(batched_line_graph[:, i] for i in (0, 1, 3, 2, 4))
            )
            line_graph_atom_index = packed[0]
            line_graph_UDE_target_index = packed[1]
            line_graph_UDE_source_index = packed[2]
            line_graph_DE_target_index = packed[3]
            line_graph_DE_source_index = packed[4]
            batch_line_graph_compress = packed[:3].t()
        else:
            # line graph index
            line_graph_atom_index = batched_line_graph[:, 0]
            line_graph_UDE_target_index = batched_line_graph[:, 1]
            line_graph_DE_target_index = batched_line_graph[:, 2]
            line_graph_UDE_source_index = batched_line_graph[:, 3]
            line_graph_DE_source_index = batched_line_graph[:, 4]

            batch_line_graph_compress = batched_line_graph.new_zeros(
                [batched_line_graph.shape[0], 3]
            )
            batch_line_graph_compress[:, 0] = line_graph_atom_index
            batch_line_graph_compress[:, 1] = line_graph_UDE_target_index
            batch_line_graph_compress[:, 2] = line_graph_UDE_source_index

    # ========================================
    strains = None
    batch_cart_coords.requires_grad_(True)
    if compute_stress:
        strains = torch.zeros((num_graphs, 3, 3), dtype=dtype, device=device,)

        strains.requires_grad_(True)
        symmetric_strains = 0.5 * (strains + strains.transpose(-1, -2))

        batch_cart_coords = batch_cart_coords + torch.einsum(
            "be,bec->bc", batch_cart_coords, symmetric_strains[atom_segment]
        )
        batch_lattice = batch_lattice.view(-1, 3, 3)
        batch_lattice = batch_lattice + torch.matmul(batch_lattice, symmetric_strains)
        batch_lattice = batch_lattice.view(-1, 3)
    # ========================================

    # node index
    target_index, source_index = batched_atom_graph[:, 0], batched_atom_graph[:, 1]
    # ===== compute pairwise distance and edge vectors =====
    center_pos = batch_cart_coords[target_index]
    neighbor_pos = batch_cart_coords[source_index]
    neighbor_pos = neighbor_pos + batch_image @ batch_lattice
    edge_vectors = center_pos - neighbor_pos
    edge_lengths = torch.norm(edge_vectors, dim=1)  # pairwise distance
    unit_edge_vectors = edge_vectors / edge_lengths[:, None]  # edge vectors

    batched_graph["atomic_numbers"] = atomic_numbers  # [atoms]
    batched_graph["edge_lengths"] = edge_lengths  # [direct_edge_num]
    batched_graph["unit_edge_vectors"] = unit_edge_vectors  # [direct_edge_num, 3]
    batched_graph["atom_segment"] = atom_segment.to(device=device)  # [atoms]
    batched_graph["atoms_per_graph"] = atoms_per_graph  # List, size:[atoms]
    batched_graph["directed2undirected"] = directed2undirected.to(
        torch.int64
    )  # [direct_edge_num]
    batched_graph["undirected2directed"] = undirected2directed.to(
        torch.int64
    )  # [undirect_edge_num]
    batched_graph["volumes"] = volumes  # [num_graphs, 1, 1]
    batched_graph["lattice"] = batch_lattice  # [num_graphs*3, 3]
    batched_graph["batch_cart_coords"] = batch_cart_coords  # [atoms, 3]
    batched_graph["batch_strains"] = strains  # [num_graphs, 3, 3]

    atom_graph_dict, line_graph_dict = {}, {}
    atom_graph_dict["atom_graph"] = batched_atom_graph  # [direct_edge_num, 2]
    atom_graph_dict["target_index"] = target_index  # [direct_edge_num]
    atom_graph_dict["source_index"] = source_index  # [direct_edge_num]
    atom_graph_dict["num_segment"] = len(atomic_numbers)

    line_graph_dict["line_graph"] = batch_line_graph_compress
    if len(line_graph_dict["line_graph"]) != 0:
        line_graph_dict["atom_list"] = line_graph_atom_index  # [angles, 3]
        line_graph_dict["target_index"] = line_graph_UDE_target_index  # [angles]
        line_graph_dict["source_index"] = line_graph_UDE_source_index  # [angles]
        line_graph_dict["target_DE_index"] = line_graph_DE_target_index  # [angles]
        line_graph_dict["source_DE_index"] = line_graph_DE_source_index  # [angles]
    else:
        empty_index = torch.empty(0, dtype=torch.int64, device=device)
        line_graph_dict.update({name: empty_index for name in (
            "atom_list", "target_index", "source_index", "target_DE_index", "source_DE_index"
        )})

    for graph_dict, segments in (
        (atom_graph_dict, len(atomic_numbers)),
        (line_graph_dict, len(undirected2directed)),
    ):
        if "source_index" not in graph_dict:
            continue
        if fast and device.type == "cuda":
            from .op.attention import segment_layout

            graph_dict["source_index"] = graph_dict["source_index"].contiguous()
            graph_dict["target_index"] = graph_dict["target_index"].contiguous()
            layouts = (
                segment_layout(graph_dict["source_index"], segments),
                segment_layout(graph_dict["target_index"], segments),
            )
            graph_dict["softmax_layout"] = layouts
            for name, (ptr, _) in zip(("source", "target"), layouts):
                graph_dict[f"{name}_bincount"] = (
                    (ptr[1:] - ptr[:-1]).clamp_min(1).to(torch.int64)
                )
        else:
            for name in ("source", "target"):
                counts = torch.bincount(graph_dict[f"{name}_index"])
                graph_dict[f"{name}_bincount"] = counts.clamp_min(1)

    batched_graph["atom_graph_dict"] = atom_graph_dict
    batched_graph["line_graph_dict"] = line_graph_dict

    return batched_graph
