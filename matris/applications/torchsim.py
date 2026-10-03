"""TorchSim interface: batched predictions selected by target."""

from os import PathLike

import torch
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

from ..graph import RadiusGraph
from ..model import MatRIS


class MatRISTorchSimModel(ModelInterface):
    """Run MatRIS from TorchSim states without an ASE round trip on CUDA.

    Energies are total eV, forces eV/Angstrom and stresses eV/Angstrom^3.
    Target is 'e', 'em', 'ef', 'efs', or 'efsm'; 'm' adds per-atom magmoms in muB.
    The fast operators use float32. Non-periodic molecules receive an internal
    computational cell and zero stress; the input state is never modified.
    Missing non-periodic cell vectors are completed internally.
    Isolated atoms retain the model's atomic energy prediction and have zero force.
    Outputs are detached for simulation; differentiable trajectories are not supported.
    """

    def __init__(
        self,
        model: str | PathLike | MatRIS = "matris_10m_oam",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        mode: str = "fast",
        target: str = "efs",
        activation_checkpoint: bool = True,
        tf32: bool = False,
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise ValueError("MatRISTorchSimModel currently supports torch.float32.")
        if mode not in ("fast", "torch"):
            raise ValueError("mode must be 'fast' or 'torch'")
        if target not in ("e", "em", "ef", "efs", "efsm"):
            raise ValueError("target must be 'e', 'em', 'ef', 'efs', or 'efsm'")
        self.model = model if isinstance(model, MatRIS) else MatRIS.load(model, device=device)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        parameter = next(self.model.parameters())
        if parameter.dtype != dtype:
            raise ValueError("MatRIS weights must use torch.float32.")
        self._device = parameter.device
        self._dtype = dtype
        self.target = target
        self.mode = mode
        self.activation_checkpoint = activation_checkpoint
        self.tf32 = tf32
        self.cutoff = self.model.config["pairwise_cutoff"]

    @ModelInterface.compute_forces.getter
    def compute_forces(self) -> bool:
        return "f" in self.target

    @ModelInterface.compute_stress.getter
    def compute_stress(self) -> bool:
        return "s" in self.target

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Build each system's graph on-device, then evaluate the batch once."""
        if state.device != self.device or state.dtype != self.dtype:
            raise ValueError("SimState device and dtype must match the MatRIS adapter.")
        counts = state.n_atoms_per_system.tolist()
        cells = state.row_vector_cell.detach().to(torch.float64)
        periodic = state.pbc.to(device=self.device).expand(len(counts), 3)
        periodic_dims = periodic.sum(dim=1).tolist()
        line_cutoff = self.model.config["three_body_cutoff"]
        graphs = []
        start = 0
        for index, count in enumerate(counts):
            positions = state.positions[start : start + count].detach().to(torch.float64)
            cell = cells[index]
            if not periodic_dims[index]:
                positions = positions - positions.mean(dim=0)
                length = 2 * (positions.abs().max() + self.cutoff)
                cell = torch.eye(3, device=self.device, dtype=positions.dtype) * length
            elif periodic_dims[index] < 3:
                missing = (cell.norm(dim=1) == 0) & ~periodic[index]
                if missing.any():
                    # Complete only absent non-periodic vectors, perpendicular
                    # to the existing periodic subspace, without moving atoms.
                    basis = torch.linalg.svd(cell).Vh
                    cell = cell.clone()
                    length = 2 * (positions.abs().max() + self.cutoff)
                    cell[missing] = basis[-int(missing.sum()):] * length
            if self.device.type == "cuda" and self.mode == "fast":
                from ..model.op.graph import build_graph_tensors_gpu, gpu_neighbor_list

                edges, ptr, images, distances = gpu_neighbor_list(
                    positions, cell, self.cutoff, device=self.device, pbc=periodic[index]
                )
                tensors = build_graph_tensors_gpu(
                    edges, ptr, images, distances, line_cutoff=line_cutoff
                )
            else:
                from ase.neighborlist import primitive_neighbor_list

                from ..graph.converter import build_graph_tensors

                centers, neighbors, images, distances = primitive_neighbor_list(
                    "ijSd", periodic[index].cpu().numpy(), cell.cpu().numpy(),
                    positions.cpu().numpy(), self.cutoff + 1e-8,
                )
                tensors = build_graph_tensors(
                    count, centers, neighbors, images, distances, line_cutoff=line_cutoff
                )
            atom_graph, d2u, u2d, line_graph = tensors
            graphs.append(RadiusGraph(
                graph_id=index, mp_id=None, composition=None,
                atomic_number=state.atomic_numbers[start : start + count].to(torch.int32, copy=True),
                atom_frac_coord=torch.linalg.solve(cell.mT, positions.mT).mT.to(self.dtype),
                lattice=cell.to(self.dtype),
                neighbor_image=torch.as_tensor(images, device=self.device, dtype=self.dtype),
                atom_graph=atom_graph.to(self.device),
                directed2undirected=d2u.to(self.device),
                undirected2directed=u2d.to(self.device),
                line_graph=line_graph.to(self.device),
                atom_graph_cutoff=self.cutoff, line_graph_cutoff=line_cutoff,
            ))
            start += count
        prediction = self.model(
            graphs, task=self.target, mode=self.mode,
            activation_checkpoint=self.activation_checkpoint, tf32=self.tf32,
        )
        energy = prediction["e"]
        if self.model.is_intensive:
            energy = energy * prediction["atoms_per_graph"]
        results = {"energy": energy.detach()}
        if self.compute_forces:
            results["forces"] = torch.cat(prediction["f"]).detach()
        if self.compute_stress:
            stress = torch.stack(prediction["s"]).detach() / 160.21766208
            results["stress"] = torch.where(periodic.any(dim=1)[:, None, None], stress, 0)
        if "m" in self.target:
            results["magmoms"] = torch.cat(prediction["m"]).detach()
        return results
