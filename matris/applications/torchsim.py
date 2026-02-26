"""torch-sim ModelInterface wrapper for MatRIS."""

from __future__ import annotations

import torch
from torch import Tensor

from ..model.model import MatRIS
from ..graph import RadiusGraph

try:
    from torch_sim import units as _ts_units

    _GPa = _ts_units.MetalUnits.pressure * 1000
    from torch_sim.state import SimState
    from torch_sim.models.interface import ModelInterface
except ImportError as e:
    raise ImportError(
        "torch-sim is not installed. Please install torch-sim to use MatRISModel."
    ) from e


class MatRISModel(ModelInterface):
    """torch-sim compatible wrapper for MatRIS.

    Converts torch-sim SimState inputs to RadiusGraph format, runs the
    MatRIS model, and returns outputs in torch-sim format.

    Output units:
        energy : eV  (total, per system)
        forces : eV/Å
        stress : eV/Å³

    Args:
        model: A :class:`~matris.model.MatRIS` instance, or the name of a
            pretrained checkpoint (e.g. ``"matris_10m_oam"``).
        compute_stress: Whether to compute the stress tensor.
        compute_magmom: Whether to compute per-atom magnetic moments.
        device: Device to run inference on.  Defaults to the device of the
            loaded model parameters.
        dtype: Floating-point dtype exposed to torch-sim.  Defaults to
            ``torch.float32``.

    Example::

        from matris.applications.torchsim import MatRISModel

        model = MatRISModel("matris_10m_oam", compute_stress=True, device="cuda")
        results = model(sim_state)   # returns {"energy", "forces", "stress"}
    """

    def __init__(
        self,
        model: MatRIS | str = "matris_10m_oam",
        compute_stress: bool = True,
        compute_magmom: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            dev = str(device) if device is not None else None
            self.model = MatRIS.load(model_name=model, device=dev)
        else:
            self.model = model

        self._compute_stress = compute_stress
        self._compute_magmom = compute_magmom
        self._dtype = dtype

        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # ModelInterface properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def compute_forces(self) -> bool:
        return True

    @property
    def compute_magmom(self) -> bool:
        return self._compute_magmom

    @property
    def memory_scales_with(self) -> str:
        # MatRIS uses a radial cutoff, so memory scales with n_atoms × density
        return "n_atoms_x_density"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state_to_graphs(
        self,
        state: SimState,
    ) -> list[RadiusGraph]:
        """Convert a SimState (or StateDict) to a list of RadiusGraph objects.

        The cell matrix is treated as row-major (each row is a lattice vector),
        matching the ASE / pymatgen convention.
        """

        structures = state.to_structures()
        return [
            self.model.graph_converter(structure).to(str(self._device))
            for structure in structures
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        state: SimState,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Compute energy, forces, and optionally stress from a SimState.

        Args:
            state: A torch-sim ``SimState`` or a ``StateDict`` with keys
                ``positions``, ``cell``, ``atomic_numbers``, and
                ``system_idx``.

        Returns:
            Dictionary with:
                ``"energy"``  : Tensor of shape ``[n_systems]`` — total energy in eV.
                ``"forces"``  : Tensor of shape ``[n_atoms, 3]`` — forces in eV/Å.
                ``"stress"``  : Tensor of shape ``[n_systems, 3, 3]`` — stress in
                    eV/Å³ (only present when ``compute_stress=True``).
                ``"magmoms"`` : Tensor of shape ``[n_atoms]`` — per-atom magnetic
                    moments in μB (only present when ``compute_magmom=True``).
        """
        graphs = self._state_to_graphs(state)

        task = "ef"
        if self._compute_stress:
            task += "s"
        if self._compute_magmom:
            task += "m"
        result = self.model(graphs, task=task, is_training=False)

        # MatRIS returns per-atom energy for intensive models; convert to total
        energy: Tensor = result["e"]  # [n_systems]
        if self.model.is_intensive:
            atoms_per_graph = result["atoms_per_graph"].to(dtype=energy.dtype)
            energy = energy * atoms_per_graph

        output: dict[str, Tensor] = {
            "energy": energy,  # [n_systems]
            "forces": torch.concat(result["f"], dim=0),  # [n_atoms, 3]
        }

        if self._compute_stress:
            # MatRIS outputs stress in GPa; convert to eV/Å³
            output["stress"] = result["s"] * _GPa

        if self._compute_magmom:
            output["magmoms"] = torch.concat(result["m"], dim=0)  # [n_atoms]

        return output
