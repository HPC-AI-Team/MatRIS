"""ASE calculator and trajectory observers for MatRIS applications."""

import pickle

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin
from pymatgen.io.ase import AseAtomsAdaptor

from ..model.model import MatRIS

OPTIMIZERS = {
    cls.__name__: cls
    for cls in (BFGS, BFGSLineSearch, FIRE, LBFGS, LBFGSLineSearch, MDMin)
}


class MatRISCalculator(Calculator):
    """ASE calculator returning total energies, forces, and stress in ASE units."""

    implemented_properties = ("energy", "forces", "stress", "magmoms")

    def __init__(
        self,
        model: str = "matris_10m_oam",
        task: str = "efs",
        device: str | None = "cpu",
        mode: str = "fast",
        activation_checkpoint: bool = True,
        tf32: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the calculator.

        Args:
            model: Pretrained model name passed to ``MatRIS.load``.
            task: Prediction task: 'e', 'em', 'ef', 'efs', or 'efsm'.
            device: Model device; None selects CUDA when available, otherwise CPU.
            mode: 'fast' (operators + compile) or 'torch' (eager PyTorch).
            activation_checkpoint: Enable adaptive inference recomputation; disabled by default.
            tf32: Permit TF32 for interaction-block FP32 GEMMs only.
            **kwargs: Passed to the ASE Calculator.
        """
        if "checkpoint" in kwargs:
            raise TypeError("Use activation_checkpoint instead of checkpoint.")
        super().__init__(**kwargs)
        self.task = task
        if mode not in ("fast", "torch"):
            raise ValueError(f"Unknown mode {mode!r}; choose 'fast' or 'torch'.")
        self.mode = mode
        self.activation_checkpoint = activation_checkpoint
        self.tf32 = tf32
        self.model = MatRIS.load(model_name=model, device=device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = next(self.model.parameters()).device
        self.stress_unit = units.GPa
        self.implemented_properties = [
            name
            for key, name in zip("efsm", ("energy", "forces", "stress", "magmoms"))
            if key in task
        ]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate one structure without modifying the caller's atoms."""
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        pbc = atoms.get_pbc()
        if not pbc.all():
            atoms = atoms.copy()
            cell = atoms.cell.copy()
            cutoff = self.model.config["pairwise_cutoff"]
            expand = max(5, self.model.config["num_layers"])
            padding = (np.abs(atoms.positions).max() + 1) * expand * cutoff
            # Complete the non-periodic directions without changing periodic vectors.
            cell[~pbc] = 0
            cell = cell.complete()
            cell[~pbc] *= padding
            atoms.set_cell(cell, scale_atoms=False)

        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(
            structure,
            atomic_numbers=atoms.get_atomic_numbers(),
            frac_coords=atoms.get_scaled_positions(wrap=False),
            lattice_matrix=atoms.cell.array,
            cart_coords=atoms.get_positions(),
            mode=self.mode,
        ).to(self.device)
        prediction = self.model(
            [graph],
            task=self.task,
            is_training=False,
            mode=self.mode,
            activation_checkpoint=self.activation_checkpoint,
            tf32=self.tf32,
        )
        n_atoms = len(atoms) if self.model.is_intensive else 1
        ref_energy = prediction["ref_energy"]
        self.results = {
            "energy": prediction["e"][0].detach().cpu().item() * n_atoms,
            "ref_energy": (
                ref_energy[0].detach().cpu().item()
                if self.model.reference_energy is not None
                else ref_energy
            )
            * n_atoms,
        }
        for key, name in (("f", "forces"), ("s", "stress"), ("m", "magmoms")):
            if key in prediction:
                value = prediction[key][0].detach().cpu().numpy()
                self.results[name] = value * self.stress_unit if key == "s" else value


class TrajectoryObserver:
    """Record relaxation frames. Adapted from https://github.com/CederGroupHub/chgnet."""

    def __init__(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray | None] = []
        self.magmoms: list[np.ndarray | None] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record energy, forces, geometry, and available optional properties."""
        self.energies.append(self.atoms.get_potential_energy())
        self.forces.append(self.atoms.get_forces())
        properties = self.atoms.calc.implemented_properties
        self.stresses.append(
            self.atoms.get_stress() if "stress" in properties else None
        )
        self.magmoms.append(
            self.atoms.get_magnetic_moments() if "magmoms" in properties else None
        )
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.cell.array.copy())

    def __len__(self) -> int:
        """Number of recorded frames."""
        return len(self.energies)

    def save(self, filename: str) -> None:
        """Save the trajectory as a pickle file."""
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
