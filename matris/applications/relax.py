"""Structure relaxation, adapted from https://github.com/CederGroupHub/chgnet."""

import contextlib
import io
import sys

import ase.filters as filter_classes
from ase import Atoms
from ase.filters import Filter
from ase.optimize.optimize import Optimizer
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .base import OPTIMIZERS, MatRISCalculator, TrajectoryObserver


class StructOptimizer:
    """Relax atomic positions and optionally the cell with an ASE optimizer."""

    def __init__(
        self,
        model: str = "matris_10m_oam",
        task: str = "efs",
        optimizer: str = "FIRE",
        device: str | None = "cpu",
        mode: str = "fast",
        activation_checkpoint: bool = False,
        tf32: bool = False,
    ) -> None:
        """Initialize a named ASE optimizer and a MatRIS calculator.

        ``model`` is a pretrained model name. ``task``, ``device``, ``mode``,
        ``activation_checkpoint`` and ``tf32`` are passed to ``MatRISCalculator``.
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Optimizer {optimizer} not found. Select from {list(OPTIMIZERS)}"
            )
        self.optimizer: type[Optimizer] = OPTIMIZERS[optimizer]

        self.calculator = MatRISCalculator(
            model=model,
            mode=mode,
            activation_checkpoint=activation_checkpoint,
            tf32=tf32,
            task=task,
            device=device,
        )

    def relax(
        self,
        atoms: Structure | Molecule | Atoms,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        ase_filter: str | type[Filter] = "FrechetCellFilter",
        save_path: str | None = None,
        loginterval: int = 1,
        verbose: bool = True,
        assign_magmoms: bool = True,
        **kwargs,
    ) -> dict[str, Structure | Molecule | TrajectoryObserver]:
        """
        Args:
            atoms (Structure | Molecule | Atoms): The structure or molecule to relax.
            fmax (float): The maximum force tolerance for relaxation.
            steps (int): The maximum number of steps for relaxation.
            relax_cell (bool): Whether to relax the cell as well. Set to False
                for molecules without a cell.
            ase_filter (str | type[Filter]): ASE filter for cell relaxation.
            save_path (str): The path to save the trajectory.
            loginterval (int): Positive step interval for recording frames.
            verbose (bool): Whether to print the output of the ASE optimizer.
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure.
            **kwargs: Additional parameters for the optimizer.

        Returns:
            A dictionary with ``final_structure`` and ``trajectory``. The final
            structure is a Molecule for fully nonperiodic inputs and a Structure
            if any direction is periodic.
        """

        if loginterval <= 0:
            raise ValueError("loginterval must be positive")
        if relax_cell and isinstance(ase_filter, str):
            filter_type = getattr(filter_classes, ase_filter, None)
            if not isinstance(filter_type, type) or not issubclass(filter_type, Filter):
                raise ValueError(f"Unknown ASE cell filter: {ase_filter}")
            ase_filter = filter_type

        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        atoms.calc = self.calculator

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            optimizer = self.optimizer(
                ase_filter(atoms) if relax_cell else atoms, **kwargs
            )
            optimizer.attach(obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            if optimizer.nsteps % loginterval:
                obs()

        if save_path is not None:
            obs.save(save_path)

        if atoms.pbc.any():
            struct = AseAtomsAdaptor.get_structure(atoms)
        else:
            struct = AseAtomsAdaptor.get_molecule(atoms)

        if assign_magmoms and "magmoms" in self.calculator.implemented_properties:
            struct.add_site_property("magmom", atoms.get_magnetic_moments().tolist())

        return {"final_structure": struct, "trajectory": obs}
