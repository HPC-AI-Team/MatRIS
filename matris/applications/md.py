"""Molecular dynamics, adapted from https://github.com/CederGroupHub/chgnet."""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from ase.io import Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .base import MatRISCalculator


class MolecularDynamics:
    """Run ASE molecular dynamics with a MatRIS calculator."""

    def __init__(
        self,
        atoms: Atoms | Structure | Molecule,
        model: str = "matris_10m_oam",
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        temperature: float = 300,
        starting_temperature: float | None = None,
        timestep: float = 2.0,
        pressure: float = 1.01325e-4,
        taut: float | None = None,
        taup: float | None = None,
        bulk_modulus: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        task: str = "efs",
        device: str | None = None,
        mode: str = "fast",
        activation_checkpoint: bool = False,
        tf32: bool = False,
    ) -> None:
        """
        Args:
            atoms (Atoms | Structure | Molecule): Atomic configuration for MD.
            model (str): Pretrained model name passed to MatRISCalculator.
            ensemble (str): choose from 'nve', 'nvt', 'npt'
                Default = "nvt"
            thermostat (str): Thermostat to use
                choose from "Nose-Hoover", "Berendsen", "Berendsen_inhomogeneous"
                Default = "Berendsen_inhomogeneous"
            temperature (float): temperature for MD simulation, in K
                Default = 300
            starting_temperature (float): starting temperature of MD simulation, in K
                if set as None, the MD starts with the momentum carried by ase.Atoms
                if input is a pymatgen.core.Structure, the MD starts at 0K
                Default = None
            timestep (float): time step in fs
                Default = 2
            pressure (float): pressure in GPa
                Can be 3x3 or 6 np.array if thermostat is "Nose-Hoover"
                Default = 1.01325e-4 GPa = 1 atm
            taut (float): time constant for temperature coupling in fs.
                The temperature will be raised to target temperature in approximate
                10 * taut time.
                Default = 100 * timestep
            taup (float): time constant for pressure coupling in fs
                Default = 1000 * timestep
            bulk_modulus (float): bulk modulus of the material in GPa.
            trajectory (str or Trajectory): Attach trajectory object
                Default = None
            logfile (str): open this file for recording MD outputs
                Default = None
            loginterval (int): write to log file every interval steps
                Default = 1
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): Model device; None selects CUDA when available, else CPU.
            mode (str): 'fast' or 'torch'.
            activation_checkpoint (bool): Activation checkpointing; disabled by default.
            tf32 (bool): Permit TF32 for interaction-block FP32 GEMMs only.
        """
        self.ensemble = ensemble = ensemble.lower()
        self.thermostat = thermostat = thermostat.lower()
        if ensemble not in ("nve", "nvt", "npt"):
            raise ValueError("ensemble must be 'nve', 'nvt', or 'npt'")
        if ensemble != "nve" and thermostat not in (
            "nose-hoover",
            "berendsen",
            "berendsen_inhomogeneous",
            "npt_berendsen",
        ):
            raise ValueError(
                "thermostat must be 'Nose-Hoover', 'Berendsen', or "
                "'Berendsen_inhomogeneous'"
            )
        if ensemble == "npt" and (bulk_modulus is None or bulk_modulus <= 0):
            raise ValueError("NPT requires a positive bulk_modulus in GPa")
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)

        self.atoms = atoms

        self.atoms.calc = MatRISCalculator(
            model=model,
            mode=mode,
            activation_checkpoint=activation_checkpoint,
            tf32=tf32,
            device=device,
            task=task,
        )

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        dynamics_kwargs = {
            "atoms": self.atoms,
            "timestep": timestep * units.fs,
            "trajectory": trajectory,
            "logfile": logfile,
            "loginterval": loginterval,
            "append_trajectory": append_trajectory,
        }
        if ensemble == "nve":
            dynamics_type = VelocityVerlet
        elif thermostat == "nose-hoover":
            self.upper_triangular_cell()
            dynamics_type = NPT
            dynamics_kwargs.update(
                temperature_K=temperature,
                externalstress=pressure * units.GPa,
                ttime=taut * units.fs,
                pfactor=(
                    (bulk_modulus * units.GPa * (taup * units.fs) ** 2)
                    if ensemble == "npt"
                    else None
                ),
            )
        elif ensemble == "nvt":
            dynamics_type = NVTBerendsen
            dynamics_kwargs.update(
                temperature_K=temperature,
                taut=taut * units.fs,
            )
        else:
            dynamics_type = (
                Inhomogeneous_NPTBerendsen
                if thermostat == "berendsen_inhomogeneous"
                else NPTBerendsen
            )
            dynamics_kwargs.update(
                temperature_K=temperature,
                pressure_au=pressure * units.GPa,
                taut=taut * units.fs,
                taup=taup * units.fs,
                compressibility_au=1 / (bulk_modulus * units.GPa),
            )
        self._dynamics_kwargs = dynamics_kwargs
        self.dyn = dynamics_type(**dynamics_kwargs)
        if ensemble == "npt":
            self.bulk_modulus = bulk_modulus

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int) -> None:
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms) -> None:
        """Start fresh integration for new atoms, retaining calculator and outputs.

        Existing trajectory files are appended. Integrator state and step count
        restart; attach any custom ASE observers to the new ``dyn`` instance.
        """
        calculator = self.atoms.calc
        self.dyn.close()
        self.atoms = atoms
        atoms.calc = calculator
        if self.thermostat == "nose-hoover" and self.ensemble != "nve":
            self.upper_triangular_cell()
        self._dynamics_kwargs.update(atoms=atoms, append_trajectory=True)
        if isinstance(self.trajectory, TrajectoryWriter):
            self.trajectory.atoms = atoms
        self.dyn = type(self.dyn)(**self._dynamics_kwargs)

    def upper_triangular_cell(self, verbose: bool = False) -> None:
        """Rotate the cell, positions, and momenta into ASE NPT's upper form."""
        cell = self.atoms.cell
        if not np.array_equal(cell, np.triu(cell)):
            cell, rotation = cell.standard_form("upper")
            momenta = self.atoms.get_momenta() @ rotation.T
            self.atoms.set_cell(cell, scale_atoms=True)
            self.atoms.set_momenta(momenta, apply_constraint=False)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)
