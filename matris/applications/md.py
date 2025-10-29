""" Modified from CHGNet: https://github.com/CederGroupHub/chgnet """
from __future__ import annotations

import inspect
import sys
import io
import contextlib
from ase import Atoms, units
from ase.io import Trajectory
from ase.optimize.optimize import Optimizer
import ase.filters as filter_classes
from ase.filters import Filter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure, Molecule

from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from ..model.model import MatRIS

from .base import (
    OPTIMIZERS,
    MatRISCalculator, 
    TrajectoryObserver,
    CrystalFeasObserver
)

class MolecularDynamics:
    """This class is used for Molecular Dynamics."""

    def __init__(
        self,
        atoms: Atoms | Structure,
        model: str = "matris_10m_oam",
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        temperature: int = 300,
        starting_temperature: int | None = None,
        timestep: float = 2.0,
        pressure: float = 1.01325e-4,
        taut: float | None = None,
        taup: float | None = None,
        bulk_modulus: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        crystal_feas_logfile: str | None = None,
        append_trajectory: bool = False,
        task: str = "efs",
        device: str | None = None,
    ) -> None:
        """
        Args:
            atoms (Atoms): atoms to run the MD
            model (MatRIS): instance of a MatRIS model or MatRISCalculator.
                If set to None, the pretrained MatRIS is loaded.
                Default = None
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
            crystal_feas_logfile (str): open this file for recording crystal features
                during MD. Default = None
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): the device for the MD run
                Default = None
        """
        self.ensemble = ensemble
        self.thermostat = thermostat
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)

        self.atoms = atoms
        
        self.atoms.calc = MatRISCalculator(
            model=model,
            device=device,
            task=task,
        )

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        if ensemble.lower() == "nve":
            """
            VelocityVerlet (constant N, V, E) molecular dynamics.

            Note: it's recommended to use smaller timestep for NVE compared to other
            ensembles, since the VelocityVerlet algorithm assumes a strict conservative
            force field.
            """
            self.dyn = VelocityVerlet(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")

        elif ensemble.lower() == "nvt":
            """
            Constant volume/temperature molecular dynamics.
            """
            if thermostat.lower() == "nose-hoover":
                """
                Nose-hoover (constant N, V, T) molecular dynamics.
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure
                    * units.GPa,  # ase NPT does not like externalstress=None
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")
            elif thermostat.lower().startswith("berendsen"):
                """
                Berendsen (constant N, V, T) molecular dynamics.
                """
                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        elif ensemble.lower() == "npt":
            """
            Constant pressure/temperature molecular dynamics.
            """
            # Bulk modulus is needed for pressure damping time
            if bulk_modulus is not None:
                bulk_modulus_au = bulk_modulus * units.GPa  # GPa to eV/A^3
                compressibility_au = 1 / bulk_modulus_au
            else:
                try:
                    # Fit bulk modulus by equation of state
                    eos = EquationOfState(model=self.atoms.calc)
                    eos.fit(atoms=atoms, steps=500, fmax=0.1, verbose=False)
                    bulk_modulus = eos.get_bulk_modulus(unit="GPa")
                    bulk_modulus_au = eos.get_bulk_modulus(unit="eV/A^3")
                    compressibility_au = eos.get_compressibility(unit="A^3/eV")
                    print(
                        f"Completed bulk modulus calculation: "
                        f"k = {bulk_modulus:.3}GPa, {bulk_modulus_au:.3}eV/A^3"
                    )
                except Exception:
                    bulk_modulus_au = 2 * units.GPa
                    compressibility_au = 1 / bulk_modulus_au
                    print(
                        "Warning!!! Equation of State fitting failed, setting bulk "
                        "modulus to 2 GPa. NPT simulation can proceed with incorrect "
                        "pressure relaxation time."
                        "User input for bulk modulus is recommended."
                    )
            self.bulk_modulus = bulk_modulus

            if thermostat.lower() == "nose-hoover":
                """
                Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an
                NPT (or N,stress,T) ensemble.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/npt.py
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                ptime = taup * units.fs
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=bulk_modulus * units.GPa * ptime * ptime,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Nose-Hoover MD created")

            elif thermostat.lower() == "berendsen_inhomogeneous":
                """
                Inhomogeneous_NPTBerendsen thermo/barostat
                This is a more flexible scheme that fixes three angles of the unit
                cell but allows three lattice parameter to change independently.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )
                print("NPT-Berendsen-inhomogeneous-MD created")

            elif thermostat.lower() == "npt_berendsen":
                """
                This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                This is a less flexible scheme that fixes the shape of the
                cell - three angles are fixed and the ratios between the three
                lattice constants.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )
        
        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep
        self.crystal_feas_logfile = crystal_feas_logfile

    def run(self, steps: int) -> None:
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        if self.crystal_feas_logfile:
            obs = CrystalFeasObserver(self.atoms)
            self.dyn.attach(obs, interval=self.loginterval)

        self.dyn.run(steps)

        if self.crystal_feas_logfile:
            obs.save(self.crystal_feas_logfile)

    def set_atoms(self, atoms: Atoms) -> None:
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator

    def upper_triangular_cell(self, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)
