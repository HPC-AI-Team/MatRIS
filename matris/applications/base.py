from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
import numpy as np

from ..model.model import MatRIS

from pymatgen.io.ase import AseAtomsAdaptor
from ..graph import RadiusGraph

from ase.optimize import (
    BFGS, BFGSLineSearch, 
    FIRE, LBFGS, 
    LBFGSLineSearch, MDMin
)

names = [
    "BFGS", "BFGSLineSearch", 
    "FIRE", "LBFGS", 
    "LBFGSLineSearch", "MDMin"
]

OPTIMIZERS = {name: globals()[name] for name in names}

class MatRISCalculator(Calculator):
    """MatRIS Calculator for ASE applications."""
    
    implemented_properties = ("energy", "forces", "stress", "magmoms")  # type: ignore
    
    def __init__(
        self,
        model: str = "matris_10m_oam",
        task: str = "efs",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """
        Args:
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): The device to be used for predictions,
            stress_unit (float): the conversion factor to convert GPa(MatRIS default) to eV/A^3.
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)
        self.task=task
        self.device = device
        self.model = MatRIS.load(model_name=model, device=self.device)
        
        self.stress_unit = units.GPa
        key = ["atoms_per_graph", "ref_energy"]
        for t in task:
            key.append(t)
        self.key = set(key)
     
    def calculate(
        self,
        atoms: Atoms,
        properties: list,
        system_changes: list,
    ) -> None:
        
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        pbc = atoms.get_pbc()
        if (not pbc[0]) or (not pbc[1]) or (not pbc[2]):
            pos = atoms.get_positions()
            cell = np.array(atoms.get_cell())
            
            pbc_x = pbc[0]
            pbc_y = pbc[1]
            pbc_z = pbc[2]
            identity = np.identity(3, dtype=float)
            max_positions = np.max(np.absolute(pos)) + 1
            
            cutoff = self.model.config["pairwise_cutoff"]
            expand = max(5, self.model.config["num_layers"])

            # Extend cell in non-periodic directions
            if not pbc_x:
                cell[0, :] = max_positions * expand * cutoff * identity[0, :]
            if not pbc_y:
                cell[1, :] = max_positions * expand * cutoff * identity[1, :]
            if not pbc_z:
                cell[2, :] = max_positions * expand * cutoff * identity[2, :]
            
            # update
            atoms.set_cell(cell, scale_atoms=False)
        
        
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure).to(self.device) # convert to List
        
        graphs = [graph] if isinstance(graph, RadiusGraph) else graph
        
        model_prediction = self.model(
            graphs,
            task = self.task,
            is_training = False,
        )
        model_predictions = {}
        
        for key in self.key & set(model_prediction.keys()):
            for idx, tensor in enumerate(model_prediction[key]):
                model_predictions[key] = tensor.cpu().detach().numpy()
        
        # Convert Result
        n_atoms = 1 if not self.model.is_intensive else structure.composition.num_atoms
        
        self.results.update(
            ref_energy=model_predictions["ref_energy"] * n_atoms,
            energy=model_predictions["e"] * n_atoms,  # Total Energy
            forces=model_predictions.get("f", None),
            # Stress: GPa -> eV/A^3
            stress=model_predictions.get("s", None) * self.stress_unit if model_predictions.get("s", None) is not None else None,
            magmoms=model_predictions.get("m", None),
        )


class TrajectoryObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
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


class CrystalFeasObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
                      