import ase
from ase.build import bulk
import torch
from pymatgen.core.structure import Structure
from matris.applications.relax import StructOptimizer

model_name = "matris_10m_oam"
device = "cuda" if torch.cuda.is_available() else "cpu"

matris_opt = StructOptimizer(
    model=model_name, 
    task = "efsm",
    optimizer = "FIRE", # FIRE, BFGS ...
    device=device
)

atom = Structure.from_file(f"example/cif_file/demo.cif")

max_steps = 500
fmax = 0.05
opt_result = matris_opt.relax(
        atoms=atom, # pymatgen.Structure or ase.Atoms
        verbose=True,
        steps=max_steps,
        fmax=fmax,
        relax_cell=max_steps > 0,
        ase_filter="FrechetCellFilter",
    )

trajectory = opt_result['trajectory']
energy = trajectory.energies[-1]
force = trajectory.forces[-1]
stress = trajectory.stresses[-1]
magmom = trajectory.magmoms[-1]

final_structure = opt_result['final_structure']