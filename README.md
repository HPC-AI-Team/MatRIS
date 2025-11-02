# MatRIS

This repository contains the official PyTorch implementation of **MatRIS** (a foundation model for **Mat**erials **R**epresentation and **I**nteraction **S**imulation). 

### Requirements

```
ase >= 3.23.0
numpy >= 2.0.0
pymatgen > 2024.9.10
torch > 2.6.0
```



### Pretrained Models

We offer three pretrained models:

   - **MatRIS-10M-Omat**: A model trained on the OMat24 dataset.
     - Model key: matris_10m_omat
   - **MatRIS-10M-OAM**: A model trained on the OMat24 dataset, and finetuned on sAlex+Mptrj dataset.
     - Model key: matris_10m_oam
   - **MatRIS-10M-MP**: A model trained on the MPTrj dataset.
     - Model key: matris_10m_mp



###  Usage

There are some examples how to use MatRIS, including calculator, geometry optimization and molecular dynamics.

### ASE Calculator

```python
import ase
from ase.build import bulk
import torch
from matris.applications.base import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model='matris_10m_oam', # matris_10m_oam, matris_10m_mp
    task='efsm', # Can be e/ef/efs/efsm 
    device=device # cpu or cuda
)

cu = bulk('Cu', a=5.43, cubic=True)
cu.calc = calc

energy = cu.get_potential_energy()   # total energy(eV)
forces = cu.get_forces()             # forces (eV/A)          
stress = cu.get_stress()             # stress (eV/A^3)  
magmoms = cu.get_magnetic_moments()  # magmom (muB)
```


### Structure Optimization

```python
import ase
from ase.build import bulk
import torch

from matris.applications.relax import StructOptimizer

model_name = "matris_10m_oam"
device = "cuda" if torch.cuda.is_available() else "cpu"

matris_opt = StructOptimizer(
    model = model_name, 
    task = "efsm",
    optimizer = "FIRE", # FIRE, BFGS ...
    device=device
)

atom = bulk('Cu', a=5.43, cubic=True)
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
energy = trajectory.energies[-1] # final energy
force = trajectory.forces[-1]
stress = trajectory.stresses[-1]
magmom = trajectory.magmoms[-1]

final_structure = opt_result['final_structure'] # pymatgen.core.structure.Structure

```


### Molecular Dynamics 

```python
import ase
from ase.build import bulk
import torch
from matris.applications import MolecularDynamics
# Molecular D
#atoms = Structure.from_file(f"xxx.cif")
atom = bulk('Cu', a=5.43, cubic=True)

md = MolecularDynamics(
    atoms=atom, # pymatgen.Structure or ase.Atoms
    model="matris_10m_oam",
    ensemble="nvt", # nvt, nve ...
    temperature=300,  # in K
    timestep=1,  # in femto-seconds
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
    task="efsm",
    device="cuda",
)
md.run(1000)
```

### LICENSE
MatRIS is licensed under the BSD-3-Clause License. Please see the [LICENSE](LICENSE) file for details.
