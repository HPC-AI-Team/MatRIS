# MatRIS: Toward Reliable and Efficient Pretrained Machine Learning Interatomic Potentials

**[Paper](https://arxiv.org/abs/2603.02002)** | **[Poster](https://iclr.cc/media/PosterPDFs/ICLR%202026/10011410.png?t=1776772805.5085049)**

Official PyTorch implementation of **MatRIS** (**Mat**erials **R**epresentation and **I**nteraction **S**imulation), a pretrained interatomic potential for materials simulation.

## Update

**Oct 2026**

- Optimized kernels and `torch.compile` support (~5× faster inference).
- TorchSim integration.
- New models: OMat24 and the MatPES PBE/r2SCAN family.

## Installation

- Python ≥ 3.12, PyTorch ≥ 2.11
- NumPy, ASE, pymatgen, nvalchemi-toolkit-ops, Ninja

## Pretrained Models

We offer seven pretrained models:

| Model key | Checkpoint | Dataset | Target |
| --- | --- | --- | --- |
| `matris_10m_omat` | [Download](https://api.figshare.com/v2/file/download/69575544) | OMat24 | efs |
| `matris_10m_oam` | [Download](https://api.figshare.com/v2/file/download/59142728) | OMat24 → sAlex + MPtrj | efsm |
| `matris_10m_mp` | [Download](https://api.figshare.com/v2/file/download/59143058) | MPtrj | efsm |
| `matris_4m_matpes_pbev1` | [Download](https://api.figshare.com/v2/file/download/69575532) | MatPES-PBE (2025.1) | efs |
| `matris_4m_matpes_pbev2` | [Download](https://api.figshare.com/v2/file/download/69575535) | MatPES-PBE (2025.2) | efs |
| `matris_4m_matpes_r2scanv1` | [Download](https://api.figshare.com/v2/file/download/69575538) | MatPES-r2SCAN (2025.1) | efs |
| `matris_4m_matpes_r2scanv2` | [Download](https://api.figshare.com/v2/file/download/69576381) | MatPES-r2SCAN (2025.2) | efs |

If you need other models, feel free to [contact me](mailto:zhouyuanchang23s@ict.ac.cn).

## Usage

There are some examples how to use MatRIS, including calculator, geometry optimization, molecular dynamics and TorchSim.

### ASE Calculator

```python
from ase.build import bulk
import torch

from matris.applications.base import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model="matris_10m_oam", # model name or checkpoint path
    task="efsm",
    device=device,
    mode="fast",                 # optimized kernel + compile; "torch" for eager mode
    activation_checkpoint=True,  # set True for adaptive memory-saving recomputation
    tf32=False,                  # set True to allow TF32 in interaction blocks (~1.5x faster)
)

atoms = bulk("Cu", cubic=True)
atoms.calc = calc

energy = atoms.get_potential_energy()   # total energy, eV
forces = atoms.get_forces()             # eV/A
stress = atoms.get_stress()             # eV/A^3, ASE convention
magmoms = atoms.get_magnetic_moments()  # muB
```

### Structure Optimization

```python
from ase.build import bulk
import torch

from matris.applications.relax import StructOptimizer

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = StructOptimizer(
    model="matris_10m_oam",
    task="efsm",
    optimizer="FIRE",
    device=device,
    mode="fast",
    activation_checkpoint=False,
    tf32=False,
)

atoms = bulk("Cu", cubic=True)
result = optimizer.relax(
    atoms=atoms,
    verbose=True,
    steps=500,
    fmax=0.05,
    relax_cell=True,
    ase_filter="FrechetCellFilter",
)

final_structure = result["final_structure"]
trajectory = result["trajectory"]
```

### Molecular Dynamics

```python
from ase.build import bulk

from matris.applications import MolecularDynamics

atoms = bulk("Cu", cubic=True)

md = MolecularDynamics(
    atoms=atoms,
    model="matris_10m_oam",
    ensemble="nvt",
    temperature=300,
    starting_temperature=300,
    timestep=1,
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
    task="efsm",
    device="cuda",
    mode="fast",
    activation_checkpoint=False,
    tf32=True,
)
md.run(1000)
```

`md.set_atoms(new_atoms)` restarts the integrator and step count for a new system,
retains the calculator, and appends trajectory output. Reattach custom ASE
observers to `md.dyn` after replacing atoms.

## TorchSim

Install `torch-sim-atomistic`, then use pretrained names or local weights:

```python
import torch
import torch_sim as ts
from ase.build import bulk
from matris.applications.torchsim import MatRISTorchSimModel

model = MatRISTorchSimModel(
    model="matris_10m_oam",
    target="efsm",  # OAM/MP: efsm; OMat/MatPES: efs
    device="cuda",
    mode="fast",
    activation_checkpoint=True,
    tf32=False,
)
state = ts.io.atoms_to_state(
    [bulk("Si", "diamond", a=5.43)], device=model.device, dtype=torch.float32,
)
results = model(state)  # energy (eV), forces (eV/Å), stress (eV/Å³), magmoms (μB)
state = ts.integrate(
    state, model, integrator=ts.Integrator.nvt_langevin,
    n_steps=100, temperature=300, timestep=0.001,  # ps
)
```

## Citation

If you use MatRIS in your work, please cite:

```bibtex
@inproceedings{
zhou2026matris,
title={Mat{RIS}: Toward Reliable and Efficient Pretrained Machine Learning Interatomic Potentials},
author={Yuanchang Zhou and Siyu Hu and Xiangyu Zhang and Hongyu Wang and Guangming Tan and Weile Jia},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=5xBT5Ziute}
}
```

## License

MatRIS is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.
