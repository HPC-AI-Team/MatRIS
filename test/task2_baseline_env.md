# Task 2 Baseline Environment

## Hardware
- GPU: NVIDIA A800-SXM4-80GB
- Visible GPU count: 1
- CUDA_VISIBLE_DEVICES: 0

## Software
- Python: 3.11
- PyTorch: 2.6.0
- CUDA in torch: 12.4
- ASE: 3.23.0
- NumPy: 2.3.4
- pymatgen: 2025.10.7
- Environment: conda env `matris311`

## MatRIS
- Repo path: /home/lht/lab/MatRIS
- Model key: matris_10m_oam
- Checkpoint: /home/lht/.cache/matris/MatRIS_10M_OAM.pth.tar
- Graph converter: fast

## Test Structures
- e / ef / efs: ASE bulk Cu, a=5.43, cubic=True
- Relaxation: MatRIS/example/cif_file/demo.cif
- MD: ASE bulk Cu, a=5.43, cubic=True

## Baseline Parameters
- batch size: 1
- task e
- task ef
- task efs
- relaxation optimizer: FIRE
- relaxation steps: 20
- relaxation fmax: 0.1
- relaxation relax_cell: True
- MD ensemble: NVT
- MD thermostat: Berendsen
- MD temperature: 300 K
- MD timestep: 1.0 fs
- MD steps: 10
- MD loginterval: 1

