import os
import sys
import importlib.metadata
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.graph.converter import GraphConverter

def pkg_ver(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Found"

print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA in torch:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("ASE:", pkg_ver("ase"))
print("NumPy:", pkg_ver("numpy"))
print("pymatgen:", pkg_ver("pymatgen"))
print("matris:", pkg_ver("matris"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GraphConverter algorithm:", GraphConverter().algorithm)
print("Model key:", "matris_10m_oam")
print("Checkpoint:", "/home/lht/.cache/matris/MatRIS_10M_OAM.pth.tar")
print("Checkpoint exists:", os.path.exists("/home/lht/.cache/matris/MatRIS_10M_OAM.pth.tar"))
