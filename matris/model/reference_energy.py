""" Modified from CHGNet: https://github.com/CederGroupHub/chgnet """

import collections

import numpy as np
import torch
from pymatgen.core import Structure
from torch import Tensor, nn

from collections.abc import Sequence
from ..graph.radiusgraph import RadiusGraph


class AtomRef(nn.Module):
    """A linear regression for elemental energy.
    From: https://github.com/materialsvirtuallab/m3gnet/.
    """

    def __init__(self, 
                 reference_energy: str = None,
                 is_intensive: bool = True, 
                 max_num_elements: int = 94) -> None:
        
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc = nn.Linear(max_num_elements, 1, bias=False)
        self.fitted = False
        
        self.initialize_from(reference_energy)
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, graphs: list[RadiusGraph]):
        """Get the energy of a list of CrystalGraphs.

        Args:
            graphs (List(RadiusGraph)): a list of Crystal Graph to compute

        Returns:
            energy (tensor)
        """
        composition_feas = []
        for graph in graphs:
            composition_fea = torch.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea / n_atom
            composition_feas.append(composition_fea)
        composition_feas = torch.stack(composition_feas, dim=0).float()
         
        ref_energy = self.fc(composition_feas).view(-1) 
        return ref_energy

    def fit(
        self,
        structures_or_graphs: Sequence[Structure | RadiusGraph],
        energies: Sequence[float],
    ) -> None:
        """Fit the model to a list of crystals and energies.

        Args:
            structures_or_graphs (list[Structure  |  RadiusGraph]): Any iterable of
                pymatgen structures and/or graphs.
            energies (list[float]): Target energies.
        """
        num_data = len(energies)
        composition_feas = torch.zeros([num_data, self.max_num_elements])
        e = torch.zeros([num_data])
        for index, (structure, energy) in enumerate(
            zip(structures_or_graphs, energies)
        ):
            if isinstance(structure, Structure):
                atomic_number = torch.tensor(
                    [site.specie.Z for site in structure],
                    dtype=int,
                    requires_grad=False,
                )
            else:
                atomic_number = structure.atomic_number
            composition_fea = torch.bincount(
                atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                composition_fea = composition_fea / atomic_number.shape[0]
            composition_feas[index, :] = composition_fea
            e[index] = energy

        # Use numpy for pinv
        self.feature_matrix = composition_feas.detach().numpy()
        self.energies = e.detach().numpy()
        state_dict = collections.OrderedDict()
        weight = (
            np.linalg.pinv(self.feature_matrix.T @ self.feature_matrix)
            @ self.feature_matrix.T
            @ self.energies
        )
        state_dict["weight"] = torch.tensor(weight).view(1, 94)
        self.fc.load_state_dict(state_dict)
        self.fitted = True

    def initialize_from(self, dataset: str):
        """Initialize pre-fitted weights from a dataset."""
        dataset = dataset.lower()
        if dataset in ["mptrj", "demo"]:
            self.initialize_from_MPtrj()
        elif dataset in ["omat"]:
            self.initialize_from_OMat()
        elif dataset in ["mpa"]:
            self.initialize_from_MPA() 
        else:
            raise NotImplementedError(f"{dataset=} not supported yet")
    
    def initialize_from_MPtrj(self):
        """Initialize refernece energy of MPtrj(uncorrected energy)."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(
            [
                -3.4631e+00, -2.7309e-01, -2.8321e+00, -3.6718e+00, -7.6282e+00,
                -8.3344e+00, -7.5666e+00, -7.4720e+00, -5.0677e+00, -3.0058e-02,
                -2.0124e+00, -1.6125e+00, -4.4889e+00, -6.3631e+00, -6.4289e+00,
                -5.2847e+00, -3.0972e+00, -6.3163e-02, -1.7411e+00, -3.7065e+00,
                -6.8949e+00, -9.5306e+00, -8.7439e+00, -9.0234e+00, -8.4526e+00,
                -7.4091e+00, -6.1867e+00, -5.4593e+00, -3.3224e+00, -8.2911e-01,
                -3.2546e+00, -4.9264e+00, -4.6854e+00, -4.5014e+00, -2.7020e+00,
                7.7055e-01, -1.6062e+00, -3.4319e+00, -7.7220e+00, -9.6074e+00,
                -1.0678e+01, -9.2717e+00, -6.8685e+00, -8.2849e+00, -7.0391e+00,
                -5.0754e+00, -1.7952e+00, -3.5316e-01, -2.5871e+00, -3.9603e+00,
                -3.9316e+00, -3.6352e+00, -2.1676e+00,  2.7399e+00, -2.1183e+00,
                -3.9493e+00, -7.1591e+00, -7.4358e+00, -6.5724e+00, -6.7311e+00,
                -5.1406e+00, -6.7412e+00, -1.1751e+01, -1.6219e+01, -6.4131e+00,
                -6.4793e+00, -6.3707e+00, -6.3809e+00, -6.3572e+00, -2.8174e+00,
                -6.4382e+00, -1.0711e+01, -1.2433e+01, -1.0198e+01, -1.0641e+01,
                -9.1970e+00, -8.2220e+00, -6.0345e+00, -2.7692e+00,  6.6550e-01,
                -1.5987e+00, -3.2421e+00, -3.4098e+00,  1.0903e-16,  6.1043e-16,
                1.2149e-15,  3.4812e-16, -3.3371e-18, -4.2623e+00, -9.0547e+00,
                -1.0285e+01, -1.2643e+01, -1.2739e+01, -1.4317e+01
            ]
        ).view([1, 94])

        self.fc.load_state_dict(state_dict)
        self.is_intensive = True
        self.fitted = True

    def initialize_from_OMat(self):
        """Initialize refernece energy of Omat24."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(
            [
                -3.0332e+00,  2.7781e-02, -2.0769e+00, -2.8506e+00, -5.5714e+00,
                -6.9814e+00, -7.5916e+00, -6.8493e+00, -4.5307e+00,  3.7653e-01,
                -1.2547e+00, -1.2771e+00, -3.5325e+00, -5.1621e+00, -5.2792e+00,
                -5.1759e+00, -2.9546e+00,  9.4408e-02, -5.2747e-01, -1.9239e+00,
                -6.3525e+00, -7.7363e+00, -8.1060e+00, -8.0094e+00, -7.9600e+00,
                -6.9398e+00, -5.9301e+00, -4.8372e+00, -2.9134e+00, -9.2360e-01,
                -2.8440e+00, -4.3353e+00, -4.5904e+00, -4.3398e+00, -2.4037e+00,
                2.3849e-01, -9.0706e-01, -1.9711e+00, -6.5034e+00, -8.4291e+00,
                -9.0198e+00, -9.1690e+00, -8.9550e+00, -8.0029e+00, -6.8800e+00,
                -5.2120e+00, -2.0796e+00, -3.0074e-01, -2.2362e+00, -3.7322e+00,
                -3.9127e+00, -3.5653e+00, -1.9106e+00,  3.2219e+00, -9.6049e-01,
                -2.0196e+00, -5.0013e+00, -6.5140e+00, -4.8854e+00, -4.8524e+00,
                -4.9177e+00, -4.7900e+00, -1.0408e+01, -1.3873e+01, -4.7775e+00,
                -4.5076e+00, -4.4782e+00, -4.4732e+00, -4.4918e+00, -4.9103e+00,
                -4.4665e+00, -9.2834e+00, -1.0675e+01, -1.0502e+01, -1.0141e+01,
                -9.1863e+00, -7.9159e+00, -6.0584e+00, -3.0528e+00,  1.7787e-01,
                -1.9486e+00, -3.1409e+00, -3.5926e+00,  1.2319e-13, -3.4435e-13,
                2.9553e-12,  8.9567e-16, -2.9915e-17, -4.4779e+00, -7.9138e+00,
                -1.0146e+01, -1.2389e+01, -1.3565e+01, -1.4339e+01
            ]
        ).view([1, 94])

        self.fc.load_state_dict(state_dict)
        self.is_intensive = True
        self.fitted = True
        
    def initialize_from_MPA(self):
        """Initialize refernece energy of sAlex+MPtrj."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(
            [
                -3.1737e+00, -2.6391e-01, -2.2369e+00, -3.4445e+00, -6.8277e+00,
                -8.1473e+00, -8.8195e+00, -7.5632e+00, -5.0891e+00,  1.1973e-03,
                -1.5680e+00, -1.6372e+00, -4.0944e+00, -5.8544e+00, -6.0222e+00,
                -5.5525e+00, -3.0572e+00, -3.5725e-02, -8.9308e-01, -2.4140e+00,
                -6.4643e+00, -8.3088e+00, -8.9037e+00, -8.8312e+00, -8.5999e+00,
                -7.8178e+00, -6.7030e+00, -5.7014e+00, -3.7271e+00, -1.2587e+00,
                -3.3469e+00, -5.0977e+00, -5.1029e+00, -4.7616e+00, -2.5345e+00,
                7.0565e-01, -1.3512e+00, -2.0171e+00, -6.4545e+00, -8.6960e+00,
                -9.9447e+00, -9.8603e+00, -9.6430e+00, -8.9905e+00, -7.6219e+00,
                -5.5586e+00, -2.4417e+00, -7.8289e-01, -2.6984e+00, -4.1773e+00,
                -4.2378e+00, -3.8948e+00, -2.3448e+00,  2.6678e+00, -1.4716e+00,
                -2.3332e+00, -5.1416e+00, -6.5570e+00, -4.8068e+00, -4.7959e+00,
                -4.8117e+00, -4.7422e+00, -1.0256e+01, -1.3897e+01, -4.5928e+00,
                -4.5515e+00, -4.5212e+00, -4.5335e+00, -4.4311e+00, -2.4671e+00,
                -4.7002e+00, -1.0072e+01, -1.1422e+01, -1.1575e+01, -1.1294e+01,
                -1.0299e+01, -8.8652e+00, -6.5876e+00, -3.3634e+00, -1.2897e-01,
                -2.0918e+00, -3.4916e+00, -3.9858e+00,  1.0642e-14,  1.1423e-16,
                -2.4249e-15,  6.9084e-17, -5.6083e-18, -4.4531e+00, -7.6120e+00,
                -9.9345e+00, -1.1656e+01, -1.2598e+01, -1.4325e+01
            ]
        ).view([1, 94])

        self.fc.load_state_dict(state_dict)
        self.is_intensive = True
        self.fitted = True
        