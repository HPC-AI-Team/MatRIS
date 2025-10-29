from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import  Literal, Union, Sequence

from .functions import (
    MLP, 
    GatedMLP,
    SwishLayer,
    GraphPooling,
    reshape_stress,
    Sphere,
    aggregate,
    get_activation,
)

class EnergyHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 1,
        mlp_type: str = "GateMLP",
        activation_type: str = "silu",
    ):
        """
        Args:
            feat_dim : Dimension of the input node (atom) features.
            hidden_dim : Hidden layer size(s).
            output_dim : Output dimension (usually 1 for energy).
            mlp_type : Type of MLP to use: "MLP", "GateMLP" or "MOE".
            activation_type : Activation function to use: "silu", "relu", "tanh", "gelu".
        """
        super().__init__()

        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if mlp_type.lower() == "mlp":
            self.energy_head = MLP(
                input_dim=feat_dim,
                hidden_dim=hidden_dims,
                output_dim=output_dim,
                activation=activation_type,
            )
        elif mlp_type.lower() == "gatemlp":
            self.energy_head = nn.Sequential(
                GatedMLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=hidden_dims[-1],
                    norm_type="layer",
                    activation=activation_type,
                ),
                nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
            )
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")

        self.pooling = GraphPooling(average=False)

    def forward(self, batch_graph, node_feat: Tensor) -> Tensor:
        """Compute the total energy per graph/batch from atomic features.

        Args:
            atom_feat : Tensor(N_atoms, feat_dim) – atomic feature vectors.
            batch_index : Tensor(N_atoms,) – graph index for each atom.
        """
        energies = self.energy_head(node_feat)  # (N_atoms, output_dim)
        total_energy = self.pooling(energies, batch_graph['atom_segment']).view(-1)  # (N_graphs,)
        return total_energy

class MagmomHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 1,
        mlp_type: str = "GateMLP",
        activation_type: Literal["silu", "relu", "tanh", "gelu"] = "silu",
    ):
        """
        Args:
            feat_dim : Dimension of the input node (atom) features.
            hidden_dim : Hidden layer size(s).
            output_dim : Output dimension.
            mlp_type : Type of MLP to use: "MLP", "GateMLP" or "MOE".
            activation_type : Activation function to use: "silu", "relu", "tanh", "gelu".
        """
        super().__init__()
        
        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if mlp_type.lower() == "mlp":
            self.magmom_head = MLP(
                input_dim=feat_dim,
                hidden_dim=hidden_dims,
                output_dim=output_dim,
                activation=activation_type,
            )
        elif mlp_type.lower() == "gatemlp":
            self.magmom_head = nn.Sequential(
                GatedMLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=hidden_dims[-1],
                    norm_type="layer",
                    activation=activation_type,
                ),
                nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
            )
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")

    def forward(self, batch_graph, node_feat: Tensor) -> Tensor:
        """Compute the magmom per atom from atomic features.
        
        Args:
            atom_feat : Tensor,
            batch_index : List(N_graphs,) - atom numbers for per graph.
        """

        magmom_feat = torch.abs(self.magmom_head(node_feat)) # [N_atoms, 1]
        magmoms = torch.split(magmom_feat.view(-1), batch_graph['atoms_per_graph']) # Tuple
        return list(magmoms) # [N_atoms]


class ForceStressHead(nn.Module):
    def __init__(
        self,
        is_conservation: bool = True,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 3,
        mlp_type: str = "GateMLP",
        activation_type: str = "silu",
    ):
        """
        Args:
            is_conservation (bool): using 'direct' or 'autograd(cons)' method.
            feat_dim (int): edge feat dim.
            hidden_dim (int): MLP hidden dim.
            output_dim (int): output dim, force label should be 3.
            mlp_type (str): MLP type.
                Can be "MLP", "GateMLP".
            activation_type (str): activation type.
                Can be "SiLU", "Sigmoid"... See fucntion.py for more informations.
        """
        super().__init__()
        self.is_conservation = is_conservation
        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
            
        if not self.is_conservation:
            self.stress_head = EquivariantHead(
                feas_dim=feat_dim,
                lmax=2,
                use_bias=False,
                activation=activation_type,
            )
            if mlp_type.lower() == "mlp":
                self.force_head = MLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=output_dim,
                    activation=activation_type,
                )
            elif mlp_type.lower() == "gatemlp":
                self.force_head = nn.Sequential(
                    GatedMLP(
                        input_dim=feat_dim,
                        hidden_dim=hidden_dims,
                        output_dim=hidden_dims[-1],
                        norm_type="layer",
                        activation=activation_type,
                    ),
                    nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
                )
            else:
                raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
            
    def forward(self, 
                batch_graph,
                compute_force: bool = True, 
                compute_stress: bool = True,
                total_energy: Tensor = None, 
                node_feat: Tensor = None, 
                edge_feat: Tensor = None, 
                is_training: bool=False):
        predict = {}
        if self.is_conservation:
            assert total_energy is not None
            if compute_force and compute_stress:
                grad = torch.autograd.grad(
                    total_energy.sum(), [batch_graph['batch_cart_coords'], batch_graph['batch_strains']], 
                    create_graph=is_training, retain_graph=is_training
                )
                # force
                force = -1 * grad[0]
                predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])
                # stress 
                stress = grad[1]
                scale = 1 / batch_graph['volumes'] * 160.21766208 # units.GPa
                stress = stress * scale
                stress = list(torch.unbind(stress, dim=0))
                predict["s"] = stress 
            elif compute_force:
                grad = torch.autograd.grad(
                    total_energy.sum(), [batch_graph['batch_cart_coords']], 
                    create_graph=is_training, retain_graph=is_training
                ) 
                force = -1 * grad[0]
                predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])
        else:
            if compute_stress:
                assert node_feat is not None
                atom_coord = batch_graph['batch_cart_coords'] / ((torch.norm(batch_graph['batch_cart_coords'], dim=1)*(1-1e-6) + 1e-6).unsqueeze(1))
                stress = self.stress_head(node_feat, atom_coord) # [atoms, 1+3+5]
                stress = aggregate(stress,  batch_graph['atom_segment'], average=True) #[graphs , 1+3+5]
                
                # Reshape 
                isotropic = stress[:, 0] # L=0
                anisotropic = stress[:, [4,5,6,7,8]] # L=2
                stress = reshape_stress(isotropic, anisotropic, stress.shape[0]) #[graphs, 3, 3]
                scale = (1 / batch_graph['volumes'] * 160.21766208)
                stress = stress * scale
                stress = list(torch.unbind(stress, dim=0))
                predict["s"] = stress 
            if compute_force:
                assert edge_feat is not None
                direct_edge_feas = torch.index_select(edge_feat, 0, batch_graph['directed2undirected'])
                direct_edge_feas = self.force_head(direct_edge_feas)
                direct_edge_feas = direct_edge_feas * batch_graph['unit_edge_vectors']

                force = aggregate(
                    direct_edge_feas, batch_graph['atom_graph_dict']['target_index'],
                    average=False, 
                    num_segment=len(node_feat)
                ) # [N_atoms, 3]
                predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])
        
        return predict 


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        activation="silu",
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        
        self.act = get_activation(activation)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, out_channels*2),
        )

    def forward(self, feas, vec_feas):
        # feas: [N_atoms, dim]
        # vec_feas: [N_atoms, 3, dim]
        vec1 = torch.norm(self.vec1_proj(vec_feas), dim=-2) # [N_atoms, dim]
         
        vec2 = self.vec2_proj(vec_feas)
        
        feas = torch.cat([feas, vec1], dim=-1)
        feas, vec_feas = torch.split(self.update_net(feas), self.out_channels, dim=-1)
        vec_feas = vec_feas.unsqueeze(1) * vec2
        
        if self.act is not None:
            feas = self.act(feas)
        return feas, vec_feas

class EquivariantHead(nn.Module):
    def __init__(
        self, 
        feas_dim: int = 64,
        lmax: int = 1,
        use_bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        assert lmax<=2, "Only support lmax <= 2"
        
        self.sh = Sphere(lmax = lmax)
        self.feas_wise_linear = nn.Linear(feas_dim, feas_dim, bias=False)
        self.basis_wise_linear = nn.Linear(1, feas_dim, bias=False)
        
        self.equivariant_block = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    feas_dim,
                    feas_dim// 2,
                    activation=activation,
                ),
                GatedEquivariantBlock(feas_dim // 2, 1, activation=activation),
            ]
        )
    
    def forward(self, feas, vec):
        # feas: [N_atoms, dim]
        # vec: [N_atoms, 3]
        sh_basis = self.sh(vec) # [N_atoms, ( (lmax+1) ** 2) - 1]
        # reshape: [edges, (lmax+1)**2 - 1] -> [N_atoms, (lmax+1)**2 - 1, 1]
        sh_basis = sh_basis.unsqueeze(2)
        vec_feas = self.basis_wise_linear(sh_basis) #[N_atoms, 3, dim]
        
        feas = self.feas_wise_linear(feas) # [N_atoms, dim]
        
        for layer in self.equivariant_block:
            feas, vec_feas = layer(feas, vec_feas)
        
        return vec_feas.squeeze() #[N_atoms, 3]