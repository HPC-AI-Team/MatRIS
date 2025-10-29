from typing import Literal, Union, Sequence

import torch
from torch import Tensor, nn
import os
from collections.abc import Sequence

from ..graph import RadiusGraph, GraphConverter, datatype
from .reference_energy import AtomRef
from .processgraph import process_graphs
from .feature_embed import (
    ThreebodyFourierExpansion, 
    AtomTypeEmbedding, 
    EdgeBasisEmbedding, 
    ThreebodyEmbedding
)
from .functions import (
    MLP,
    GatedMLP,
    get_normalization
)
from .interaction_block import Interaction_Block
from .readout import (
    EnergyHead,
    MagmomHead,
    ForceStressHead,
)


class MatRIS(nn.Module):
    """ Init MatRIS Potential """
    
    def __init__(
        self,
        num_layers: int = 6,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        three_body_feat_dim: int = 128,
        mlp_hidden_dims: Union[int, Sequence[int]] = (128, 128),
        dropout: float = 0.0,
        use_bias: bool = False, 
        distance_expansion: str = "Bessel", 
        three_body_expansion: str = "SH",
        num_radial: int = 7,
        num_angular: int = 7,
        max_l: int = 4,
        max_n: int = 4,
        envelope_exponent: int = 8,
        graph_conv_mlp: str = "GateMLP",
        activation_type: str = "silu",
        norm_type: str = "rms",
        pairwise_cutoff: float = 6,
        three_body_cutoff: float = 4,
        use_smoothed_for_delta_edge: bool = False,
        learnable_basis: bool = True,
        is_intensive: bool = True,
        is_conservation: bool = True,
        reference_energy: str | None = None,
    ):
        """
        Args:
            num_layers (int): message passing layers.
            node_feat_dim (int): atom feature embedding dim.
            edge_feat_dim (int): edge(pairwise) feature embedding dim.
            three_body_feat_dim (int): angle(three body) feature embedding dim.
            mlp_hidden_dims (List or int): hidden dims of MLP. 
                Can be 'int' or 'list'.
            dropout (float): dropout rate in MLP.
            use_bias (bool): whether use bias in Interaction block.
            distance_expansion (str):  The function of pairwise basis. 
                Can be "Bessel" or "Gaussian".
            three_body_expansion (str): The function of three body basis. 
                Can be "Fourier(fourier)" or "Spherical Harmonics(sh)".
            num_radial (int): number of radial basis used in Bessel and Gaussian basis.
            num_angular (int): number of three_body basis used in Fourier basis.
            max_l (int): Maximum l value for Spherical Harmonics basis (SH).
            max_n (int): Maximum n value for Spherical Harmonics basis (SH).
            envelope_exponent (int): exponent of 'PolynomialEnvelope'.
            graph_conv_mlp (str): The type of MLP in mp layers. 
                Can be "MLP", "GatedMLP" and "MoE". 
                See fucntion.py for more informations.
            activation_type (str): activation function. 
                Can be "SiLU(silu)", "Sigmoid(sigmoid)", "ReLU(relu)"...
                See fucntion.py for more informations.
            norm_type (str): normalization function used in MLP.
                Can be "LayerNorm(layer)", "BatchNorm(batch)", "RMSNorm(rms)"...
                See fucntion.py for more informations.
            pairwise_cutoff (float): The cutoff of Atom graph.
            three_body_cutoff (float): The cutoff of Line graph.
            use_smoothed_for_delta_edge (bool): Whether to use the smoothed features for edge feature update.
            learnable_basis (bool): Whether the basis functions are learnable.
            is_intensive (bool): whether the model outputs energy per atom (True) or total energy (False).
            is_conservation (bool): whether use conservate force and stress.
            reference_energy (str): refernece energy of 'str'(eg. MPtrj, OMat..) dataset(Caculated by linear regression).
                more details can be found at reference_energy.py.
        """
        
        super().__init__()
        # model configs
        self.config = { k: v for k, v in locals().items() if k not in ["self", "__class__"] }

        self.is_intensive = is_intensive
        
        self.reference_energy = None
        if reference_energy is not None:
            self.reference_energy = AtomRef(
                reference_energy=reference_energy,
                is_intensive=is_intensive
            ) 
        
        # Define Graph Converter
        self.graph_converter = GraphConverter(
            atom_graph_cutoff=pairwise_cutoff,
            line_graph_cutoff=three_body_cutoff,
        )

        # ====== embedding layers ========
        self.atom_embedding = AtomTypeEmbedding(atom_feat_dim=node_feat_dim)
        self.edge_embedding = EdgeBasisEmbedding(
            pairwise_cutoff=pairwise_cutoff,
            three_body_cutoff=three_body_cutoff,
            num_radial=num_radial,
            edge_feat_dim=edge_feat_dim,
            envelope_exponent=envelope_exponent,
            learnable=learnable_basis,
            distance_expansion=distance_expansion,
        )
        self.three_body_embedding = ThreebodyEmbedding(
            num_angular = num_angular, # Fourier
            max_n=max_n, max_l=max_l, cutoff=pairwise_cutoff, # Spherical Harmonics
            three_body_feat_dim = three_body_feat_dim,
            three_body_expansion = three_body_expansion,
            learnable = learnable_basis
        )
        # ====== Interaction layers ========
        interaction_block = [
            Interaction_Block(
                node_feat_dim=node_feat_dim, 
                edge_feat_dim=edge_feat_dim,
                three_body_feat_dim=three_body_feat_dim,
                num_radial=num_radial,
                num_angular=num_angular,
                dropout=dropout,
                use_bias=use_bias,
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
                mlp_type=graph_conv_mlp,
                norm_type=norm_type,
                activation_type=activation_type,
            )
            for _ in range(num_layers)
        ]
        self.interaction_block = nn.ModuleList(interaction_block)

        # ====== Readout layers ======== 
        self.readout_norm = get_normalization(norm_type, dim=node_feat_dim)

        self.energy_head = EnergyHead(
            feat_dim = node_feat_dim,
            hidden_dim = mlp_hidden_dims,
            output_dim = 1,
            mlp_type = "mlp",
            activation_type = activation_type,
        )
        self.magmom_head = MagmomHead(
            feat_dim = node_feat_dim,
            hidden_dim = 2 * node_feat_dim,
            output_dim = 1,
            mlp_type = "mlp",
            activation_type = activation_type,
        )
        self.force_stress_head = ForceStressHead(
            is_conservation = is_conservation,
            feat_dim = edge_feat_dim, # is_conservation == False
            hidden_dim = mlp_hidden_dims, # is_conservation == False
            output_dim = 3, # is_conservation == False
            mlp_type = "mlp", # is_conservation == False
            activation_type = activation_type, # is_conservation == False
        )
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"MatRIS initialized with {self.get_params()} parameters")

    def forward(
        self,
        graphs: Sequence[RadiusGraph],
        task: str = "ef",
        is_training: bool = False,
    ) -> dict[str, Tensor]:
        """
        Args:
            graphs (List): a list of RadiusGraph.
            task (str): the prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
        """
        prediction = {}
        # ======== Graph processing ========
        batch_graph = process_graphs(graphs, compute_stress="s" in task)
        
        # ======== Feature embedding ========
        node_feat = self.atom_embedding( batch_graph['atomic_numbers'] - 1 ) # atom type feature init (use 0 for 'H')
        edge_feat, smooth_weight = self.edge_embedding(graphs=batch_graph) # pairwise feature init
        threebody_feat = None 
        if len(batch_graph['line_graph_dict']['line_graph']) != 0:
            threebody_feat = self.three_body_embedding(graphs=batch_graph) # three body feature init
        
        # ======== Interaction Block =======
        for mp_layer in self.interaction_block:
            node_feat, edge_feat, threebody_feat = mp_layer(
                batch_graph=batch_graph,
                node_feat=node_feat,
                edge_feat=edge_feat,
                threebody_feat=threebody_feat,
                smooth_weight=smooth_weight,
            )
        
        # ======== Readout Block ======= 
        node_feat = self.readout_norm(node_feat)
        
        total_energy = self.energy_head(batch_graph = batch_graph, node_feat = node_feat)
        
        force_stress_dict = self.force_stress_head(
            batch_graph = batch_graph, 
            compute_force="f" in task,
            compute_stress="s" in task,
            total_energy = total_energy, 
            node_feat = node_feat, 
            edge_feat = edge_feat, 
            is_training = is_training)
        prediction.update(force_stress_dict)
        
        if "m" in task:
            magmom = self.magmom_head(batch_graph = batch_graph, node_feat = node_feat)
            prediction["m"] = magmom
        
        atoms_per_graph_tensor = torch.tensor(batch_graph['atoms_per_graph'], 
                                                  dtype=torch.int32, 
                                                  device=total_energy.device)
        if self.is_intensive:
            energy_per_atom = total_energy / atoms_per_graph_tensor
            prediction["e"] = energy_per_atom
        else:
            prediction["e"] = total_energy

        prediction["atoms_per_graph"] = atoms_per_graph_tensor 

        ref_energy = (
            0 if self.reference_energy is None else self.reference_energy(graphs)
        )
        prediction["e"] += ref_energy
        prediction["ref_energy"] = ref_energy
        return prediction
    
    def get_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_dict(cls, dct: dict):
        matris = MatRIS(**dct["config"])
        matris.load_state_dict(dct["state_dict"])
        return matris
    
    @classmethod
    def load(
        cls,
        model_name: str = "matris_10m_oam",
        device: str | None = None,
    ):
        """Load pretrained model."""
        model_name = model_name.lower()
        supported_models = ["matris_10m_oam", "matris_10m_mp"]
        if model_name not in supported_models:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported models are: {supported_models}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        cache_dir = os.path.expanduser("~/.cache/matris")
        os.makedirs(cache_dir, exist_ok=True)

        checkpoint_files = {
            "matris_10m_omat": "MatRIS_10M_OMAT.pth.tar",
            "matris_10m_oam": "MatRIS_10M_OAM.pth.tar",
            "matris_10m_mp": "MatRIS_10M_MP.pth.tar",
            "matris_6m_mp": "MatRIS_6M_MP.pth.tar",
        }

        DOWNLOAD_URLS = {
            "matris_10m_omat": "",  # TODO
            "matris_10m_oam": "https://figshare.com/ndownloader/files/59142728",
            "matris_10m_mp": "https://figshare.com/ndownloader/files/59143058",
            "matris_6m_mp": "",  # TODO
        }

        ckpt_filename = checkpoint_files[model_name]
        ckpt_path = os.path.join(cache_dir, ckpt_filename)

        if not os.path.exists(ckpt_path):
            url = DOWNLOAD_URLS.get(model_name)
            if not url:
                raise ValueError(f"No download URL provided for model: {model_name}")

            print(f"Checkpoint not found, downloading to {ckpt_path} ...")
            torch.hub.download_url_to_file(url, ckpt_path)
        
        
        ckpt_state = torch.load(
            ckpt_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        model = MatRIS.from_dict(ckpt_state)
        
        model = model.to(device)
        print(f"Loading {model_name} successfully, running on {device}.")
        
        return model
