from typing import Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
import os
from collections.abc import Sequence

from ..graph import RadiusGraph, GraphConverter
from .reference_energy import AtomRef
from .processgraph import process_graphs
from .feature_embed import AtomTypeEmbedding, EdgeBasisEmbedding, ThreebodyEmbedding
from .functions import get_normalization
from .interaction_block import Interaction_Block
from .readout import (
    EnergyHead,
    MagmomHead,
    ForceStressHead,
)


class MatRIS(nn.Module):
    """Init MatRIS Potential"""

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
        self.config = {
            k: v for k, v in locals().items() if k not in ["self", "__class__"]
        }

        self._compiled_forward = None
        self._compiled_training_forward = {}
        self.is_intensive = is_intensive

        self.reference_energy = None
        if reference_energy is not None:
            self.reference_energy = AtomRef(
                reference_energy=reference_energy, is_intensive=is_intensive
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
            num_angular=num_angular,  # Fourier
            max_n=max_n,
            max_l=max_l,
            cutoff=pairwise_cutoff,  # Spherical Harmonics
            three_body_feat_dim=three_body_feat_dim,
            three_body_expansion=three_body_expansion,
            learnable=learnable_basis,
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
            feat_dim=node_feat_dim,
            hidden_dim=mlp_hidden_dims,
            output_dim=1,
            mlp_type="mlp",
            activation_type=activation_type,
        )
        self.magmom_head = MagmomHead(
            feat_dim=node_feat_dim,
            hidden_dim=2 * node_feat_dim,
            output_dim=1,
            mlp_type="mlp",
            activation_type=activation_type,
        )
        self.force_stress_head = ForceStressHead(
            is_conservation=is_conservation,
            feat_dim=edge_feat_dim,  # is_conservation == False
            hidden_dim=mlp_hidden_dims,  # is_conservation == False
            output_dim=3,  # is_conservation == False
            mlp_type="mlp",  # is_conservation == False
            activation_type=activation_type,  # is_conservation == False
        )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"MatRIS initialized with {self.get_params()} parameters")

    def forward(
        self,
        graphs: Sequence[RadiusGraph],
        task: str = "ef",
        is_training: bool = False,
        mode: str = "fast",
        *,
        activation_checkpoint: bool = False,
        tf32: bool = False,
    ) -> dict[str, Tensor]:
        
        if mode not in ("fast", "torch"):
            raise ValueError(f"Unknown mode {mode!r}; choose 'fast' or 'torch'.")
        if task not in ("e", "em", "ef", "efs", "efsm"):
            raise ValueError(f"Unsupported task {task!r}; choose e, em, ef, efs, or efsm.")
        parameter = next(self.parameters())
        device = parameter.device
        fast = mode == "fast" and device.type == "cuda"
        if fast:
            from .compile import compile_training

        training = self.training or is_training
        use_checkpoint = activation_checkpoint and torch.is_grad_enabled()
        if use_checkpoint and fast and not training:
            # Feature storage scales with graph density, width and model depth.
            feature_bytes = parameter.element_size() * sum(
                g.atomic_number.shape[0] * self.config["node_feat_dim"]
                + g.undirected2directed.shape[0] * max(
                    self.config["node_feat_dim"], self.config["edge_feat_dim"]
                )
                + g.line_graph.shape[0] * max(
                    self.config["edge_feat_dim"], self.config["three_body_feat_dim"]
                )
                for g in graphs
            )
            # Conservative estimate: ~20 saved feature buffers per compiled layer.
            factor = 20 * len(self.interaction_block)
            allocated = torch.cuda.memory_allocated(device)
            free, total = torch.cuda.mem_get_info(device)
            available = min(total, free + torch.cuda.memory_reserved(device) - allocated)
            # Reserve backward workspace and 5% for allocator/kernel variability.
            required = feature_bytes * (factor + 2) + 64 * 1024**2
            use_checkpoint = required > 0.95 * available
        if use_checkpoint and not fast:
            return checkpoint(
                self.forward,
                graphs,
                task=task,
                is_training=is_training,
                mode=mode,
                activation_checkpoint=False,
                tf32=tf32,
                use_reentrant=False,
            )
        batch_graph = process_graphs(graphs, compute_stress="s" in task, fast=fast)
        geometry_grads = None
        if fast and training:
            signature = (
                task,
                tf32,
                self.training,
                use_checkpoint,
                len(graphs) == 1,
                batch_graph["atomic_numbers"].shape[0] == 1,
                bool(batch_graph["line_graph_dict"]["line_graph"].shape[0]),
                batch_graph["edge_lengths"].dtype,
                batch_graph["edge_lengths"].device,
            )
            if signature not in self._compiled_training_forward:
                self._compiled_training_forward[signature] = compile_training(
                    self, batch_graph, task, tf32=tf32, activation_checkpoint=use_checkpoint
                )
            tensor_forward = self._compiled_training_forward[signature]
            node_feat, edge_feat, geometry_grads = tensor_forward(batch_graph)
            # Keep energy-only parameters out of force/stress-only backward.
            total_energy = self.energy_head(
                batch_graph=batch_graph,
                node_feat=node_feat,
                fast=fast,
                training=training,
            )
        else:
            if fast and self._compiled_forward is None:
                self._compiled_forward = torch.compile(self.forward_core, dynamic=True)
            tensor_forward = self._compiled_forward if fast else self.forward_core
            node_feat, edge_feat, total_energy = tensor_forward(
                batch_graph, tf32=tf32, fast=fast, training=training,
                activation_checkpoint=use_checkpoint,
            )
        prediction = {}
        force_stress_dict = self.force_stress_head(
            batch_graph=batch_graph,
            compute_force="f" in task,
            compute_stress="s" in task,
            total_energy=total_energy,
            node_feat=node_feat,
            edge_feat=edge_feat,
            is_training=is_training,
            energy_derivatives=geometry_grads,
            fast=fast,
            training=training,
        )
        prediction.update(force_stress_dict)

        if "m" in task:
            magmom = self.magmom_head(
                batch_graph=batch_graph,
                node_feat=node_feat,
                fast=fast,
                training=training,
            )
            prediction["m"] = magmom

        atoms_per_graph_tensor = torch.tensor(
            batch_graph["atoms_per_graph"],
            dtype=torch.int32,
            device=total_energy.device,
        )
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

    def forward_core(
        self,
        batch_graph,
        *,
        tf32: bool = False,
        fast: bool = False,
        training: bool = True,
        activation_checkpoint: bool = False,
    ):
        """Tensor computation shared by eager and compiled execution."""
        from functools import partial

        # ======== Feature embedding ========
        node_feat = self.atom_embedding(
            batch_graph["atomic_numbers"] - 1
        )  # atom type feature init (use 0 for 'H')
        edge_feat, smooth_weight = self.edge_embedding(
            graphs=batch_graph, fast=fast
        )  # pairwise feature init
        threebody_feat = self.three_body_embedding(
            graphs=batch_graph,
            fast=fast and batch_graph["line_graph_dict"]["line_graph"].shape[0] != 0,
        )

        # ======== Interaction Block =======
        for mp_layer in self.interaction_block:
            block = (
                partial(checkpoint, mp_layer, use_reentrant=False)
                if activation_checkpoint else mp_layer
            )
            node_feat, edge_feat, threebody_feat = block(
                batch_graph=batch_graph,
                node_feat=node_feat,
                edge_feat=edge_feat,
                threebody_feat=threebody_feat,
                smooth_weight=smooth_weight,
                tf32=tf32,
                fast=fast,
                training=training,
            )

        # ======== Readout Block =======
        node_feat = self.readout_norm(node_feat)

        total_energy = self.energy_head(
            batch_graph=batch_graph, node_feat=node_feat, fast=fast, training=training
        )

        return node_feat, edge_feat, total_energy

    def get_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_dict(cls, dct: dict):
        """Restore architecture and reference energies from checkpoint weights."""
        dct = dct.get("model", dct)
        state_dict = dict(dct["state_dict"])
        if "composition_model.fc.weight" in state_dict:
            legacy_reference = state_dict.pop("composition_model.fc.weight")
            reference = state_dict.setdefault("reference_energy.fc.weight", legacy_reference)
            if not torch.equal(reference, legacy_reference):
                raise ValueError("Checkpoint contains conflicting reference energy weights")
        config = dict(dct["config"])
        reference_name = config.pop("reference_energy", None)
        model = cls(**config)
        reference_weight = state_dict.get("reference_energy.fc.weight")
        if reference_weight is not None or reference_name is not None:
            model.reference_energy = AtomRef(
                is_intensive=model.is_intensive,
                max_num_elements=reference_weight.shape[1] if reference_weight is not None else 94,
            )
        model.config["reference_energy"] = reference_name
        model.load_state_dict(state_dict)
        if model.reference_energy is not None:
            model.reference_energy.fitted = True
        return model

    @classmethod
    def load(
        cls,
        model_name: str | os.PathLike = "matris_10m_oam",
        device: str | None = None,
    ):
        """Load a local checkpoint, HTTP(S) URL, or cached model name.

        Checkpoints contain ``config`` and ``state_dict``, including fitted
        reference energies. New dataset names need no local reference preset.
        The two original model aliases retain their published download URLs.
        Other model names are resolved from filenames in ``~/.cache/matris``.
        """
        from hashlib import sha256
        from pathlib import Path
        import pickle
        import tempfile

        source = os.fspath(model_name)
        path = Path(source).expanduser()
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            cache_dir = Path.home() / ".cache" / "matris"
            checkpoint_files = {
                "matris_10m_omat": "MatRIS_10M_OMAT.pth.tar",
                "matris_10m_oam": "MatRIS_10M_OAM.pth.tar",
                "matris_10m_mp": "MatRIS_10M_MP.pth.tar",
                "matris_4m_matpes_r2scanv1": "MatRIS_4M_MatPES_r2SCANv1.pth.tar",
                "matris_4m_matpes_pbev1": "MatRIS_4M_MatPES_PBEv1.pth.tar",
                "matris_4m_matpes_r2scanv2": "MatRIS_4M_MatPES_r2SCANv2.pth.tar",
                "matris_4m_matpes_pbev2": "MatRIS_4M_MatPES_PBEv2.pth.tar",
            }
            DOWNLOAD_URLS = {
                "matris_10m_omat": "https://api.figshare.com/v2/file/download/69575544",
                "matris_10m_oam": "https://api.figshare.com/v2/file/download/59142728",
                "matris_10m_mp": "https://api.figshare.com/v2/file/download/59143058",
                "matris_4m_matpes_r2scanv1": "https://api.figshare.com/v2/file/download/69575538",
                "matris_4m_matpes_pbev1": "https://api.figshare.com/v2/file/download/69575532",
                "matris_4m_matpes_r2scanv2": "https://api.figshare.com/v2/file/download/69576381",
                "matris_4m_matpes_pbev2": "https://api.figshare.com/v2/file/download/69575535",
            }
            name = source.lower()
            url = source if source.startswith(("https://", "http://")) else DOWNLOAD_URLS.get(name)
            filename = checkpoint_files.get(name) or (
                f"{sha256(url.encode()).hexdigest()}.pth.tar" if url else name
            )
            cache_name = filename.lower()
            cached = next(
                (
                    entry for entry in sorted(cache_dir.glob("*"))
                    if entry.is_file()
                    and entry.name.lower().removesuffix(".tar").removesuffix(".pth").removesuffix(".pt")
                    == cache_name.removesuffix(".tar").removesuffix(".pth").removesuffix(".pt")
                ),
                None,
            )
            state = None
            if cached is not None:
                try:
                    state = torch.load(cached, map_location="cpu", weights_only=True)
                except (EOFError, RuntimeError, pickle.UnpicklingError):
                    if not url:
                        raise
            if state is None:
                if not url:
                    if name in checkpoint_files:
                        raise ValueError(
                            f"No download URL provided for model: {source}. "
                            f"Place {checkpoint_files[name]} in {cache_dir}, "
                            "or pass a checkpoint path or download URL."
                        )
                    raise FileNotFoundError(
                        f"Model {source!r} was not found locally or in {cache_dir}. "
                        "Pass a checkpoint path or its HTTP(S) download URL."
                    )
                cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
                    download = Path(temporary) / filename
                    torch.hub.download_url_to_file(url, str(download))
                    state = torch.load(download, map_location="cpu", weights_only=True)
                    model = cls.from_dict(state)
                    # Publish only a complete, loadable checkpoint; failed downloads
                    # leave no cache entry and never replace existing user weights.
                    os.replace(download, cached or cache_dir / filename)
                device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device).eval()
                print(f"Load {source} successfully")
                return model
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls.from_dict(state).to(device).eval()
        print(f"Load {source} successfully")
        return model
