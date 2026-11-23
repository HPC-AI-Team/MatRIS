from __future__ import annotations

import torch
from torch import Tensor, nn

from .basis_function import (
    FourierExpansion,
    LegendreExpansion,
    PolynomialEnvelope,
    BesselExpansion,
    GaussianExpansion,
    SphericalExpansion,
)
from .functions import get_normalization, SwishLayer, directed_average


def prepare_electronic_input(values, graphs, target: str, device) -> Tensor:
    """Validate discrete system labels outside the compiled tensor core."""
    if values is None:
        values = [0 if getattr(g, target, None) is None else getattr(g, target) for g in graphs]
    if isinstance(values, Tensor) and values.is_complex():
        raise ValueError(f"{target} must contain real integer labels")
    try:
        # Graph metadata lives on the CPU; validate it there before transferring
        # labels so normal batches do not need a CUDA synchronization per input.
        values = torch.as_tensor(values, dtype=torch.float64).detach()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{target} must be a scalar or one integer label per graph") from exc
    if values.ndim == 0:
        values = values.expand(len(graphs))
    else:
        values = values.reshape(-1)
    if values.numel() != len(graphs):
        raise ValueError(f"{target} requires one value per graph; expected {len(graphs)}")
    low, high = (-100, 100) if target == "charge" else (0, 100)
    valid = torch.isfinite(values) & (values == values.round()) & (values >= low) & (values <= high)
    if not bool(valid.all()):
        raise ValueError(f"{target} must contain integer values in [{low}, {high}]")
    return values.to(device=device, dtype=torch.int64)


class ChargeSpinEmbedding(nn.Module):
    """Learn system charge/multiplicity tables, then mix them before broadcasting.

    Follows fairchem AllScAIP's trainable ``rand_emb`` scheme: charge indexes
    -100..100, multiplicity indexes 0..100, with 0 reserved for unknown spin.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.charge_embedding = nn.Embedding(201, feat_dim)
        self.spin_embedding = nn.Embedding(101, feat_dim)
        self.mix = nn.Linear(2 * feat_dim, feat_dim)

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(self, charge: Tensor, spin: Tensor) -> Tensor:
        values = torch.cat((self.charge_embedding(charge + 100), self.spin_embedding(spin)), dim=-1)
        return torch.nn.functional.silu(self.mix(values))


class AtomTypeEmbedding(nn.Module):
    """Encode an atom by its atomic number using 'nn.Embedding'."""

    def __init__(self, atom_feat_dim: int, max_num_elements: int = 94):
        """
        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feat_dim)
        self.atom_swish_layer = SwishLayer(
            input_dim=atom_feat_dim, output_dim=atom_feat_dim, bias=False
        )

    def forward(self, atom_type: Tensor) -> Tensor:
        """
        Args:
            atom_type (Tensor): [nAtoms, 1].
        """
        node_feat = self.embedding(atom_type)
        node_feat = self.atom_swish_layer(node_feat)
        return node_feat


class EdgeBasisEmbedding(nn.Module):
    """Embed edge length using 'Bessel' or 'Gaussian'."""

    def __init__(
        self,
        pairwise_cutoff: float = 6,
        three_body_cutoff: float = 4,
        num_radial: int = 7,
        edge_feat_dim: int = 128,
        envelope_exponent: int = 8,
        distance_expansion: str = "Bessel",
        learnable: bool = False,
        smooth_attention: bool = False,
    ):
        """
        Args:
            pairwise_cutoff (float): The cutoff for pairwise interaction.
                Default = 6
            three_body_cutoff (float): The cutoff for three-body interaction.
                Default = 4
            num_radial (int): The number of radial.
                Default = 7
            envelope_exponent (int): envelope exponent of Envelope function.
                Default = 8
            distance_expansion (str): The function of basis, "Bessel" or "Gaussian".
                Default = "Bessel"
            learnable(bool): Whether the frequency in bessel expansion is learnable.
                Default = False
        """
        super().__init__()

        self.distance_expansion = distance_expansion
        self.smooth_attention = smooth_attention
        if self.distance_expansion.lower() == "bessel":
            self.pairwise_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=pairwise_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
            self.threebody_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=three_body_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
        elif self.distance_expansion.lower() == "gaussian":
            if smooth_attention:
                self.pairwise_envelope = PolynomialEnvelope(pairwise_cutoff, envelope_exponent)
                self.threebody_envelope = PolynomialEnvelope(three_body_cutoff, envelope_exponent)
            self.pairwise_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=pairwise_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
            self.threebody_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=three_body_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
        else:
            raise NotImplementedError

        self.edge_linear1 = nn.Linear(
            in_features=num_radial, out_features=edge_feat_dim, bias=False
        )
        self.edge_linear2 = nn.Linear(
            in_features=edge_feat_dim, out_features=edge_feat_dim, bias=False
        )
        self.swish_layer = SwishLayer(
            input_dim=edge_feat_dim, output_dim=edge_feat_dim, bias=False
        )
        self.edge_init_norm = get_normalization(name="layer", dim=edge_feat_dim)

    def forward(self, graphs, *, fast: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            center_pos (Tensor): coord of center atoms.
            neighbor_pos (Tensor): coordinates of neighbor atoms.
            undirected2directed (Tensor): mapping from undirected vectors to one of its
                directed vectors. Following repo: https://github.com/CederGroupHub/chgnet
            image (Tensor): the periodic image specifying the location of neighbor atoms.
            lattice (Tensor): the lattice matrix of structure.
        """

        # From directed edge to undirected edge, this motivation is inspired by https://github.com/CederGroupHub/chgnet
        unique_edge_lengths = torch.index_select(
            graphs["edge_lengths"], 0, graphs["undirected2directed"]
        )  # [nEdges/2]

        cutoff_weights = {}
        if self.smooth_attention and self.distance_expansion.lower() == "bessel":
            pairwise_rbf, pairwise_cutoff = self.pairwise_rbf_expansion(
                unique_edge_lengths, return_smooth_factor=True, fast=fast)
            threebody_rbf, threebody_cutoff = self.threebody_rbf_expansion(
                unique_edge_lengths, return_smooth_factor=True, fast=fast)
            cutoff_weights = {"atom cutoff": pairwise_cutoff.flatten(),
                              "line cutoff": threebody_cutoff.flatten()}
        else:
            pairwise_rbf = self.pairwise_rbf_expansion(unique_edge_lengths, fast=fast)
            threebody_rbf = self.threebody_rbf_expansion(unique_edge_lengths, fast=fast)
            if self.smooth_attention:
                pairwise_cutoff = self.pairwise_envelope(unique_edge_lengths, fast=fast)
                threebody_cutoff = self.threebody_envelope(unique_edge_lengths, fast=fast)
                cutoff_weights = {"atom cutoff": pairwise_cutoff, "line cutoff": threebody_cutoff}
                pairwise_rbf = pairwise_rbf * pairwise_cutoff[:, None]
                threebody_rbf = threebody_rbf * threebody_cutoff[:, None]

        edge_feat_undirect = self.edge_linear1(pairwise_rbf)
        # [edges, dim] -> [2*edges, dim]
        edge_feat_direct = torch.index_select(
            edge_feat_undirect, 0, graphs["directed2undirected"]
        )
        edge_feat_direct = self.edge_linear2(edge_feat_direct)
        edge_feat_direct = self.edge_init_norm(edge_feat_direct)
        # Aggregate to bond_feas_ude
        edge_feat = directed_average(
            edge_feat_direct,
            graphs["directed2undirected"],
            unique_edge_lengths.shape[0],
            fast=fast,
        )
        edge_feat = self.swish_layer(edge_feat)
        smooth_weight = {"atom graph": pairwise_rbf, "line graph": threebody_rbf, **cutoff_weights}
        # return edge_feat, pairwise_rbf, threebody_rbf
        return edge_feat, smooth_weight


class ThreebodyEmbedding(nn.Module):
    """Embed threebody using 'Fourier' or 'Sphere Harmonic'."""

    def __init__(
        self,
        num_angular: int = 7,  # Fourier
        max_n: int = 4,
        max_l: int = 4,
        cutoff: int = 6,  # SH
        three_body_feat_dim: int = 128,
        three_body_expansion: str = "fourier",
        learnable: bool = True,
    ):
        super().__init__()

        self.three_body_expansion = three_body_expansion.lower()
        if self.three_body_expansion == "fourier":
            max_f = (num_angular - 1) // 2
            self.expansion = FourierExpansion(max_f=max_f, learnable=learnable)
            self.angle_embedding = nn.Linear(
                in_features=num_angular, out_features=three_body_feat_dim, bias=False
            )
        elif self.three_body_expansion == "legendre":
            self.expansion = LegendreExpansion(num_angular)
            self.angle_embedding = nn.Linear(num_angular, three_body_feat_dim, bias=False)
        elif self.three_body_expansion == "sh":
            self.expansion = SphericalExpansion(max_n=max_n, max_l=max_l, cutoff=cutoff)
            self.angle_embedding = nn.Linear(
                in_features=max_n * max_l, out_features=three_body_feat_dim, bias=False
            )
        else:
            raise NotImplementedError

        self.swish_layer = SwishLayer(
            input_dim=three_body_feat_dim, output_dim=three_body_feat_dim, bias=False
        )

    def forward(self, graphs, *, fast: bool = False):
        edge_vecs_ij = torch.index_select(
            graphs["unit_edge_vectors"], 0, graphs["line_graph_dict"]["target_DE_index"]
        )  # normalized edge vector ij [nAngle, 3]
        edge_vecs_jk = torch.index_select(
            graphs["unit_edge_vectors"], 0, graphs["line_graph_dict"]["source_DE_index"]
        )  # normalized edge vector jk [nAngle, 3]

        cosine = torch.sum(edge_vecs_ij * edge_vecs_jk, dim=1)
        if self.three_body_expansion == "legendre":
            three_body_basis = self.expansion(cosine)
        else:
            angle = torch.acos(cosine * (1 - 1e-6))
            if self.three_body_expansion == "sh":
                edge_vecs_jk_len = torch.norm(edge_vecs_jk, dim=1)
                three_body_basis = self.expansion(edge_vecs_jk_len, angle)
            else:
                three_body_basis = self.expansion(angle)

        threebody_feat = self.angle_embedding(three_body_basis)
        threebody_feat = self.swish_layer(threebody_feat)

        return threebody_feat
