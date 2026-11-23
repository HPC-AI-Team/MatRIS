"""Smooth, invariant atom -> virtual node -> atom feature communication."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InvariantLongRange(nn.Module):
    """Generate interblock features with a fixed number of virtual nodes.

    Uses RANGE's attention aggregation/broadcast pattern, with SiLU and an
    absolute squared-radius encoding instead of LeakyReLU and max-radius
    normalization. Virtual nodes share the graph's geometric center and have
    separate learned feature embeddings. A joint inverse-square envelope uses
    the receiving atom's squared radius and the virtual node's source moment.
    No spatial or angular grid is used.
    Time and storage are O(N) at fixed feature widths and virtual-node count.
    """

    hidden_dim = 8
    num_centers = 4
    length_scale = 24.0

    def __init__(self, feat_dim: int):
        super().__init__()
        hidden_dim = self.hidden_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.virtual_nodes = nn.Parameter(torch.randn(self.num_centers, hidden_dim) / math.sqrt(hidden_dim))
        self.key = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.query = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.value = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.radial = nn.Linear(8, hidden_dim, bias=False)
        self.virtual_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.virtual_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.aggregate_attention = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.broadcast_attention = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.output = nn.Linear(hidden_dim, feat_dim, bias=False)
        self.strength = nn.Parameter(torch.full((hidden_dim,), .01))
        self.register_buffer("radial_centers", torch.linspace(0., 1., 8))

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def prepare_geometry(self, positions: Tensor, segment: Tensor, num_graphs: int) -> dict[str, Tensor]:
        """Share invariant, differentiable radial inputs across all blocks.

        Squared radii are smooth even when an atom lies at the graph center.
        The fixed physical length preserves distance information under scaling.
        """
        count = positions.new_zeros(num_graphs).index_add(
            0, segment, positions.new_ones(len(positions)))
        center = positions.new_zeros((num_graphs, 3)).index_add(0, segment, positions)
        center = center / count[:, None]
        radius_sq = (positions - center[segment]).square().sum(-1)
        scaled_sq = radius_sq / self.length_scale ** 2
        radius = scaled_sq / (1 + scaled_sq)
        basis = torch.exp(-49. * (radius[:, None] - self.radial_centers).square())
        # The source moment and receiving radius share this coordinate path,
        # including its derivative in the staged force-training VJP.
        return {"long_range_radial_features": torch.cat((basis, scaled_sq[:, None]), dim=-1)}

    @staticmethod
    def _aggregate(weights: Tensor, values: Tensor, segment: Tensor,
                   num_graphs: int) -> tuple[Tensor, Tensor, Tensor]:
        """Pool weighted values with a unit normalization baseline."""
        contribution = weights[..., None] * values[:, None, :]
        if num_graphs == 1:
            pooled = contribution.sum(0, keepdim=True)
            mass = weights.sum(0, keepdim=True)
        else:
            pooled = values.new_zeros((num_graphs, weights.shape[1], values.shape[1])).index_add(
                0, segment, contribution)
            mass = weights.new_zeros((num_graphs, weights.shape[1])).index_add(0, segment, weights)
        return pooled, 1 + mass, contribution

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(self, node_feat: Tensor, positions: Tensor, segment: Tensor,
                num_graphs: int, geometry: dict[str, Tensor] | None = None) -> Tensor:
        if geometry is None:
            geometry = self.prepare_geometry(positions, segment, num_graphs)
        features = self.norm(node_feat)
        key, value = self.key(features), self.value(features)
        radial_input, scaled_sq = geometry["long_range_radial_features"].split((8, 1), dim=-1)
        radial = self.radial(radial_input)
        logits = (F.silu(key[:, None, :] + self.virtual_nodes[None, :, :]
                         + radial[:, None, :]) * self.aggregate_attention).sum(-1)
        weights = torch.sigmoid(logits)
        # Append one scalar moment to the feature pooling, keeping O(N) cost.
        pooled, denominator, contribution = self._aggregate(
            weights, torch.cat((value, scaled_sq), dim=-1), segment, num_graphs)
        pooled_value, source_moment = pooled[..., :-1], pooled[..., -1:]
        virtual = pooled_value / (denominator[..., None] + source_moment)

        query = self.query(features)
        virtual_key = self.virtual_key(virtual + self.virtual_nodes[None, :, :])
        # Only other atoms supply message values. Learned virtual embeddings
        # provide attention keys, not an undamped constant value source.
        # Use a single joint denominator, rather than multiplying two envelopes.
        # Each j -> virtual -> i coefficient is bounded by
        # 1 / (1 + |r_i-r_j|^2 / (2 L^2)).
        joint_denominator = (denominator[segment, :, None] * (1. + scaled_sq[:, None, :])
                             + source_moment[segment])
        others = (pooled_value[segment] - contribution[..., :-1]) / joint_denominator
        virtual_value = self.virtual_value(others)
        logits = (F.silu(query[:, None, :] + virtual_key[segment]
                         + radial[:, None, :]) * self.broadcast_attention).sum(-1)
        # The self slot means "no long-range update"; its message value is zero.
        self_logits = (F.silu(query + key) * self.broadcast_attention).sum(-1, keepdim=True)
        weights = torch.softmax(torch.cat((logits, self_logits), dim=-1), dim=-1)
        message = (weights[:, :-1, None] * virtual_value).sum(1)
        return self.output(message * self.strength)
