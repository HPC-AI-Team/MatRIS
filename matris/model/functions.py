from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math


def scale_res(
    update: Tensor, residual: Tensor, scale: Tensor, *, fast: bool = False
) -> Tensor:
    """Add a channel-scaled residual without a materialized product in fast mode."""
    if fast:
        return torch.addcmul(update, scale, residual)
    return update + scale * residual


def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()


def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None

    normalization_map = {
        "layer": nn.LayerNorm,
        "rms": nn.RMSNorm,  # torch >= 2.6.0
        "batch": nn.BatchNorm1d,
    }
    name_lower = name.lower()
    return normalization_map[name_lower](dim)


class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)

        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(self.linear(feas))


def Dimwise_softmax(
    feas: Tensor, segment: Tensor, num_segment=None, *, fast: bool = False
) -> Tensor:
    """Computes a sparsely evaluated softmax.

    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1]
    """
    if fast and feas.is_cuda and feas.dtype == torch.float32:
        from .op.attention import online_softmax

        if num_segment is None:
            num_segment = int(segment.max()) + 1
        return online_softmax(feas, segment, num_segment)

    num, dim = feas.shape
    if num_segment is None:
        num_segment = int(segment.max()) + 1

    segment_expanded = segment.unsqueeze(1).expand(-1, dim)  # [num, dim]

    feas_max = torch.empty(num_segment, dim, dtype=feas.dtype, device=feas.device)
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas, reduce="amax", include_self=False,
    )  # [num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas - feas_max).exp()

    # =========== scatter sum ============
    out_sum = out.new_zeros((num_segment, dim))
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce="sum", include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    return score


def aggregate(
    data: torch.Tensor,
    segment: torch.Tensor,
    bin_count: torch.Tensor = None,
    average=True,
    num_segment=None,
) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    if bin_count is None:
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    return output


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")

        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim, bias=bias),
                    activation_func,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim, bias=bias),
                ]
            )
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim[0], bias=bias),
                    activation_func,
                ]
            )
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend(
                    [
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                        activation_func,
                    ]
                )
            # Output layer
            layers.extend(
                [nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)]
            )
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )

        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        feas: Tensor,
        *,
        tf32: bool = False,
        fast: bool = False,
        training: bool = True,
    ) -> Tensor:
        """
        Args:
            feas: Input tensor of shape (features, input_dim)
            training: Retain parameter gradients in fast mode, independently of
                train()/eval(), which control dropout.
        Returns:
            Output tensor of shape (features, output_dim)
        """
        if fast or tf32:
            out = feas
            for layer in self.layers:
                out = (
                    linear(
                        out,
                        layer.weight,
                        layer.bias,
                        tf32=tf32,
                        fast=fast,
                        training=training,
                    )
                    if isinstance(layer, nn.Linear)
                    else layer(out)
                )
        else:
            out = self.layers(feas)
        return out


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu",
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
        """
        super().__init__()
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )

    def forward(
        self,
        feas: Tensor,
        *,
        tf32: bool = False,
        fast: bool = False,
        training: bool = True,
    ) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        core = self.mlp_core(feas, tf32=tf32, fast=fast, training=training)
        gate = self.mlp_gate(feas, tf32=tf32, fast=fast, training=training)
        if self.gate_norm is not None:
            if isinstance(self.activation_func, nn.SiLU) and isinstance(
                self.activation_gate, nn.Sigmoid
            ):
                fused = gated_tail(
                    core,
                    gate,
                    self.core_norm,
                    self.gate_norm,
                    fast=fast,
                    training=training,
                )
                if fused is not None:
                    return fused
            core = self.core_norm(core)
            gate = self.gate_norm(gate)
        return self.activation_func(core) * self.activation_gate(gate)



class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:

        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

        output = node_feat.new_zeros([bin_count.shape[0], node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device=L0out.device,
    )
    # L=0
    L = 0
    pred_irreps[:, irreps_sum(L - 1) : irreps_sum(L)] = L0out.view(batch_size, -1)

    L = 2
    pred_irreps[:, irreps_sum(L - 1) : irreps_sum(L)] = L2out.view(batch_size, -1)

    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device=L0out.device),
        pred_irreps,
    )

    return pred.view(batch_size, 3, 3)


class Sphere(nn.Module):
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax

    def forward(self, edge_vec):
        edge_sh = self.spherical_harmonics(
            self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    @staticmethod
    def spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack(
                [
                    sh_0_0,
                ],
                dim=-1,
            )

        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack(
                [
                    sh_0_0,
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                ],
                dim=-1,
            )


def linear(
    x,
    weight,
    bias=None,
    *,
    tf32: bool = False,
    fast: bool = False,
    training: bool = True,
):
    if fast and not training:
        weight = weight.detach()
        bias = None if bias is None else bias.detach()
    if (
        tf32
        and x.is_cuda
        and x.dtype == weight.dtype == torch.float32
        and not torch.is_autocast_enabled("cuda")
    ):
        from .op.gemm import TF32MatmulFunction

        out = TF32MatmulFunction.apply(x.reshape(-1, x.shape[-1]), weight.t())
        out = out.reshape(*x.shape[:-1], weight.shape[0])
        return out if bias is None else out + bias
    return F.linear(x, weight, bias)


def directed_average(
    data: Tensor, segment: Tensor, num_segment: int, *, fast: bool = False
) -> Tensor:
    if (
        fast
        and data.ndim == 2
        and data.shape[1] > 0
        and data.shape[1] % 32 == 0
        and data.dtype == torch.float32
        and data.is_cuda
        and segment.ndim == 1
        and segment.numel() == data.shape[0]
        and segment.dtype == torch.int64
        and segment.is_cuda
        and data.shape[0] == num_segment * 2
    ):
        from .op.reductions import DirectedAverageFunction

        return DirectedAverageFunction.apply(data, segment, num_segment)
    return aggregate(data, segment)


def refine_scatter(
    input: Tensor,
    smooth: Tensor,
    target_index: Tensor,
    num_nodes: int,
    *,
    fast: bool = False,
) -> Tensor | None:
    if not fast:
        return None

    from .op.reductions import SmoothScatterFunction

    if (
        input.ndim != 2
        or smooth.ndim != 2
        or input.shape != smooth.shape
        or (input.shape[1] <= 0 or input.shape[1] % 32 != 0)
        or (input.dtype != torch.float32)
        or (smooth.dtype != torch.float32)
        or (target_index.dtype != torch.int64)
        or (not input.is_cuda)
        or (not smooth.is_cuda)
        or (not target_index.is_cuda)
    ):
        return None
    return SmoothScatterFunction.apply(input, smooth, target_index, num_nodes)


def refine_envelope_scatter(
    input: Tensor,
    base_envelope: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    num_nodes: int,
    *,
    layout=None,
    fast: bool = False,
) -> Tensor | None:
    if not fast:
        return None

    from .op.reductions import EnvelopeScatterFunction

    if (
        input.ndim != 2
        or base_envelope.ndim != 2
        or input.shape[1] <= 0
        or input.shape[1] % 32 != 0
        or (base_envelope.shape[1] != input.shape[1])
        or (input.dtype != torch.float32)
        or (base_envelope.dtype != torch.float32)
        or (source_index.dtype != torch.int64)
        or (target_index.dtype != torch.int64)
        or (not input.is_cuda)
        or (not base_envelope.is_cuda)
        or (not source_index.is_cuda)
        or (not target_index.is_cuda)
    ):
        return None
    return EnvelopeScatterFunction.apply(
        input,
        base_envelope,
        source_index,
        target_index,
        num_nodes,
        *(layout if layout is not None else (None, None)),
    )


def gated_tail(
    core: Tensor,
    gate: Tensor,
    core_norm: nn.Module | None,
    gate_norm: nn.Module | None,
    *,
    fast: bool = False,
    training: bool = True,
) -> Tensor | None:
    if not fast:
        return None

    from .op.gated_mlp import GatedTailFunction

    if not isinstance(core_norm, nn.LayerNorm) or not isinstance(
        gate_norm, nn.LayerNorm
    ):
        return None
    if core_norm.eps != gate_norm.eps:
        return None
    if (
        core_norm.weight is None
        or core_norm.bias is None
        or gate_norm.weight is None
        or (gate_norm.bias is None)
    ):
        return None
    if (
        core.ndim != 2
        or core.shape != gate.shape
        or core.shape[1] <= 0
        or core.shape[1] % 32 != 0
        or (core.dtype != torch.float32)
        or (gate.dtype != torch.float32)
        or (not core.is_cuda)
        or (not gate.is_cuda)
        or (core_norm.weight.dtype != torch.float32)
        or (gate_norm.weight.dtype != torch.float32)
        or (not core_norm.weight.is_cuda)
        or (not gate_norm.weight.is_cuda)
        or (not core_norm.bias.is_cuda)
        or (not gate_norm.bias.is_cuda)
    ):
        return None
    return GatedTailFunction.apply(
        core,
        gate,
        core_norm.weight if training else core_norm.weight.detach(),
        core_norm.bias if training else core_norm.bias.detach(),
        gate_norm.weight if training else gate_norm.weight.detach(),
        gate_norm.bias if training else gate_norm.bias.detach(),
        float(core_norm.eps),
    )


def attn_line_projection(
    module: GatedMLP,
    node_feat: Tensor,
    edge_feat: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    *,
    tf32: bool = False,
    fast: bool = False,
    training: bool = True,
) -> Tensor | None:
    if not fast or torch.is_autocast_enabled("cuda"):
        return None

    from .op.gated_mlp import AttnProjectFunction

    if not isinstance(
        getattr(module, "core_norm", None), nn.LayerNorm
    ) or not isinstance(getattr(module, "gate_norm", None), nn.LayerNorm):
        return None
    for mlp in (module.mlp_core, module.mlp_gate):
        if len(mlp.layers) != 4:
            return None
        first, activation, dropout, second = mlp.layers
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            return None
        if not isinstance(dropout, nn.Dropout) or dropout.p != 0.0:
            return None
        if activation.__class__.__name__.lower() not in {"fusedsilu", "silu"}:
            return None
    core_first, _, _, core_second = module.mlp_core.layers
    gate_first, _, _, gate_second = module.mlp_gate.layers
    dim = core_first.out_features
    edge_dim, node_dim = edge_feat.shape[1], node_feat.shape[1]
    if (
        dim <= 0
        or dim % 32 != 0
        or not node_feat.is_cuda
        or not edge_feat.is_cuda
        or node_feat.dtype != torch.float32
        or (edge_feat.dtype != torch.float32)
        or (core_first.weight.shape != (dim, edge_dim + 2 * node_dim))
        or (gate_first.weight.shape != (dim, edge_dim + 2 * node_dim))
        or (core_first.bias is None)
        or (gate_first.bias is None)
        or (core_second.weight.shape[1] != dim)
        or (gate_second.weight.shape[1] != dim)
        or (source_index.dtype != torch.int64)
        or (target_index.dtype != torch.int64)
        or (not source_index.is_cuda)
        or (not target_index.is_cuda)
    ):
        return None
    core_w = core_first.weight
    gate_w = gate_first.weight
    core_edge = linear(
        edge_feat, core_w[:, :edge_dim], None, tf32=tf32, fast=fast, training=training
    )
    core_target = linear(
        node_feat,
        core_w[:, edge_dim : edge_dim + node_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    core_source = linear(
        node_feat,
        core_w[:, edge_dim + node_dim :],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate_edge = linear(
        edge_feat, gate_w[:, :edge_dim], None, tf32=tf32, fast=fast, training=training
    )
    gate_target = linear(
        node_feat,
        gate_w[:, edge_dim : edge_dim + node_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate_source = linear(
        node_feat,
        gate_w[:, edge_dim + node_dim :],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    core_act, gate_act = AttnProjectFunction.apply(
        core_edge,
        core_target,
        core_source,
        core_first.bias if training else core_first.bias.detach(),
        gate_edge,
        gate_target,
        gate_source,
        gate_first.bias if training else gate_first.bias.detach(),
        source_index,
        target_index,
    )
    core = linear(
        core_act,
        core_second.weight,
        core_second.bias,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate = linear(
        gate_act,
        gate_second.weight,
        gate_second.bias,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    fused_tail = gated_tail(
        core, gate, module.core_norm, module.gate_norm, fast=fast, training=training
    )
    if fused_tail is not None:
        return fused_tail
    return F.silu(module.core_norm(core)) * torch.sigmoid(module.gate_norm(gate))


def refine_line_projection(
    module: GatedMLP,
    edge_feat: Tensor,
    atom_feat: Tensor,
    node_feat: Tensor,
    atom_index: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    *,
    tf32: bool = False,
    fast: bool = False,
    training: bool = True,
) -> Tensor | None:
    if not fast or torch.is_autocast_enabled("cuda"):
        return None

    from .op.gated_mlp import RefineProjectFunction

    if not isinstance(
        getattr(module, "core_norm", None), nn.LayerNorm
    ) or not isinstance(getattr(module, "gate_norm", None), nn.LayerNorm):
        return None
    for mlp in (module.mlp_core, module.mlp_gate):
        if len(mlp.layers) != 4:
            return None
        first, activation, dropout, second = mlp.layers
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            return None
        if not isinstance(dropout, nn.Dropout) or dropout.p != 0.0:
            return None
        if activation.__class__.__name__.lower() not in {"fusedsilu", "silu"}:
            return None
    core_first, _, _, core_second = module.mlp_core.layers
    gate_first, _, _, gate_second = module.mlp_gate.layers
    dim = core_first.out_features
    edge_dim, node_dim = edge_feat.shape[1], node_feat.shape[1]
    atom_dim = atom_feat.shape[1]
    if (
        dim <= 0
        or dim % 32 != 0
        or not edge_feat.is_cuda
        or not atom_feat.is_cuda
        or (not node_feat.is_cuda)
        or (edge_feat.dtype != torch.float32)
        or (atom_feat.dtype != torch.float32)
        or (node_feat.dtype != torch.float32)
        or (core_first.weight.shape != (dim, edge_dim + atom_dim + 2 * node_dim))
        or (gate_first.weight.shape != (dim, edge_dim + atom_dim + 2 * node_dim))
        or (core_second.weight.shape[1] != dim)
        or (gate_second.weight.shape[1] != dim)
        or (core_first.bias is None)
        or (gate_first.bias is None)
        or (atom_index.dtype != torch.int64)
        or (source_index.dtype != torch.int64)
        or (target_index.dtype != torch.int64)
        or (not atom_index.is_cuda)
        or (not source_index.is_cuda)
        or (not target_index.is_cuda)
    ):
        return None
    core_w = core_first.weight
    gate_w = gate_first.weight
    core_hidden = linear(
        edge_feat, core_w[:, :edge_dim], None, tf32=tf32, fast=fast, training=training
    )
    core_atom = linear(
        atom_feat,
        core_w[:, edge_dim : edge_dim + atom_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    core_target = linear(
        node_feat,
        core_w[:, edge_dim + atom_dim : edge_dim + atom_dim + node_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    core_source = linear(
        node_feat,
        core_w[:, edge_dim + atom_dim + node_dim :],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate_hidden = linear(
        edge_feat, gate_w[:, :edge_dim], None, tf32=tf32, fast=fast, training=training
    )
    gate_atom = linear(
        atom_feat,
        gate_w[:, edge_dim : edge_dim + atom_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate_target = linear(
        node_feat,
        gate_w[:, edge_dim + atom_dim : edge_dim + atom_dim + node_dim],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate_source = linear(
        node_feat,
        gate_w[:, edge_dim + atom_dim + node_dim :],
        None,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    core_act, gate_act = RefineProjectFunction.apply(
        core_hidden,
        core_atom,
        core_target,
        core_source,
        core_first.bias if training else core_first.bias.detach(),
        gate_hidden,
        gate_atom,
        gate_target,
        gate_source,
        gate_first.bias if training else gate_first.bias.detach(),
        atom_index,
        source_index,
        target_index,
    )
    core = linear(
        core_act,
        core_second.weight,
        core_second.bias,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    gate = linear(
        gate_act,
        gate_second.weight,
        gate_second.bias,
        tf32=tf32,
        fast=fast,
        training=training,
    )
    fused_tail = gated_tail(
        core, gate, module.core_norm, module.gate_norm, fast=fast, training=training
    )
    if fused_tail is not None:
        return fused_tail
    return F.silu(module.core_norm(core)) * torch.sigmoid(module.gate_norm(gate))


def fused_attention(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    num_segments: int,
    *,
    layout=None,
    fast: bool = False,
) -> tuple[Tensor, Tensor] | None:
    if not fast:
        return None

    from .op.attention import AttentionFunction, segment_layout

    if (
        source_logits.ndim != 2
        or source_logits.shape != target_logits.shape
        or source_logits.shape != values.shape
        or (source_logits.shape[1] <= 0 or source_logits.shape[1] % 32 != 0)
        or (source_logits.dtype != torch.float32)
        or (target_logits.dtype != torch.float32)
        or (values.dtype != torch.float32)
        or (source_index.dtype != torch.int64)
        or (target_index.dtype != torch.int64)
        or (not source_logits.is_cuda)
        or (not target_logits.is_cuda)
        or (not values.is_cuda)
        or (not source_index.is_cuda)
        or (not target_index.is_cuda)
    ):
        return None
    if layout is None:
        layout = (
            segment_layout(source_index, num_segments),
            segment_layout(target_index, num_segments),
        )
    so, to, _, _ = AttentionFunction.apply(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        *layout[0],
        *layout[1],
        torch.is_grad_enabled()
        and any(x.requires_grad for x in (source_logits, target_logits, values)),
    )
    return so, to


def fused_attention_with_residual(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    num_segments: int,
    edge_feat: Tensor,
    edge_res_weight: Tensor,
    *,
    layout=None,
    fast: bool = False,
    training: bool = True,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if not fast:
        return None

    from .op.attention import AttentionFunction, LineResidualFunction, segment_layout

    if (
        source_logits.ndim != 2
        or source_logits.shape != target_logits.shape
        or source_logits.shape != values.shape
        or (source_logits.shape != edge_feat.shape)
        or (source_logits.shape[1] <= 0 or source_logits.shape[1] % 32 != 0)
        or (source_logits.dtype != torch.float32)
        or (target_logits.dtype != torch.float32)
        or (values.dtype != torch.float32)
        or (edge_feat.dtype != torch.float32)
        or (edge_res_weight.dtype != torch.float32)
        or (source_index.dtype != torch.int64)
        or (target_index.dtype != torch.int64)
        or (not source_logits.is_cuda)
        or (not target_logits.is_cuda)
        or (not values.is_cuda)
        or (not edge_feat.is_cuda)
        or (not edge_res_weight.is_cuda)
        or (not source_index.is_cuda)
        or (not target_index.is_cuda)
        or (edge_res_weight.numel() != values.shape[1])
    ):
        return None
    if layout is None:
        layout = (
            segment_layout(source_index, num_segments),
            segment_layout(target_index, num_segments),
        )
    so, to, _, _ = AttentionFunction.apply(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        *layout[0],
        *layout[1],
        torch.is_grad_enabled()
        and any(x.requires_grad for x in (source_logits, target_logits, values)),
    )
    edge = LineResidualFunction.apply(
        values, edge_feat, edge_res_weight if training else edge_res_weight.detach(),
    )
    return so, to, edge
