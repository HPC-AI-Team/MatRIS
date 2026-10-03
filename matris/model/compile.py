"""Compilation precision defaults and staged force/stress training."""
import warnings

import torch
import torch._inductor.config as inductor_config
from torch import nn
from torch._decomp import get_decompositions
from torch.func import functional_call
from torch.fx.experimental import _config as symbolic_config
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.checkpoint import checkpoint


warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled\.",
    category=UserWarning,
    module=r"torch\._inductor\.compile_fx$",
)

# Fixed process-wide compilation defaults.
inductor_config.emulate_precision_casts = True
inductor_config.eager_numerics.division_rounding = True
inductor_config.triton.mix_order_reduction = False
# Avoid the compiler pool's idle/wakeup shutdown race (PyTorch 2.11).
inductor_config.quiesce_async_compile_pool = False
# Equal dimensions or storage offsets need not stay equal on the next call.
symbolic_config.use_duck_shape = False


class _TrainingCore(nn.Module):
    """Trace the tensor core with parameters as explicit graph inputs."""

    def __init__(self, model, task, tf32):
        super().__init__()
        self.model = model
        self.tf32 = tf32
        self.compute_force = "f" in task and model.force_stress_head.is_conservation
        self.edge_output = "f" in task and not model.force_stress_head.is_conservation

    def forward(self, batch_graph):
        node, edge, energy = self.model.forward_core(
            batch_graph, tf32=self.tf32, fast=True, training=True
        )
        geometry = (batch_graph["edge_lengths"], batch_graph["unit_edge_vectors"])
        derivatives = None
        if self.compute_force:
            grads = torch.autograd.grad(
                energy.sum(), geometry, create_graph=True, allow_unused=True
            )
            derivatives = tuple(
                torch.zeros_like(x) if g is None else g for x, g in zip(geometry, grads)
            )
        return (
            node,
            edge if self.edge_output else None,
            derivatives,
        )


def compile_training(model, batch_graph, task, *, tf32=False, activation_checkpoint=False):
    """Stage the geometric VJP before compiling the parameter backward."""
    from functools import partial

    from .op.elementwise import get_extension

    get_extension()
    if tf32:
        from .op.gemm import get_extension as get_gemm_extension

        get_gemm_extension()
    # The tensor core does not use this Python list; its length would specialize
    # the input tree for every batch size produced by load balancing.
    batch_graph = {k: v for k, v in batch_graph.items() if k != "atoms_per_graph"}
    core = _TrainingCore(model, task, tf32)
    state = dict(core.named_parameters()) | dict(core.named_buffers())

    def tensor_core(batch, *values):
        # composition_model aliases the reference-energy module. Swapping its
        # shared attribute twice leaves a FakeTensor behind during restoration.
        return functional_call(
            core, dict(zip(state, values)), (batch,), tie_weights=False
        )

    # Equal sizes in the first batch need not stay equal (e.g. two atoms and
    # the two columns of edge_index). Keep their symbols independent.
    with torch.enable_grad(), symbolic_config.patch(use_duck_shape=False):
        graph = make_fx(
            tensor_core,
            # Native LayerNorm double backward specializes the batch dimension.
            # Decompose both directions so its statistics remain differentiable.
            decomposition_table=get_decompositions(
                [
                    torch.ops.aten.native_layer_norm.default,
                    torch.ops.aten.native_layer_norm_backward.default,
                ]
            ),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(batch_graph, *state.values())
    native = {torch.ops.aten.pow.Tensor_Scalar: torch.ops.matris.pow.default}
    attention_backward = getattr(getattr(torch.ops.matris, "attention_backward", None), "default", None)
    for node in list(graph.graph.nodes):
        node.target = native.get(node.target, node.target)
        # Saved-tensor detaches are tracing artifacts; attention's cached-output
        # boundary is semantic because its analytic second VJP handles that path.
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.detach.default
            and not any(
                user.target == attention_backward
                for user in node.users
            )
        ):
            node.replace_all_uses_with(node.args[0])
            graph.graph.erase_node(node)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    for node in graph.graph.nodes:
        node.meta.clear()
    graph.meta.clear()
    # Capture recomputation inside AOTAutograd, rather than checkpointing an
    # already compiled call from eager backward.
    compiled = torch.compile(
        partial(checkpoint, graph, use_reentrant=False) if activation_checkpoint else graph,
        fullgraph=True,
        dynamic=True,
        options={
            "triton.cudagraphs": False,
        },
    )

    def forward(batch):
        return compiled(
            {k: v for k, v in batch.items() if k != "atoms_per_graph"},
            *core.parameters(),
            *core.buffers(),
        )

    return forward
