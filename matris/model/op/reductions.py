"""CUDA source, lazy inline loading and Python interfaces for reductions."""

from __future__ import annotations

import hashlib
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

from .cuda_common import CUDA_SOURCE as COMMON_SOURCE

_EXTENSION = None


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor directed2undirected_average_forward(
    const torch::Tensor &input, const torch::Tensor &segment, int64_t num_segment);
torch::Tensor directed_average_backward(const torch::Tensor &grad, const torch::Tensor &segment);
torch::Tensor refine_line_smooth_scatter_forward(const torch::Tensor &input,
    const torch::Tensor &smooth, const torch::Tensor &target_index, int64_t num_nodes);
torch::Tensor refine_line_envelope_smooth_scatter_forward(const torch::Tensor &input,
    const torch::Tensor &base_envelope, const torch::Tensor &source_index,
    const torch::Tensor &target_index, int64_t num_nodes);
void envelope_reduce(const at::Tensor &input, const at::Tensor &envelope,
    const at::Tensor &source_index, const at::Tensor &row_ptr, const at::Tensor &edge_order,
    const at::Tensor &output, int64_t channels);
void envelope_derivatives(const at::Tensor &input, const at::Tensor &envelope,
    const at::Tensor &grad, const at::Tensor &source_index, const at::Tensor &target_index,
    const at::Tensor &direction_input, const at::Tensor &direction_envelope,
    const at::Tensor &grad_input, const at::Tensor &grad_envelope, const at::Tensor &grad_grad,
    int64_t num_rows, int64_t channels, bool second_order);

namespace matris_ops {
using Tensor = at::Tensor;
using OptionalTensor = std::optional<Tensor>;

Tensor directed_average(const Tensor &data, const Tensor &segment, c10::SymInt num_segment) {
    return directed2undirected_average_forward(
        data.contiguous(), segment.contiguous(), num_segment.expect_int());
}

Tensor directed_average_meta(const Tensor &data, const Tensor &segment, c10::SymInt num_segment) {
    return at::empty_symint({num_segment, data.sym_size(1)}, data.options());
}

Tensor directed_average_backward_meta(const Tensor &grad, const Tensor &segment) {
    return at::empty_symint({segment.sym_size(0), grad.sym_size(1)}, grad.options());
}

Tensor smooth_scatter(
    const Tensor &input, const Tensor &smooth, const Tensor &target_index, c10::SymInt num_nodes) {
    return refine_line_smooth_scatter_forward(
        input.contiguous(), smooth.contiguous(), target_index.contiguous(), num_nodes.expect_int());
}

Tensor smooth_scatter_meta(
    const Tensor &input, const Tensor &smooth, const Tensor &target_index, c10::SymInt num_nodes) {
    return at::empty_symint({num_nodes, input.sym_size(1)}, input.options());
}

std::tuple<Tensor, Tensor> envelope_scatter_backward(const Tensor &grad, const Tensor &input,
    const Tensor &base, const Tensor &source, const Tensor &target) {
    auto dx = at::empty_like(input, input.options(), at::MemoryFormat::Contiguous);
    auto db = at::zeros_like(base, base.options(), at::MemoryFormat::Contiguous);
    envelope_derivatives(input.contiguous(), base.contiguous(), grad.contiguous(), source, target,
        input, base, dx, db, grad, input.size(0), input.size(1), false);
    return {dx, db};
}

std::tuple<Tensor, Tensor> envelope_scatter_backward_meta(const Tensor &grad, const Tensor &input,
    const Tensor &base, const Tensor &source, const Tensor &target) {
    return {at::empty_symint(input.sym_sizes(), input.options()),
        at::empty_symint(base.sym_sizes(), base.options())};
}

std::tuple<Tensor, Tensor, Tensor> envelope_scatter_double_backward(const Tensor &grad,
    const Tensor &input, const Tensor &base, const Tensor &source, const Tensor &target,
    const Tensor &hx, const Tensor &hb) {
    auto dx = at::empty_like(input, input.options(), at::MemoryFormat::Contiguous);
    auto db = at::zeros_like(base, base.options(), at::MemoryFormat::Contiguous);
    auto dg = at::zeros_like(grad, grad.options(), at::MemoryFormat::Contiguous);
    envelope_derivatives(input.contiguous(), base.contiguous(), grad.contiguous(), source, target,
        hx.contiguous(), hb.contiguous(), dx, db, dg, input.size(0), input.size(1), true);
    return {dg, dx, db};
}

std::tuple<Tensor, Tensor, Tensor> envelope_scatter_double_backward_meta(const Tensor &grad,
    const Tensor &input, const Tensor &base, const Tensor &source, const Tensor &target,
    const Tensor &hx, const Tensor &hb) {
    return {at::empty_symint(grad.sym_sizes(), grad.options()),
        at::empty_symint(input.sym_sizes(), input.options()),
        at::empty_symint(base.sym_sizes(), base.options())};
}

Tensor envelope_scatter(const Tensor &input, const Tensor &base_envelope,
    const Tensor &source_index, const Tensor &target_index, c10::SymInt num_nodes,
    const OptionalTensor &ptr, const OptionalTensor &order) {
    auto source = source_index.contiguous(), target = target_index.contiguous();
    if (!ptr)
        return refine_line_envelope_smooth_scatter_forward(
            input.contiguous(), base_envelope.contiguous(), source, target, num_nodes.expect_int());
    auto output = at::empty_symint({num_nodes, input.sym_size(1)}, input.options());
    envelope_reduce(input.contiguous(), base_envelope.contiguous(), source, *ptr, *order, output,
        input.size(1));
    return output;
}

Tensor envelope_scatter_meta(const Tensor &input, const Tensor &base_envelope,
    const Tensor &source_index, const Tensor &target_index, c10::SymInt num_nodes,
    const OptionalTensor &ptr, const OptionalTensor &order) {
    return at::empty_symint({num_nodes, input.sym_size(1)}, input.options());
}
} // namespace matris_ops

TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def("directed_average(Tensor data, Tensor segment, SymInt num_segment) -> Tensor");
    m.def("directed_average_backward(Tensor grad, Tensor segment) -> Tensor");
    m.def("smooth_scatter(Tensor input, Tensor smooth, Tensor target_index, SymInt num_nodes) -> "
          "Tensor");
    m.def("envelope_scatter_backward(Tensor grad, Tensor input, Tensor base, Tensor source, Tensor "
          "target) -> (Tensor, Tensor)");
    m.def("envelope_scatter_double_backward(Tensor grad, Tensor input, Tensor base, Tensor source, "
          "Tensor target, Tensor hx, Tensor hb) -> (Tensor, Tensor, Tensor)");
    m.def("envelope_scatter(Tensor input, Tensor base_envelope, Tensor source_index, Tensor "
          "target_index, SymInt num_nodes, Tensor? ptr, Tensor? order) -> Tensor");
}

TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("directed_average", TORCH_FN(matris_ops::directed_average));
    m.impl("directed_average_backward", TORCH_FN(directed_average_backward));
    m.impl("smooth_scatter", TORCH_FN(matris_ops::smooth_scatter));
    m.impl("envelope_scatter_backward", TORCH_FN(matris_ops::envelope_scatter_backward));
    m.impl(
        "envelope_scatter_double_backward", TORCH_FN(matris_ops::envelope_scatter_double_backward));
    m.impl("envelope_scatter", TORCH_FN(matris_ops::envelope_scatter));
}

TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("directed_average", TORCH_FN(matris_ops::directed_average_meta));
    m.impl("directed_average_backward", TORCH_FN(matris_ops::directed_average_backward_meta));
    m.impl("smooth_scatter", TORCH_FN(matris_ops::smooth_scatter_meta));
    m.impl("envelope_scatter_backward", TORCH_FN(matris_ops::envelope_scatter_backward_meta));
    m.impl("envelope_scatter_double_backward",
        TORCH_FN(matris_ops::envelope_scatter_double_backward_meta));
    m.impl("envelope_scatter", TORCH_FN(matris_ops::envelope_scatter_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto register_gradient = py::module_::import("torch.library").attr("register_autograd");
        auto reductions = py::module_::import("matris.model.op.reductions");
        register_gradient("matris::directed_average",
            reductions.attr("DirectedAverageFunction").attr("backward"),
            py::arg("setup_context") =
                reductions.attr("DirectedAverageFunction").attr("setup_context"));
        register_gradient("matris::directed_average_backward",
            reductions.attr("DirectedAverageBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                reductions.attr("DirectedAverageBackwardFunction").attr("setup_context"));
        register_gradient("matris::smooth_scatter",
            reductions.attr("SmoothScatterFunction").attr("backward"),
            py::arg("setup_context") =
                reductions.attr("SmoothScatterFunction").attr("setup_context"));
        register_gradient("matris::envelope_scatter_backward",
            reductions.attr("EnvelopeScatterBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                reductions.attr("EnvelopeScatterBackwardFunction").attr("setup_context"));
        register_gradient("matris::envelope_scatter",
            reductions.attr("EnvelopeScatterFunction").attr("backward"),
            py::arg("setup_context") =
                reductions.attr("EnvelopeScatterFunction").attr("setup_context"));
    });
    m.def("directed2undirected_average_forward", &directed2undirected_average_forward,
        "directed2undirected_average_forward");
    m.def("refine_line_smooth_scatter_forward", &refine_line_smooth_scatter_forward,
        "refine_line_smooth_scatter_forward");
    m.def("envelope_reduce", &envelope_reduce);
    m.def("envelope_derivatives", &envelope_derivatives);
}
"""


CUDA_SOURCE = COMMON_SOURCE + r"""

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockThreads = 256;

__global__ void directed_average_backward_kernel(const float *grad, const int64_t *segment,
    float *output, int64_t total, int64_t dim, int64_t row_stride, int64_t col_stride,
    int64_t segment_stride) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total)
        output[i] = grad[segment[(i / dim) * segment_stride] * row_stride
                         + (i % dim) * col_stride] * 0.5f;
}

__global__ void directed2undirected_average_forward_kernel(const float *__restrict__ input,
                                                           const int64_t *__restrict__ segment,
                                                           float *__restrict__ output, int rows,
                                                           int kFeatureDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * kFeatureDim;
    if (idx >= total) {
        return;
    }
    int row = idx / kFeatureDim;
    int dim = idx - row * kFeatureDim;
    int64_t out_row = segment[row];
    atomicAdd(output + out_row * kFeatureDim + dim, input[idx] * 0.5f);
}

} // namespace

at::Tensor directed_average_backward(const at::Tensor &grad, const at::Tensor &segment) {
    const c10::cuda::CUDAGuard guard(grad.device());
    auto output = at::empty({segment.size(0), grad.size(1)}, grad.options());
    int64_t total = output.numel();
    if (total) {
        directed_average_backward_kernel<<<(total + kBlockThreads - 1) / kBlockThreads,
            kBlockThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad.data_ptr<float>(), segment.data_ptr<int64_t>(), output.data_ptr<float>(),
            total, grad.size(1), grad.stride(0), grad.stride(1), segment.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

at::Tensor directed2undirected_average_forward(const at::Tensor &input, const at::Tensor &segment,
                                               int64_t num_segment) {
    const int kFeatureDim = input.size(1);
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(segment.is_cuda(), "segment must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(segment.scalar_type() == at::kLong, "segment must be int64");
    TORCH_CHECK(input.dim() == 2 && kFeatureDim > 0 && kFeatureDim % 32 == 0,
                "input must have shape [N, D], D a positive multiple of 32");
    TORCH_CHECK(segment.dim() == 1 && segment.size(0) == input.size(0),
                "segment must have shape [N]");

    auto output = at::zeros({num_segment, kFeatureDim}, input.options());
    int rows = static_cast<int>(input.size(0));
    int total = rows * kFeatureDim;
    int blocks = (total + kBlockThreads - 1) / kBlockThreads;
    directed2undirected_average_forward_kernel<<<blocks, kBlockThreads, 0,
                                                 c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), segment.data_ptr<int64_t>(), output.data_ptr<float>(), rows,
        kFeatureDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}



using namespace matris_cuda;

namespace {
__global__ void refine_line_smooth_scatter_forward_kernel(const float *__restrict__ input,
    const float *__restrict__ smooth, const int64_t *__restrict__ target_index,
    float *__restrict__ out, int64_t rows, int kDim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t row = idx / kDim;
    int dim = static_cast<int>(idx - row * kDim);
    int64_t target = target_index[row];
    atomicAdd(out + target * kDim + dim, input[idx] * smooth[idx]);
}

__global__ void refine_line_envelope_smooth_scatter_forward_kernel(const float *__restrict__ input,
    const float *__restrict__ base_envelope, const int64_t *__restrict__ source_index,
    const int64_t *__restrict__ target_index, float *__restrict__ out, int64_t rows, int kDim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t row = idx / kDim;
    int dim = static_cast<int>(idx - row * kDim);
    int64_t source = source_index[row];
    int64_t target = target_index[row];
    float smooth = base_envelope[source * kDim + dim] * base_envelope[target * kDim + dim];
    atomicAdd(out + target * kDim + dim, input[idx] * smooth);
}

template <typename I>
__global__ void envelope_reduce_kernel(const float *__restrict__ input,
    const float *__restrict__ envelope, const I *source_index, const int32_t *row_ptr,
    const int32_t *edge_order, float *output, int channels) {
    int node = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= channels)
        return;
    float sum = 0.f, target_envelope = envelope[(int64_t)node * channels + col];
    for (int64_t i = row_ptr[node]; i < row_ptr[node + 1]; ++i) {
        int64_t edge = edge_order[i];
        sum += input[edge * channels + col] *
               (envelope[int64_t(source_index[edge]) * channels + col] * target_envelope);
    }
    output[(int64_t)node * channels + col] = sum;
}

__global__ void envelope_derivative_kernel(const float *__restrict__ input,
    const float *__restrict__ envelope, const float *__restrict__ grad, Index source_index,
    Index target_index, const float *__restrict__ direction_input,
    const float *__restrict__ direction_envelope, float *grad_input, float *grad_envelope,
    float *grad_grad, int64_t num_rows, int channels, bool second_order) {
    int64_t offset = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= num_rows * channels)
        return;
    int64_t source_offset = source_index[offset / channels] * channels + offset % channels,
            target_offset = target_index[offset / channels] * channels + offset % channels;
    float source_envelope = envelope[source_offset], target_envelope = envelope[target_offset],
          value = input[offset], upstream = grad[target_offset], grad_source, grad_target;
    if (second_order) {
        float input_direction = direction_input[offset],
              source_direction = direction_envelope[source_offset],
              target_direction = direction_envelope[target_offset];
        grad_input[offset] =
            upstream * (source_direction * target_envelope + source_envelope * target_direction);
        grad_source = upstream * (input_direction * target_envelope + value * target_direction);
        grad_target = upstream * (input_direction * source_envelope + value * source_direction);
        atomicAdd(grad_grad + target_offset,
            input_direction * (source_envelope * target_envelope) +
                value * (source_direction * target_envelope + source_envelope * target_direction));
    } else {
        grad_input[offset] = upstream * (source_envelope * target_envelope);
        grad_source = upstream * value * target_envelope;
        grad_target = upstream * value * source_envelope;
    }
    atomicAdd(grad_envelope + source_offset, grad_source);
    atomicAdd(grad_envelope + target_offset, grad_target);
}

} // namespace

void envelope_reduce(const at::Tensor &input, const at::Tensor &envelope,
    const at::Tensor &source_index, const at::Tensor &row_ptr, const at::Tensor &edge_order,
    const at::Tensor &output, int64_t channels) {
    const c10::cuda::CUDAGuard guard(input.device());
#define LAUNCH_ENVELOPE_REDUCE(I)                                                                  \
    envelope_reduce_kernel<I><<<dim3(row_ptr.numel() - 1, (channels + 127) / 128), 128, 0,         \
        at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), envelope.data_ptr<float>(),   \
        source_index.data_ptr<I>(), row_ptr.data_ptr<int32_t>(), edge_order.data_ptr<int32_t>(),   \
        output.data_ptr<float>(), channels)
    if (row_ptr.numel() > 1) {
        if (source_index.scalar_type() == at::kLong) {
            LAUNCH_ENVELOPE_REDUCE(int64_t);
        } else {
            LAUNCH_ENVELOPE_REDUCE(int32_t);
        }
    }
#undef LAUNCH_ENVELOPE_REDUCE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void envelope_derivatives(const at::Tensor &input, const at::Tensor &envelope,
    const at::Tensor &grad, const at::Tensor &source_index, const at::Tensor &target_index,
    const at::Tensor &direction_input, const at::Tensor &direction_envelope,
    const at::Tensor &grad_input, const at::Tensor &grad_envelope, const at::Tensor &grad_grad,
    int64_t num_rows, int64_t channels, bool second_order) {
    const c10::cuda::CUDAGuard guard(input.device());
    if (num_rows)
        envelope_derivative_kernel<<<(num_rows * channels + 255) / 256, 256, 0,
            at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), envelope.data_ptr<float>(),
            grad.data_ptr<float>(), index_of(source_index), index_of(target_index),
            direction_input.data_ptr<float>(), direction_envelope.data_ptr<float>(),
            grad_input.data_ptr<float>(), grad_envelope.data_ptr<float>(),
            grad_grad.data_ptr<float>(), num_rows, channels, second_order);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor refine_line_smooth_scatter_forward(const at::Tensor &input, const at::Tensor &smooth,
    const at::Tensor &target_index, int64_t num_nodes) {
    const c10::cuda::CUDAGuard guard(input.device());
    const int kDim = input.size(1);

    auto out = at::zeros({num_nodes, kDim}, input.options());
    int64_t total = input.size(0) * kDim;
    if (!total)
        return out;
    int blocks = static_cast<int>((total + 256 - 1) / 256);
    refine_line_smooth_scatter_forward_kernel<<<blocks, 256, 0,
        c10::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), smooth.data_ptr<float>(),
        target_index.data_ptr<int64_t>(), out.data_ptr<float>(), input.size(0), kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor refine_line_envelope_smooth_scatter_forward(const at::Tensor &input,
    const at::Tensor &base_envelope, const at::Tensor &source_index, const at::Tensor &target_index,
    int64_t num_nodes) {
    const c10::cuda::CUDAGuard guard(input.device());
    const int kDim = input.size(1);

    auto out = at::zeros({num_nodes, kDim}, input.options());
    int64_t total = input.size(0) * kDim;
    if (!total)
        return out;
    int blocks = static_cast<int>((total + 256 - 1) / 256);
    refine_line_envelope_smooth_scatter_forward_kernel<<<blocks, 256, 0,
        c10::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(),
        base_envelope.data_ptr<float>(), source_index.data_ptr<int64_t>(),
        target_index.data_ptr<int64_t>(), out.data_ptr<float>(), input.size(0), kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


@torch.compiler.assume_constant_result
def get_extension():
    """Compile this operator family once and reuse its source-addressed cache."""
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("MAX_JOBS", "4")
        key = hashlib.sha256(
            (
                f"{sys.implementation.cache_tag}:{torch.__version__}:{torch.version.cuda}\n"
                + CPP_SOURCE
                + "\n"
                + CUDA_SOURCE
            ).encode()
        ).hexdigest()[:16]
        extension = load_inline(
            name=f"matris_reductions_{key}",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--extended-lambda"],
            verbose=False,
        )
        extension.register_autograd()
        _EXTENSION = extension
    return _EXTENSION


class DirectedAverageBackwardFunction(torch.autograd.Function):
    """Keep gather after scatter in a separate kernel, including under compile."""

    @staticmethod
    def forward(grad, segment):
        get_extension()
        return torch.ops.matris.directed_average_backward(grad, segment)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])
        ctx.num_segment = inputs[0].shape[0]

    @staticmethod
    def backward(ctx, grad):
        (segment,) = ctx.saved_tensors
        return DirectedAverageFunction.apply(grad, segment, ctx.num_segment), None


class DirectedAverageFunction(torch.autograd.Function):
    @staticmethod
    def forward(data, segment, num_segment):
        get_extension()
        return torch.ops.matris.directed_average(data, segment, num_segment)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])

    @staticmethod
    def backward(ctx, grad):
        (segment,) = ctx.saved_tensors
        return DirectedAverageBackwardFunction.apply(grad, segment), None, None


class SmoothScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, smooth, target_index, num_nodes):
        get_extension()
        return torch.ops.matris.smooth_scatter(input, smooth, target_index, num_nodes)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:3])

    @staticmethod
    def backward(ctx, grad):
        input, smooth, target = ctx.saved_tensors
        gathered = grad[target]
        return (gathered * smooth, gathered * input, None, None)


class EnvelopeScatterDoubleBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, input, base, source, target, hx, hb):
        get_extension()
        return torch.ops.matris.envelope_scatter_double_backward(
            grad, input, base, source, target, hx, hb
        )


class EnvelopeScatterBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(grad, input, base, source, target):
        get_extension()
        return torch.ops.matris.envelope_scatter_backward(
            grad, input, base, source, target
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, hx, hb):
        return (
            *EnvelopeScatterDoubleBackwardFunction.apply(*ctx.saved_tensors, hx, hb),
            None,
            None,
        )


class EnvelopeScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        input, base_envelope, source_index, target_index, num_nodes, ptr, order
    ):
        get_extension()
        return torch.ops.matris.envelope_scatter(
            input, base_envelope, source_index, target_index, num_nodes, ptr, order
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:4])

    @staticmethod
    def backward(ctx, grad):
        return (
            *EnvelopeScatterBackwardFunction.apply(grad, *ctx.saved_tensors),
            None,
            None,
            None,
            None,
            None,
        )


directed_average_op = DirectedAverageFunction.apply

smooth_scatter_op = SmoothScatterFunction.apply

envelope_scatter_op = EnvelopeScatterFunction.apply
