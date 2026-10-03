"""CUDA source, lazy inline loading and Python interfaces for attention."""

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

std::vector<torch::Tensor> fused_line_attention_backward(const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out, const torch::Tensor &values,
    const torch::Tensor &source_out, const torch::Tensor &target_out,
    const torch::Tensor &source_alpha, const torch::Tensor &target_alpha,
    const torch::Tensor &source_index, const torch::Tensor &target_index);
torch::Tensor line_edge_residual_forward(const torch::Tensor &values,
    const torch::Tensor &edge_feat, const torch::Tensor &edge_res_weight);
void online_tile(const at::Tensor &logits, const at::Tensor &values, const at::Tensor &row_ptr,
    const at::Tensor &edge_order, const at::Tensor &probabilities, const at::Tensor &output,
    int64_t channels, bool with_values, bool save_probabilities);
void softmax_backward(const at::Tensor &grad, const at::Tensor &probabilities,
    const at::Tensor &row_ptr, const at::Tensor &edge_order, const at::Tensor &grad_logits,
    int64_t channels);
void attention_double_backward(const at::Tensor &grad, const at::Tensor &values,
    const at::Tensor &output, const at::Tensor &probabilities, const at::Tensor &direction_logits,
    const at::Tensor &direction_values, const at::Tensor &row_ptr, const at::Tensor &edge_order,
    const at::Tensor &grad_logits, const at::Tensor &grad_values, const at::Tensor &grad_grad,
    int64_t channels, bool accumulate_values);

namespace matris_ops {
using Tensor = at::Tensor;
using OptionalTensor = std::optional<Tensor>;

Tensor softmax(const Tensor &x, const Tensor &index, const Tensor &ptr, const Tensor &order) {
    auto input = x.contiguous();
    auto output = at::empty_like(input);
    if (input.numel())
        online_tile(input, input, ptr, order, output, output, input.size(1), false, true);
    return output;
}

Tensor softmax_meta(const Tensor &x, const Tensor &index, const Tensor &ptr, const Tensor &order) {
    return at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
}

Tensor softmax_backward(const Tensor &grad, const Tensor &y, const Tensor &index, const Tensor &ptr,
    const Tensor &order) {
    auto dx = at::empty_like(y);
    if (y.numel())
        ::softmax_backward(grad.contiguous(), y, ptr, order, dx, y.size(1));
    return dx;
}

Tensor softmax_backward_meta(const Tensor &grad, const Tensor &y, const Tensor &index,
    const Tensor &ptr, const Tensor &order) {
    return at::empty_like(y);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> attention_double_backward(const Tensor &gs,
    const Tensor &gt, const Tensor &values, const Tensor &source_out, const Tensor &target_out,
    const Tensor &source_alpha, const Tensor &target_alpha, const Tensor &sptr,
    const Tensor &sorder, const Tensor &tptr, const Tensor &torder, const Tensor &hs,
    const Tensor &ht, const Tensor &hv) {
    auto dsource = at::empty_like(values, values.options(), at::MemoryFormat::Contiguous);
    auto dtarget = at::empty_like(values, values.options(), at::MemoryFormat::Contiguous);
    auto dvalues = at::empty_like(values, values.options(), at::MemoryFormat::Contiguous);
    auto dgs = at::empty_like(gs), dgt = at::empty_like(gt);
    auto input = values.contiguous(), direction = hv.contiguous();
    ::attention_double_backward(gs.contiguous(), input, source_out, source_alpha, hs.contiguous(),
        direction, sptr, sorder, dsource, dvalues, dgs, values.size(1), false);
    ::attention_double_backward(gt.contiguous(), input, target_out, target_alpha, ht.contiguous(),
        direction, tptr, torder, dtarget, dvalues, dgt, values.size(1), true);
    return {dgs, dgt, dsource, dtarget, dvalues};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> attention_double_backward_meta(const Tensor &gs,
    const Tensor &gt, const Tensor &values, const Tensor &source_out, const Tensor &target_out,
    const Tensor &source_alpha, const Tensor &target_alpha, const Tensor &sptr,
    const Tensor &sorder, const Tensor &tptr, const Tensor &torder, const Tensor &hs,
    const Tensor &ht, const Tensor &hv) {
    return {at::empty_symint(gs.sym_sizes(), gs.options()),
        at::empty_symint(gt.sym_sizes(), gt.options()),
        at::empty_symint(values.sym_sizes(), values.options()),
        at::empty_symint(values.sym_sizes(), values.options()),
        at::empty_symint(values.sym_sizes(), values.options())};
}

std::tuple<Tensor, Tensor, Tensor> attention_backward(const Tensor &gs, const Tensor &gt,
    const Tensor &source_logits, const Tensor &target_logits, const Tensor &values,
    const Tensor &source_out, const Tensor &target_out, const Tensor &source_alpha,
    const Tensor &target_alpha, const Tensor &source_index, const Tensor &target_index,
    const Tensor &sptr, const Tensor &sorder, const Tensor &tptr, const Tensor &torder) {
    auto grads =
        fused_line_attention_backward(gs.contiguous(), gt.contiguous(), values.contiguous(),
            source_out, target_out, source_alpha, target_alpha, source_index, target_index);
    return {grads[0], grads[1], grads[2]};
}

std::tuple<Tensor, Tensor, Tensor> attention_backward_meta(const Tensor &gs, const Tensor &gt,
    const Tensor &source_logits, const Tensor &target_logits, const Tensor &values,
    const Tensor &source_out, const Tensor &target_out, const Tensor &source_alpha,
    const Tensor &target_alpha, const Tensor &source_index, const Tensor &target_index,
    const Tensor &sptr, const Tensor &sorder, const Tensor &tptr, const Tensor &torder) {
    return {at::empty_symint(source_logits.sym_sizes(), source_logits.options()),
        at::empty_symint(target_logits.sym_sizes(), target_logits.options()),
        at::empty_symint(values.sym_sizes(), values.options())};
}

std::tuple<Tensor, Tensor, OptionalTensor, OptionalTensor> attention(const Tensor &source_logits,
    const Tensor &target_logits, const Tensor &values, const Tensor &source_index,
    const Tensor &target_index, const Tensor &sptr, const Tensor &sorder, const Tensor &tptr,
    const Tensor &torder, bool save_alpha) {
    auto source = source_logits.contiguous(), target = target_logits.contiguous();
    auto input = values.contiguous();
    auto so = at::empty({sptr.numel() - 1, values.size(1)}, source.options());
    auto to = at::empty({tptr.numel() - 1, values.size(1)}, target.options());
    OptionalTensor sa = save_alpha ? OptionalTensor(at::empty_like(source)) : std::nullopt;
    OptionalTensor ta = save_alpha ? OptionalTensor(at::empty_like(target)) : std::nullopt;
    if (sptr.numel() > 1)
        online_tile(source, input, sptr, sorder, save_alpha ? *sa : so, so, source.size(1), true,
            save_alpha);
    if (tptr.numel() > 1)
        online_tile(target, input, tptr, torder, save_alpha ? *ta : to, to, target.size(1), true,
            save_alpha);
    return {so, to, sa, ta};
}

std::tuple<Tensor, Tensor, OptionalTensor, OptionalTensor> attention_meta(
    const Tensor &source_logits, const Tensor &target_logits, const Tensor &values,
    const Tensor &source_index, const Tensor &target_index, const Tensor &sptr,
    const Tensor &sorder, const Tensor &tptr, const Tensor &torder, bool save_alpha) {
    return {at::empty_symint({sptr.sym_numel() - 1, values.sym_size(1)}, values.options()),
        at::empty_symint({tptr.sym_numel() - 1, values.sym_size(1)}, values.options()),
        save_alpha ? OptionalTensor(at::empty_symint(values.sym_sizes(), values.options()))
                   : std::nullopt,
        save_alpha ? OptionalTensor(at::empty_symint(values.sym_sizes(), values.options()))
                   : std::nullopt};
}

Tensor line_residual(const Tensor &values, const Tensor &edge_feat, const Tensor &weight) {
    return line_edge_residual_forward(
        values.contiguous(), edge_feat.contiguous(), weight.contiguous());
}

Tensor line_residual_meta(const Tensor &values, const Tensor &edge_feat, const Tensor &weight) {
    return at::empty_symint(values.sym_sizes(), values.options());
}
} // namespace matris_ops

TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def("softmax(Tensor x, Tensor index, Tensor ptr, Tensor order) -> Tensor");
    m.def("softmax_backward(Tensor grad, Tensor y, Tensor index, Tensor ptr, Tensor order) -> "
          "Tensor");
    m.def("attention_double_backward(Tensor gs, Tensor gt, Tensor values, Tensor source_out, "
          "Tensor target_out, Tensor source_alpha, Tensor target_alpha, Tensor sptr, Tensor "
          "sorder, Tensor tptr, Tensor torder, Tensor hs, Tensor ht, Tensor hv) -> (Tensor, "
          "Tensor, Tensor, Tensor, Tensor)");
    m.def("attention_backward(Tensor gs, Tensor gt, Tensor source_logits, Tensor target_logits, "
          "Tensor values, Tensor source_out, Tensor target_out, Tensor source_alpha, Tensor "
          "target_alpha, Tensor source_index, Tensor target_index, Tensor sptr, Tensor sorder, "
          "Tensor tptr, Tensor torder) -> (Tensor, Tensor, Tensor)");
    m.def("attention(Tensor source_logits, Tensor target_logits, Tensor values, Tensor "
          "source_index, Tensor target_index, Tensor sptr, Tensor sorder, Tensor tptr, Tensor "
          "torder, bool save_alpha) -> (Tensor, Tensor, Tensor?, Tensor?)");
    m.def("line_residual(Tensor values, Tensor edge_feat, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("softmax", TORCH_FN(matris_ops::softmax));
    m.impl("softmax_backward", TORCH_FN(matris_ops::softmax_backward));
    m.impl("attention_double_backward", TORCH_FN(matris_ops::attention_double_backward));
    m.impl("attention_backward", TORCH_FN(matris_ops::attention_backward));
    m.impl("attention", TORCH_FN(matris_ops::attention));
    m.impl("line_residual", TORCH_FN(matris_ops::line_residual));
}

TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("softmax", TORCH_FN(matris_ops::softmax_meta));
    m.impl("softmax_backward", TORCH_FN(matris_ops::softmax_backward_meta));
    m.impl("attention_double_backward", TORCH_FN(matris_ops::attention_double_backward_meta));
    m.impl("attention_backward", TORCH_FN(matris_ops::attention_backward_meta));
    m.impl("attention", TORCH_FN(matris_ops::attention_meta));
    m.impl("line_residual", TORCH_FN(matris_ops::line_residual_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto register_gradient = py::module_::import("torch.library").attr("register_autograd");
        auto online_softmax = py::module_::import("matris.model.op.attention");
        auto attention = py::module_::import("matris.model.op.attention");
        register_gradient("matris::softmax",
            online_softmax.attr("SoftmaxFunction").attr("backward"),
            py::arg("setup_context") =
                online_softmax.attr("SoftmaxFunction").attr("setup_context"));
        register_gradient("matris::softmax_backward",
            online_softmax.attr("SoftmaxBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                online_softmax.attr("SoftmaxBackwardFunction").attr("setup_context"));
        register_gradient("matris::attention_backward",
            attention.attr("AttentionBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                attention.attr("AttentionBackwardFunction").attr("setup_context"));
        register_gradient("matris::attention", attention.attr("AttentionFunction").attr("backward"),
            py::arg("setup_context") = attention.attr("AttentionFunction").attr("setup_context"));
        register_gradient("matris::line_residual",
            attention.attr("LineResidualFunction").attr("backward"),
            py::arg("setup_context") =
                attention.attr("LineResidualFunction").attr("setup_context"));
    });
    m.def("fused_line_attention_backward", &fused_line_attention_backward,
        "fused_line_attention_backward");
    m.def("line_edge_residual_forward", &line_edge_residual_forward, "line_edge_residual_forward");
    m.def("online_tile", &online_tile);
    m.def("softmax_backward", &softmax_backward);
    m.def("attention_double_backward", &attention_double_backward);
}
"""


CUDA_SOURCE = COMMON_SOURCE + r"""


using namespace matris_cuda;

namespace {
// Each warp reduces eight edges for sixteen contiguous features.
template <bool attention, bool save>
__global__ void online_kernel(const float *__restrict__ logits, const float *__restrict__ values,
    const int32_t *row_ptr, const int32_t *edge_order, float *probabilities, float *output,
    int channels, int segments) {
    int segment = blockIdx.x, lane = threadIdx.x & 31, edge_lane = lane / 4;
    int col = blockIdx.y * 64 + (threadIdx.x / 32) * 16 + (lane % 4) * 4;
    int begin = row_ptr[segment], end = row_ptr[segment + 1];
    float maximum[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX}, denominator[4] = {},
          numerator[4] = {};
    alignas(16) float weights[2][4];
    for (int start = begin; start < end; start += 16) {
        alignas(16) float tile[2][4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            int i = start + edge_lane + 8 * k;
#pragma unroll
            for (int c = 0; c < 4; ++c)
                tile[k][c] = -FLT_MAX;
            if (i < end) {
                int64_t offset = int64_t(edge_order[i]) * channels + col;
                if (channels % 4 == 0) {
                    if (col < channels)
                        *reinterpret_cast<float4 *>(tile[k]) =
                            *reinterpret_cast<const float4 *>(logits + offset);
                } else {
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        if (col + c < channels)
                            tile[k][c] = logits[offset + c];
                }
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float m = fmaxf(tile[0][c], tile[1][c]);
#pragma unroll
            for (int delta = 4; delta < 32; delta *= 2)
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, delta));
            float next = fmaxf(maximum[c], m);
            if (start != begin) {
                float correction = __expf(maximum[c] - next);
                denominator[c] *= correction;
                numerator[c] *= correction;
            }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                weights[k][c] = start + edge_lane + 8 * k < end && col + c < channels
                                    ? __expf(tile[k][c] - next)
                                    : 0.f;
                denominator[c] += weights[k][c];
            }
            maximum[c] = next;
        }
        if constexpr (attention) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                int i = start + edge_lane + 8 * k;
                alignas(16) float value[4] = {};
                if (i < end) {
                    int64_t offset = int64_t(edge_order[i]) * channels + col;
                    if (channels % 4 == 0) {
                        if (col < channels)
                            *reinterpret_cast<float4 *>(value) =
                                *reinterpret_cast<const float4 *>(values + offset);
                    } else {
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                            if (col + c < channels)
                                value[c] = values[offset + c];
                    }
                }
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    numerator[c] += weights[k][c] * value[c];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int delta = 4; delta < 32; delta *= 2) {
            denominator[c] += __shfl_xor_sync(0xffffffff, denominator[c], delta);
            numerator[c] += __shfl_xor_sync(0xffffffff, numerator[c], delta);
        }
        denominator[c] = begin == end ? 0.f : 1.f / denominator[c];
        if constexpr (attention)
            if (edge_lane == 0 && col + c < channels)
                output[int64_t(segment) * channels + col + c] = numerator[c] * denominator[c];
    }
    // A single tile reuses its weights; longer segments recompute without scratch.
    if constexpr (save) {
        for (int start = begin; start < end; start += 16) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                int i = start + edge_lane + 8 * k;
                if (i < end) {
                    int64_t offset = int64_t(edge_order[i]) * channels + col;
                    alignas(16) float result[4] = {};
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        if (col + c < channels)
                            result[c] =
                                (end - begin <= 16 ? weights[k][c]
                                                   : __expf(logits[offset + c] - maximum[c])) *
                                denominator[c];
                    if (channels % 4 == 0) {
                        if (col < channels)
                            *reinterpret_cast<float4 *>(probabilities + offset) =
                                *reinterpret_cast<float4 *>(result);
                    } else {
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                            if (col + c < channels)
                                probabilities[offset + c] = result[c];
                    }
                }
            }
        }
    }
}

__global__ void softmax_backward_kernel(const float *__restrict__ grad,
    const float *__restrict__ probabilities, const int32_t *row_ptr, const int32_t *edge_order,
    float *grad_logits, int channels, int segments) {
    int segment = blockIdx.x, lane = threadIdx.x & 31, edge_lane = lane / 4;
    int col = blockIdx.y * 64 + (threadIdx.x / 32) * 16 + (lane % 4) * 4;
    float dot[4] = {};
    for (int i = row_ptr[segment] + edge_lane; i < row_ptr[segment + 1]; i += 8) {
        int64_t offset = int64_t(edge_order[i]) * channels + col;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (col + c < channels)
                dot[c] += probabilities[offset + c] * grad[offset + c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int delta = 4; delta < 32; delta *= 2)
            dot[c] += __shfl_xor_sync(0xffffffff, dot[c], delta);
    }
    for (int i = row_ptr[segment] + edge_lane; i < row_ptr[segment + 1]; i += 8) {
        int64_t offset = int64_t(edge_order[i]) * channels + col;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (col + c < channels)
                grad_logits[offset + c] = probabilities[offset + c] * (grad[offset + c] - dot[c]);
    }
}

__global__ void attention_second_kernel(const float *__restrict__ grad_output,
    const float *__restrict__ values, const float *__restrict__ output,
    const float *__restrict__ probabilities, const float *__restrict__ direction_logits,
    const float *__restrict__ direction_values, const int32_t *row_ptr, const int32_t *edge_order,
    float *grad_logits, float *grad_values, float *grad_grad, int channels, int segments,
    bool accumulate_values) {
    int segment = blockIdx.x, lane = threadIdx.x & 31, edge_lane = threadIdx.x / 16;
    int col = blockIdx.y * 64 + (lane % 16) * 4;
    int begin = row_ptr[segment], end = row_ptr[segment + 1];
    float center[4], grad[4], mean_direction[4] = {}, output_direction[4] = {};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        center[c] = col + c < channels ? output[int64_t(segment) * channels + col + c] : 0.f;
        grad[c] = col + c < channels ? grad_output[int64_t(segment) * channels + col + c] : 0.f;
    }
    for (int i = begin + edge_lane; i < end; i += 8) {
        int64_t offset = int64_t(edge_order[i]) * channels + col;
        alignas(16) float p[4], h[4], v[4], value_direction[4];
        if (col < channels) {
            *reinterpret_cast<float4 *>(p) =
                *reinterpret_cast<const float4 *>(probabilities + offset);
            *reinterpret_cast<float4 *>(h) =
                *reinterpret_cast<const float4 *>(direction_logits + offset);
            *reinterpret_cast<float4 *>(v) = *reinterpret_cast<const float4 *>(values + offset);
            *reinterpret_cast<float4 *>(value_direction) =
                *reinterpret_cast<const float4 *>(direction_values + offset);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                mean_direction[c] += p[c] * h[c];
                output_direction[c] += p[c] * (value_direction[c] + h[c] * (v[c] - center[c]));
            }
        }
    }
    // Reduce the probability-weighted direction and output variation across warps.
    __shared__ float moments[2][4][64];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float m = mean_direction[c] + __shfl_xor_sync(0xffffffff, mean_direction[c], 16);
        float v = output_direction[c] + __shfl_xor_sync(0xffffffff, output_direction[c], 16);
        if (lane < 16) {
            moments[0][threadIdx.x / 32][(lane % 16) * 4 + c] = m;
            moments[1][threadIdx.x / 32][(lane % 16) * 4 + c] = v;
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        int feature = (lane % 16) * 4 + c;
        mean_direction[c] = moments[0][0][feature] + moments[0][1][feature] +
                            moments[0][2][feature] + moments[0][3][feature];
        output_direction[c] = moments[1][0][feature] + moments[1][1][feature] +
                              moments[1][2][feature] + moments[1][3][feature];
        if (edge_lane == 0 && col + c < channels)
            grad_grad[int64_t(segment) * channels + col + c] = output_direction[c];
    }
    for (int i = begin + edge_lane; i < end; i += 8) {
        int64_t offset = int64_t(edge_order[i]) * channels + col;
        if (col < channels) {
            alignas(16) float p[4], h[4], v[4], value_direction[4], value_grad[4] = {},
                                                                    logit_grad[4];
            *reinterpret_cast<float4 *>(p) =
                *reinterpret_cast<const float4 *>(probabilities + offset);
            *reinterpret_cast<float4 *>(h) =
                *reinterpret_cast<const float4 *>(direction_logits + offset);
            *reinterpret_cast<float4 *>(v) = *reinterpret_cast<const float4 *>(values + offset);
            *reinterpret_cast<float4 *>(value_direction) =
                *reinterpret_cast<const float4 *>(direction_values + offset);
            if (accumulate_values)
                *reinterpret_cast<float4 *>(value_grad) =
                    *reinterpret_cast<const float4 *>(grad_values + offset);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float centered_direction = h[c] - mean_direction[c];
                logit_grad[c] = grad[c] * p[c] *
                                (centered_direction * (v[c] - center[c]) + value_direction[c] -
                                    output_direction[c]);
                value_grad[c] += grad[c] * p[c] * centered_direction;
            }
            *reinterpret_cast<float4 *>(grad_logits + offset) =
                *reinterpret_cast<float4 *>(logit_grad);
            *reinterpret_cast<float4 *>(grad_values + offset) =
                *reinterpret_cast<float4 *>(value_grad);
        }
    }
}

} // namespace

void online_tile(const at::Tensor &logits, const at::Tensor &values, const at::Tensor &row_ptr,
    const at::Tensor &edge_order, const at::Tensor &probabilities, const at::Tensor &output,
    int64_t channels, bool with_values, bool save_probabilities) {
    const c10::cuda::CUDAGuard guard(logits.device());
#define LAUNCH_ONLINE(A, S)                                                                        \
    online_kernel<A, S><<<dim3(row_ptr.numel() - 1, (channels + 63) / 64), 128, 0,                 \
        at::cuda::getCurrentCUDAStream()>>>(logits.data_ptr<float>(), values.data_ptr<float>(),    \
        row_ptr.data_ptr<int32_t>(), edge_order.data_ptr<int32_t>(),                               \
        probabilities.data_ptr<float>(), output.data_ptr<float>(), channels, row_ptr.numel() - 1)
    if (row_ptr.numel() > 1) {
        if (with_values) {
            if (save_probabilities) {
                LAUNCH_ONLINE(true, true);
            } else {
                LAUNCH_ONLINE(true, false);
            }
        } else {
            LAUNCH_ONLINE(false, true);
        }
    }
#undef LAUNCH_ONLINE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_backward(const at::Tensor &grad, const at::Tensor &probabilities,
    const at::Tensor &row_ptr, const at::Tensor &edge_order, const at::Tensor &grad_logits,
    int64_t channels) {
    const c10::cuda::CUDAGuard guard(grad.device());
    if (row_ptr.numel() > 1)
        softmax_backward_kernel<<<dim3(row_ptr.numel() - 1, (channels + 63) / 64), 128, 0,
            at::cuda::getCurrentCUDAStream()>>>(grad.data_ptr<float>(),
            probabilities.data_ptr<float>(), row_ptr.data_ptr<int32_t>(),
            edge_order.data_ptr<int32_t>(), grad_logits.data_ptr<float>(), channels,
            row_ptr.numel() - 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_double_backward(const at::Tensor &grad, const at::Tensor &values,
    const at::Tensor &output, const at::Tensor &probabilities, const at::Tensor &direction_logits,
    const at::Tensor &direction_values, const at::Tensor &row_ptr, const at::Tensor &edge_order,
    const at::Tensor &grad_logits, const at::Tensor &grad_values, const at::Tensor &grad_grad,
    int64_t channels, bool accumulate_values) {
    const c10::cuda::CUDAGuard guard(grad.device());
    if (row_ptr.numel() > 1)
        attention_second_kernel<<<dim3(row_ptr.numel() - 1, (channels + 63) / 64), 128, 0,
            at::cuda::getCurrentCUDAStream()>>>(grad.data_ptr<float>(), values.data_ptr<float>(),
            output.data_ptr<float>(), probabilities.data_ptr<float>(),
            direction_logits.data_ptr<float>(), direction_values.data_ptr<float>(),
            row_ptr.data_ptr<int32_t>(), edge_order.data_ptr<int32_t>(),
            grad_logits.data_ptr<float>(), grad_values.data_ptr<float>(),
            grad_grad.data_ptr<float>(), channels, row_ptr.numel() - 1, accumulate_values);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kAttentionThreads = 256;

__global__ void fused_line_attention_backward_kernel(
    const float *__restrict__ grad_source_out, const float *__restrict__ grad_target_out,
    const float *__restrict__ values, const float *__restrict__ source_out,
    const float *__restrict__ target_out, const float *__restrict__ source_alpha,
    const float *__restrict__ target_alpha, const int64_t *__restrict__ source_index,
    const int64_t *__restrict__ target_index, float *__restrict__ grad_source_logits,
    float *__restrict__ grad_target_logits, float *__restrict__ grad_values, int64_t rows,
    int kDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t row = idx / kDim;
    int dim = idx - row * kDim;
    int64_t s = source_index[row];
    int64_t t = target_index[row];
    float v = values[idx];
    float sa = source_alpha[idx];
    float ta = target_alpha[idx];
    float gs = grad_source_out[s * kDim + dim];
    float gt = grad_target_out[t * kDim + dim];
    float os = source_out[s * kDim + dim];
    float ot = target_out[t * kDim + dim];
    grad_source_logits[idx] = sa * gs * (v - os);
    grad_target_logits[idx] = ta * gt * (v - ot);
    grad_values[idx] = sa * gs + ta * gt;
}

__global__ void line_edge_residual_forward_kernel(const float *__restrict__ values,
                                                  const float *__restrict__ edge_feat,
                                                  const float *__restrict__ edge_res_weight,
                                                  float *__restrict__ out, int64_t rows, int kDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int dim = idx % kDim;
    out[idx] = values[idx] + edge_res_weight[dim] * edge_feat[idx];
}

} // namespace

std::vector<at::Tensor>
fused_line_attention_backward(const at::Tensor &grad_source_out, const at::Tensor &grad_target_out,
                              const at::Tensor &values, const at::Tensor &source_out,
                              const at::Tensor &target_out, const at::Tensor &source_alpha,
                              const at::Tensor &target_alpha, const at::Tensor &source_index,
                              const at::Tensor &target_index) {
    const int kDim = values.size(1);
    TORCH_CHECK(grad_source_out.is_cuda() && grad_target_out.is_cuda() && values.is_cuda() &&
                    source_out.is_cuda() && target_out.is_cuda() && source_alpha.is_cuda() &&
                    target_alpha.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
                "fused_line_attention_backward: tensors must be CUDA");
    TORCH_CHECK(values.scalar_type() == at::kFloat && grad_source_out.scalar_type() == at::kFloat &&
                    grad_target_out.scalar_type() == at::kFloat,
                "fused_line_attention_backward: float tensors must be float32");
    auto grad_source_logits = at::empty_like(values);
    auto grad_target_logits = at::empty_like(values);
    auto grad_values = at::empty_like(values);
    auto grad_source_out_c = grad_source_out.contiguous();
    auto grad_target_out_c = grad_target_out.contiguous();
    auto values_c = values.contiguous();
    auto source_out_c = source_out.contiguous();
    auto target_out_c = target_out.contiguous();
    auto source_alpha_c = source_alpha.contiguous();
    auto target_alpha_c = target_alpha.contiguous();
    auto source_index_c = source_index.contiguous();
    auto target_index_c = target_index.contiguous();
    int64_t rows = values_c.size(0);
    int blocks = static_cast<int>((rows * kDim + kAttentionThreads - 1) / kAttentionThreads);
    fused_line_attention_backward_kernel<<<blocks, kAttentionThreads, 0,
                                           c10::cuda::getCurrentCUDAStream()>>>(
        grad_source_out_c.data_ptr<float>(), grad_target_out_c.data_ptr<float>(),
        values_c.data_ptr<float>(), source_out_c.data_ptr<float>(), target_out_c.data_ptr<float>(),
        source_alpha_c.data_ptr<float>(), target_alpha_c.data_ptr<float>(),
        source_index_c.data_ptr<int64_t>(), target_index_c.data_ptr<int64_t>(),
        grad_source_logits.data_ptr<float>(), grad_target_logits.data_ptr<float>(),
        grad_values.data_ptr<float>(), rows, kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_source_logits, grad_target_logits, grad_values};
}

at::Tensor line_edge_residual_forward(const at::Tensor &values, const at::Tensor &edge_feat,
                                      const at::Tensor &edge_res_weight) {
    const int kDim = values.size(1);
    TORCH_CHECK(values.is_cuda() && edge_feat.is_cuda() && edge_res_weight.is_cuda(),
                "line_edge_residual_forward: tensors must be CUDA");
    TORCH_CHECK(values.scalar_type() == at::kFloat && edge_feat.scalar_type() == at::kFloat &&
                    edge_res_weight.scalar_type() == at::kFloat,
                "line_edge_residual_forward: tensors must be float32");
    TORCH_CHECK(values.dim() == 2 && kDim > 0 && kDim % 32 == 0 &&
                    edge_feat.sizes() == values.sizes(),
                "line_edge_residual_forward: values/edge_feat must be [E, D]");
    TORCH_CHECK(edge_res_weight.numel() == kDim,
                "line_edge_residual_forward: edge_res_weight must have D elements");

    auto values_c = values.contiguous();
    auto edge_feat_c = edge_feat.contiguous();
    auto edge_res_weight_c = edge_res_weight.contiguous();
    auto out = at::empty_like(values_c);
    int64_t rows = values_c.size(0);
    int blocks = static_cast<int>((rows * kDim + kAttentionThreads - 1) / kAttentionThreads);
    line_edge_residual_forward_kernel<<<blocks, kAttentionThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        values_c.data_ptr<float>(), edge_feat_c.data_ptr<float>(),
        edge_res_weight_c.data_ptr<float>(), out.data_ptr<float>(), rows, kDim);
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
            name=f"matris_online_{key}",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--extended-lambda"],
            verbose=False,
        )
        extension.register_autograd()
        _EXTENSION = extension
    return _EXTENSION


_SEGMENT_EXTENSION = None


SEGMENT_CPP_SOURCE = r"""

#include <torch/extension.h>

at::Tensor segment_counts_cuda(const at::Tensor &, int64_t);
at::Tensor segment_rows_cuda(const at::Tensor &, const at::Tensor &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("counts", &segment_counts_cuda);
    m.def("rows", &segment_rows_cuda);
}
"""


SEGMENT_CUDA_SOURCE = r"""

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <cuda_runtime.h>

namespace {
__global__ void counts_kernel(const int64_t *index, int stride, int rows, int *counts) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
        atomicAdd(counts + index[int64_t(row) * stride] + 1, 1);
}

__global__ void rows_kernel(const int64_t *index, int stride, int rows, int *cursor, int *order) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int position = atomicAdd(cursor + index[int64_t(row) * stride], 1);
        order[position] = row;
    }
}

} // namespace

at::Tensor segment_counts_cuda(const at::Tensor &index, int64_t segments) {
    const c10::cuda::CUDAGuard guard(index.device());
    TORCH_CHECK(index.scalar_type() == at::kLong && index.dim() == 1 && index.numel() <= INT_MAX,
                "Expected int64 segment indices with at most INT_MAX rows");
    auto counts = at::zeros({segments + 1}, index.options().dtype(at::kInt));
    if (index.numel()) {
        counts_kernel<<<(index.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            index.data_ptr<int64_t>(), index.stride(0), index.numel(), counts.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return counts;
}

at::Tensor segment_rows_cuda(const at::Tensor &index, const at::Tensor &ptr) {
    const c10::cuda::CUDAGuard guard(index.device());
    auto cursor = ptr.slice(0, 0, ptr.numel() - 1).clone();
    auto order = at::empty({index.numel()}, ptr.options());
    if (index.numel()) {
        rows_kernel<<<(index.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            index.data_ptr<int64_t>(), index.stride(0), index.numel(), cursor.data_ptr<int>(),
            order.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return order;
}
"""


@torch.compiler.assume_constant_result
def get_segment_extension():
    """Compile this operator family once and reuse its source-addressed cache."""
    global _SEGMENT_EXTENSION
    if _SEGMENT_EXTENSION is None:
        os.environ.setdefault("MAX_JOBS", "4")
        key = hashlib.sha256(
            (
                f"{sys.implementation.cache_tag}:{torch.__version__}:{torch.version.cuda}\n"
                + SEGMENT_CPP_SOURCE
                + "\n"
                + SEGMENT_CUDA_SOURCE
            ).encode()
        ).hexdigest()[:16]
        extension = load_inline(
            name=f"matris_segments_{key}",
            cpp_sources=SEGMENT_CPP_SOURCE,
            cuda_sources=SEGMENT_CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--extended-lambda"],
            verbose=False,
        )
        _SEGMENT_EXTENSION = extension
    return _SEGMENT_EXTENSION


def online_attention(x, values, layout, *, save_alpha=True):
    """Online weighted reduction; probabilities are needed only for backward."""

    ptr, order = layout
    segments = ptr.numel() - 1
    output = x.new_empty((segments, x.shape[1]))
    alpha = torch.empty_like(x) if save_alpha else None
    if segments:
        get_extension().online_tile(
            x,
            values,
            ptr,
            order,
            alpha if save_alpha else output,
            output,
            x.shape[1],
            True,
            save_alpha,
        )
    return output, alpha


def segment_layout(index, num_segments):
    """Build an exact CSR permutation for arbitrary, potentially strided indices."""
    extension = get_segment_extension()
    ptr = extension.counts(index, num_segments)
    torch.cumsum(ptr, dim=0, out=ptr)
    return ptr, extension.rows(index, ptr)


class SoftmaxBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(grad, y, index, ptr, order):
        get_extension()
        return torch.ops.matris.softmax_backward(grad, y, index, ptr, order)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, h):
        grad, y, index, ptr, order = ctx.saved_tensors
        dot = y.new_zeros((ptr.numel() - 1, y.shape[1])).index_add(0, index, y * grad)
        hdot = y.new_zeros((ptr.numel() - 1, y.shape[1])).index_add(0, index, y * h)
        return (
            y * (h - hdot[index]),
            h * (grad - dot[index]) - grad * hdot[index],
            None,
            None,
            None,
        )


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, index, ptr, order):
        get_extension()
        return torch.ops.matris.softmax(x, index, ptr, order)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output, *inputs[1:])

    @staticmethod
    def backward(ctx, grad):
        y, index, ptr, order = ctx.saved_tensors
        return (
            SoftmaxBackwardFunction.apply(grad, y, index, ptr, order),
            None,
            None,
            None,
        )


def online_softmax(x, index, num_segments, *, layout=None):
    """Return probabilities; reusable layout contains only graph topology, not logits."""
    if layout is None:
        layout = segment_layout(index, num_segments)
    return SoftmaxFunction.apply(x, index, *layout)






class AttentionDoubleBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gs,
        gt,
        values,
        source_out,
        target_out,
        source_alpha,
        target_alpha,
        sptr,
        sorder,
        tptr,
        torder,
        hs,
        ht,
        hv,
    ):
        get_extension()
        return torch.ops.matris.attention_double_backward(
            gs,
            gt,
            values,
            source_out,
            target_out,
            source_alpha,
            target_alpha,
            sptr,
            sorder,
            tptr,
            torder,
            hs,
            ht,
            hv,
        )


class AttentionBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        gs,
        gt,
        source_logits,
        target_logits,
        values,
        source_out,
        target_out,
        source_alpha,
        target_alpha,
        source_index,
        target_index,
        sptr,
        sorder,
        tptr,
        torder,
    ):
        get_extension()
        return torch.ops.matris.attention_backward(
            gs,
            gt,
            source_logits,
            target_logits,
            values,
            source_out,
            target_out,
            source_alpha,
            target_alpha,
            source_index,
            target_index,
            sptr,
            sorder,
            tptr,
            torder,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, hs, ht, hv):
        gs, gt, _, _, values, so, to, sa, ta, _, _, sptr, sorder, tptr, torder = (
            ctx.saved_tensors
        )
        grads = AttentionDoubleBackwardFunction.apply(
            gs, gt, values, so, to, sa, ta, sptr, sorder, tptr, torder, hs, ht, hv
        )
        return (*grads, *[None] * 10)


class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        source_logits,
        target_logits,
        values,
        source_index,
        target_index,
        sptr,
        sorder,
        tptr,
        torder,
        save_alpha,
    ):
        get_extension()
        return torch.ops.matris.attention(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            sptr,
            sorder,
            tptr,
            torder,
            save_alpha,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1], *output)
        ctx.mark_non_differentiable(*(x for x in output[2:] if x is not None))

    @staticmethod
    def backward(ctx, gs, gt, gsa, gta):
        sl, tl, values, si, ti, sptr, sorder, tptr, torder, so, to, sa, ta = (
            ctx.saved_tensors
        )
        # The analytic second VJP already includes the cached outputs' dependence.
        grads = AttentionBackwardFunction.apply(
            gs,
            gt,
            sl,
            tl,
            values,
            so.detach(),
            to.detach(),
            sa,
            ta,
            si,
            ti,
            sptr,
            sorder,
            tptr,
            torder,
        )
        return (*grads, *[None] * 7)


class LineResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(values, edge_feat, weight):
        get_extension()
        return torch.ops.matris.line_residual(values, edge_feat, weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1], inputs[2])

    @staticmethod
    def backward(ctx, grad):
        edge, weight = ctx.saved_tensors
        return (grad, grad * weight, (grad * edge).sum_to_size(weight.shape))


attention_op = AttentionFunction.apply
