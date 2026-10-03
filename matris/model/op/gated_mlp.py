"""CUDA source, lazy inline loading and Python interfaces for gated_mlp."""

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

void gated_forward(const at::Tensor &core, const at::Tensor &gate, const at::Tensor &core_weight,
    const at::Tensor &core_bias, const at::Tensor &gate_weight, const at::Tensor &gate_bias,
    const at::Tensor &output, int64_t num_rows, int64_t channels, int64_t core_row_stride,
    int64_t core_col_stride, int64_t gate_row_stride, int64_t gate_col_stride, double eps,
    int64_t rows_per_block);
void gated_derivatives(const at::Tensor &core, const at::Tensor &gate,
    const at::Tensor &core_weight, const at::Tensor &core_bias, const at::Tensor &gate_weight,
    const at::Tensor &gate_bias, const at::Tensor &grad, const at::Tensor &direction_core,
    const at::Tensor &direction_gate, const at::Tensor &direction_core_weight,
    const at::Tensor &direction_core_bias, const at::Tensor &direction_gate_weight,
    const at::Tensor &direction_gate_bias, const at::Tensor &grad_core, const at::Tensor &grad_gate,
    const at::Tensor &grad_grad, const at::Tensor &partial, int64_t num_rows, int64_t channels,
    double eps, bool second_order, int64_t rows_per_block, bool parameter_grads);
void gated_parameter_reduce(const at::Tensor &partial, const at::Tensor &output, int64_t num_rows,
    int64_t channels, int64_t rows_per_block);

namespace matris_ops {
using Tensor = at::Tensor;
using OptionalTensor = std::optional<Tensor>;

std::tuple<Tensor, Tensor, Tensor> gated_backward(const Tensor &grad, const Tensor &core,
    const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    int64_t rows = core.size(0), dim = core.size(1), padded = 1;
    while (padded < dim)
        padded <<= 1;
    int64_t block_rows = std::max<int64_t>(4, 1024 / padded);
    int64_t tiles = (rows + block_rows - 1) / block_rows;
    auto dx = at::empty(core.sizes(), core.options());
    auto dz = at::empty(gate.sizes(), gate.options());
    auto partial = at::empty({tiles, 4, dim}, core.options());
    gated_derivatives(core.contiguous(), gate.contiguous(), core_weight.contiguous(),
        core_bias.contiguous(), gate_weight.contiguous(), gate_bias.contiguous(), grad.contiguous(),
        core, core, core_weight, core_bias, gate_weight, gate_bias, dx, dz, dx, partial, rows, dim,
        eps, false, block_rows, true);
    int64_t groups = (tiles + 127) / 128;
    auto parameters = at::empty({4, dim}, core.options());
    auto reduced =
        groups == 1 ? parameters.unsqueeze(0) : at::empty({groups, 4, dim}, core.options());
    gated_parameter_reduce(partial, reduced, tiles, dim, 128);
    if (groups != 1) {
        int64_t reduction_rows = 1;
        while (reduction_rows < groups)
            reduction_rows <<= 1;
        gated_parameter_reduce(reduced, parameters, groups, dim, reduction_rows);
    }
    return {dx, dz, parameters};
}

std::tuple<Tensor, Tensor, Tensor> gated_backward_meta(const Tensor &grad, const Tensor &core,
    const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    return {at::empty_symint(core.sym_sizes(), core.options()),
        at::empty_symint(gate.sym_sizes(), gate.options()),
        at::empty_symint({4, core.sym_size(1)}, core.options())};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> gated_double_backward(const Tensor &grad,
    const Tensor &core, const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps, const Tensor &hx,
    const Tensor &hz, const Tensor &hp) {
    int64_t rows = core.size(0), dim = core.size(1), padded = 1;
    while (padded < dim)
        padded <<= 1;
    int64_t block_rows = std::max<int64_t>(4, 1024 / padded);
    int64_t tiles = (rows + block_rows - 1) / block_rows;
    auto dx = at::empty(core.sizes(), core.options());
    auto dz = at::empty(gate.sizes(), gate.options());
    auto partial = at::empty({tiles, 4, dim}, core.options());
    auto dg = at::empty(core.sizes(), core.options());
    auto directions = hp.unbind();
    gated_derivatives(core.contiguous(), gate.contiguous(), core_weight.contiguous(),
        core_bias.contiguous(), gate_weight.contiguous(), gate_bias.contiguous(), grad.contiguous(),
        hx.contiguous(), hz.contiguous(), directions[0].contiguous(), directions[1].contiguous(),
        directions[2].contiguous(), directions[3].contiguous(), dx, dz, dg, partial, rows, dim, eps,
        true, block_rows, true);
    int64_t groups = (tiles + 127) / 128;
    auto parameters = at::empty({4, dim}, core.options());
    auto reduced =
        groups == 1 ? parameters.unsqueeze(0) : at::empty({groups, 4, dim}, core.options());
    gated_parameter_reduce(partial, reduced, tiles, dim, 128);
    if (groups != 1) {
        int64_t reduction_rows = 1;
        while (reduction_rows < groups)
            reduction_rows <<= 1;
        gated_parameter_reduce(reduced, parameters, groups, dim, reduction_rows);
    }
    return {dg, dx, dz, parameters};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> gated_double_backward_meta(const Tensor &grad,
    const Tensor &core, const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps, const Tensor &hx,
    const Tensor &hz, const Tensor &hp) {
    return {at::empty_symint(grad.sym_sizes(), grad.options()),
        at::empty_symint(core.sym_sizes(), core.options()),
        at::empty_symint(gate.sym_sizes(), gate.options()),
        at::empty_symint({4, core.sym_size(1)}, core.options())};
}

std::tuple<Tensor, Tensor> gated_input_backward(const Tensor &grad, const Tensor &core,
    const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    auto dx = at::empty(core.sizes(), core.options());
    auto dz = at::empty(gate.sizes(), gate.options());
    gated_derivatives(core.contiguous(), gate.contiguous(), core_weight.contiguous(),
        core_bias.contiguous(), gate_weight.contiguous(), gate_bias.contiguous(), grad.contiguous(),
        core, gate, core_weight, core_bias, gate_weight, gate_bias, dx, dz, dx, dx, core.size(0),
        core.size(1), eps, false, 4, false);
    return {dx, dz};
}

std::tuple<Tensor, Tensor> gated_input_backward_meta(const Tensor &grad, const Tensor &core,
    const Tensor &gate, const Tensor &core_weight, const Tensor &core_bias,
    const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    return {at::empty_symint(core.sym_sizes(), core.options()),
        at::empty_symint(gate.sym_sizes(), gate.options())};
}

Tensor gated_tail(const Tensor &core, const Tensor &gate, const Tensor &core_weight,
    const Tensor &core_bias, const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    auto output = at::empty(core.sizes(), core.options());
    gated_forward(core, gate, core_weight, core_bias, gate_weight, gate_bias, output, core.size(0),
        core.size(1), core.stride(0), core.stride(1), gate.stride(0), gate.stride(1), eps, 4);
    return output;
}

Tensor gated_tail_meta(const Tensor &core, const Tensor &gate, const Tensor &core_weight,
    const Tensor &core_bias, const Tensor &gate_weight, const Tensor &gate_bias, double eps) {
    return at::empty_symint(core.sym_sizes(), core.options());
}
} // namespace matris_ops

TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def(
        "gated_backward(Tensor grad, Tensor core, Tensor gate, Tensor core_weight, Tensor "
        "core_bias, Tensor gate_weight, Tensor gate_bias, float eps) -> (Tensor, Tensor, Tensor)");
    m.def("gated_double_backward(Tensor grad, Tensor core, Tensor gate, Tensor core_weight, Tensor "
          "core_bias, Tensor gate_weight, Tensor gate_bias, float eps, Tensor hx, Tensor hz, "
          "Tensor hp) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("gated_input_backward(Tensor grad, Tensor core, Tensor gate, Tensor core_weight, Tensor "
          "core_bias, Tensor gate_weight, Tensor gate_bias, float eps) -> (Tensor, Tensor)");
    m.def("gated_tail(Tensor core, Tensor gate, Tensor core_weight, Tensor core_bias, Tensor "
          "gate_weight, Tensor gate_bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("gated_backward", TORCH_FN(matris_ops::gated_backward));
    m.impl("gated_double_backward", TORCH_FN(matris_ops::gated_double_backward));
    m.impl("gated_input_backward", TORCH_FN(matris_ops::gated_input_backward));
    m.impl("gated_tail", TORCH_FN(matris_ops::gated_tail));
}

TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("gated_backward", TORCH_FN(matris_ops::gated_backward_meta));
    m.impl("gated_double_backward", TORCH_FN(matris_ops::gated_double_backward_meta));
    m.impl("gated_input_backward", TORCH_FN(matris_ops::gated_input_backward_meta));
    m.impl("gated_tail", TORCH_FN(matris_ops::gated_tail_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto register_gradient = py::module_::import("torch.library").attr("register_autograd");
        auto gated_mlp = py::module_::import("matris.model.op.gated_mlp");
        register_gradient("matris::gated_backward",
            gated_mlp.attr("GatedBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                gated_mlp.attr("GatedBackwardFunction").attr("setup_context"));
        register_gradient("matris::gated_tail",
            gated_mlp.attr("GatedTailFunction").attr("backward"),
            py::arg("setup_context") = gated_mlp.attr("GatedTailFunction").attr("setup_context"));
    });
    m.def("gated_forward", &gated_forward);
    m.def("gated_derivatives", &gated_derivatives);
    m.def("gated_parameter_reduce", &gated_parameter_reduce);
}
"""


CUDA_SOURCE = COMMON_SOURCE + r"""


using namespace matris_cuda;

namespace {
__device__ float warp_sum(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, delta);
    return x;
}

// One warp owns one row; K is the padded channel count per lane.
// The launch bounds cap register use in the mixed second derivative.
template <int K, bool Forward, bool Second, bool Parameters>
__global__ __launch_bounds__((Forward || !Parameters) ? 128 : (K <= 8 ? 1024 / K : 128),
    Second ? ((768 / (K <= 8 ? 1024 / K : 128)) > 0 ? 768 / (K <= 8 ? 1024 / K : 128) : 1)
           : 1)
void gate_kernel(const float *__restrict__ x, const float *__restrict__ z,
    const float *__restrict__ w, const float *__restrict__ b, const float *__restrict__ v,
    const float *__restrict__ c, const float *__restrict__ g, const float *__restrict__ hx,
    const float *__restrict__ hz, const float *__restrict__ hw, const float *__restrict__ hb,
    const float *__restrict__ hv, const float *__restrict__ hc, float *dx, float *dz, float *dg,
    float *partial, int64_t n, int d, float eps, float inverse_d, int64_t xs, int64_t xc,
    int64_t zs, int64_t zc) {
    extern __shared__ float parameter_sum[];
    int lane = threadIdx.x, r = threadIdx.y;
    int64_t row = (int64_t)blockIdx.x * blockDim.y + r;
    int thread = r * 32 + lane, threads = blockDim.y * 32;
    if constexpr (Parameters) {
        for (int i = lane; i < 4 * d; i += 32)
            parameter_sum[r * 4 * d + i] = 0.f;
    }
    // x/z are core/gate inputs; w,b and v,c are their LayerNorm parameters.
    // nx/nz: normalized inputs; a/q: affine values; sig/t: their sigmoids.
    // h-prefixed inputs carry the incoming second-VJP directions.
    float nx[K], nz[K], ww[K], vv[K], a[K], q[K], sig[K], t[K];
    float xsum = 0, zsum = 0;
    // Shift before computing variance to retain precision for large row offsets.
    float x0 = row < n ? x[row * xs] : 0.f, z0 = row < n ? z[row * zs] : 0.f;
    if constexpr (Forward && K >= 4) {
#pragma unroll
        for (int k = 0; k < K; k += 4) {
            int col = lane * K + k;
            float4 xx = make_float4(0, 0, 0, 0), zz = xx;
            if (row < n && col < d) {
                if (xc == 1 && xs % 4 == 0 && uintptr_t(x) % 16 == 0)
                    xx = *reinterpret_cast<const float4 *>(x + row * xs + col);
                else
                    xx = make_float4(x[row * xs + col * xc], x[row * xs + (col + 1) * xc],
                        x[row * xs + (col + 2) * xc], x[row * xs + (col + 3) * xc]);
                if (zc == 1 && zs % 4 == 0 && uintptr_t(z) % 16 == 0)
                    zz = *reinterpret_cast<const float4 *>(z + row * zs + col);
                else
                    zz = make_float4(z[row * zs + col * zc], z[row * zs + (col + 1) * zc],
                        z[row * zs + (col + 2) * zc], z[row * zs + (col + 3) * zc]);
                xx.x -= x0;
                xx.y -= x0;
                xx.z -= x0;
                xx.w -= x0;
                zz.x -= z0;
                zz.y -= z0;
                zz.z -= z0;
                zz.w -= z0;
            }
            nx[k] = xx.x;
            nx[k + 1] = xx.y;
            nx[k + 2] = xx.z;
            nx[k + 3] = xx.w;
            nz[k] = zz.x;
            nz[k + 1] = zz.y;
            nz[k + 2] = zz.z;
            nz[k + 3] = zz.w;
            xsum += xx.x + xx.y + xx.z + xx.w;
            zsum += zz.x + zz.y + zz.z + zz.w;
        }
    } else {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            int col = (Forward ? lane * K + k : k * 32 + lane);
            bool valid = row < n && col < d;
            nx[k] = valid ? x[row * xs + col * xc] - x0 : 0.f;
            nz[k] = valid ? z[row * zs + col * zc] - z0 : 0.f;
            xsum += nx[k];
            zsum += nz[k];
        }
    }
    xsum = warp_sum(xsum) * inverse_d;
    zsum = warp_sum(zsum) * inverse_d;
    float vx = 0, vz = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        bool valid = (Forward ? lane * K + k : k * 32 + lane) < d;
        nx[k] = valid ? nx[k] - xsum : 0.f;
        nz[k] = valid ? nz[k] - zsum : 0.f;
        vx += nx[k] * nx[k];
        vz += nz[k] * nz[k];
    }
    float rx = rsqrtf(warp_sum(vx) * inverse_d + eps), rz = rsqrtf(warp_sum(vz) * inverse_d + eps);
    if constexpr (Forward && K >= 4) {
#pragma unroll
        for (int k = 0; k < K; k += 4) {
            int col = lane * K + k;
            float4 weights = make_float4(0, 0, 0, 0), bias = weights, gate_weights = weights,
                   gate_bias = weights;
            if (col < d &&
                ((uintptr_t(w) | uintptr_t(b) | uintptr_t(v) | uintptr_t(c)) & 15) == 0) {
                weights = *reinterpret_cast<const float4 *>(w + col);
                bias = *reinterpret_cast<const float4 *>(b + col);
                gate_weights = *reinterpret_cast<const float4 *>(v + col);
                gate_bias = *reinterpret_cast<const float4 *>(c + col);
            } else if (col < d) {
                weights = make_float4(w[col], w[col + 1], w[col + 2], w[col + 3]);
                bias = make_float4(b[col], b[col + 1], b[col + 2], b[col + 3]);
                gate_weights = make_float4(v[col], v[col + 1], v[col + 2], v[col + 3]);
                gate_bias = make_float4(c[col], c[col + 1], c[col + 2], c[col + 3]);
            }
            ww[k] = weights.x;
            ww[k + 1] = weights.y;
            ww[k + 2] = weights.z;
            ww[k + 3] = weights.w;
            vv[k] = gate_weights.x;
            vv[k + 1] = gate_weights.y;
            vv[k + 2] = gate_weights.z;
            vv[k + 3] = gate_weights.w;
            a[k] = bias.x;
            a[k + 1] = bias.y;
            a[k + 2] = bias.z;
            a[k + 3] = bias.w;
            q[k] = gate_bias.x;
            q[k + 1] = gate_bias.y;
            q[k + 2] = gate_bias.z;
            q[k + 3] = gate_bias.w;
        }
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        int col = (Forward ? lane * K + k : k * 32 + lane);
        bool valid = col < d;
        nx[k] *= rx;
        nz[k] *= rz;
        if constexpr (!(Forward && K >= 4)) {
            ww[k] = valid ? w[col] : 0.f;
            vv[k] = valid ? v[col] : 0.f;
            a[k] = valid ? b[col] : 0.f;
            q[k] = valid ? c[col] : 0.f;
        }
        a[k] = nx[k] * ww[k] + a[k];
        q[k] = nz[k] * vv[k] + q[k];
        if constexpr (Forward) {
            a[k] = a[k] * sigmoid(a[k]) * sigmoid(q[k]);
            if constexpr (K < 4) {
                if (row < n && valid)
                    dx[row * d + col] = a[k];
            }
        } else {
            sig[k] = sigmoid(a[k]);
            t[k] = sigmoid(q[k]);
        }
    }
    if constexpr (Forward && K >= 4) {
#pragma unroll
        for (int k = 0; k < K; k += 4) {
            int col = lane * K + k;
            if (row < n && col < d)
                *reinterpret_cast<float4 *>(dx + row * d + col) =
                    make_float4(a[k], a[k + 1], a[k + 2], a[k + 3]);
        }
    }
    // First VJP through the activations, affine transforms and normalization.
    if constexpr (!Forward) {
        float da[K], dq[K], ux[K], uz[K], px[K], pz[K];
        float mx = 0, mz = 0, meanx = 0, meanz = 0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            int col = (Forward ? lane * K + k : k * 32 + lane);
            float grad = (row < n && col < d) ? g[row * d + col] : 0.f;
            float ds = sig[k] * (1 + a[k] * (1 - sig[k])), dt = t[k] * (1 - t[k]);
            da[k] = grad * ds * t[k];
            dq[k] = grad * (a[k] * sig[k]) * dt;
            ux[k] = da[k] * ww[k];
            uz[k] = dq[k] * vv[k];
            mx += ux[k] * nx[k];
            mz += uz[k] * nz[k];
            meanx += ux[k];
            meanz += uz[k];
        }
        mx = warp_sum(mx) * inverse_d;
        mz = warp_sum(mz) * inverse_d;
        meanx = warp_sum(meanx) * inverse_d;
        meanz = warp_sum(meanz) * inverse_d;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            px[k] = ux[k] - meanx - nx[k] * mx;
            pz[k] = uz[k] - meanz - nz[k] * mz;
        }
        // Differentiate the first VJP along the supplied input/parameter directions.
        if constexpr (Second) {
            float dnx[K], dnz[K], mhx = 0, mhz = 0, tx = 0, tz = 0;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                int col = (Forward ? lane * K + k : k * 32 + lane);
                bool valid = row < n && col < d;
                dnx[k] = valid ? hx[row * d + col] : 0.f;
                dnz[k] = valid ? hz[row * d + col] : 0.f;
                mhx += dnx[k];
                mhz += dnz[k];
                tx += nx[k] * dnx[k];
                tz += nz[k] * dnz[k];
            }
            mhx = warp_sum(mhx) * inverse_d;
            mhz = warp_sum(mhz) * inverse_d;
            tx = warp_sum(tx) * inverse_d;
            tz = warp_sum(tz) * inverse_d;
            float mu = 0, mv = 0, mnu = 0, mnv = 0;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                int col = (Forward ? lane * K + k : k * 32 + lane);
                bool valid = col < d;
                float dn = valid ? rx * (dnx[k] - mhx - nx[k] * tx) : 0.f,
                      dm = valid ? rz * (dnz[k] - mhz - nz[k] * tz) : 0.f;
                float hweight = valid ? hw[col] : 0.f, hvweight = valid ? hv[col] : 0.f;
                float ad = dn * ww[k] + nx[k] * hweight + (valid ? hb[col] : 0.f);
                float qd = dm * vv[k] + nz[k] * hvweight + (valid ? hc[col] : 0.f);
                float grad = row < n && valid ? g[row * d + col] : 0.f;
                float ds = sig[k] * (1 + a[k] * (1 - sig[k])), dt = t[k] * (1 - t[k]);
                float dda =
                    grad * (sig[k] * (1 - sig[k]) * (2 + a[k] * (1 - 2 * sig[k])) * t[k] * ad +
                               ds * dt * qd);
                float ddq = grad * (ds * dt * ad + a[k] * sig[k] * dt * (1 - 2 * t[k]) * qd);
                float du = dda * ww[k] + da[k] * hweight, dv = ddq * vv[k] + dq[k] * hvweight;
                mu += du;
                mv += dv;
                mnu += du * nx[k] + ux[k] * dn;
                mnv += dv * nz[k] + uz[k] * dm;
                px[k] = rx * (du - dn * mx) - rx * rx * tx * px[k];
                pz[k] = rz * (dv - dm * mz) - rz * rz * tz * pz[k];
                if (row < n && valid) {
                    dg[row * d + col] = ds * t[k] * ad + a[k] * sig[k] * dt * qd;
                    parameter_sum[r * 4 * d + col] = dda * nx[k] + da[k] * dn;
                    parameter_sum[r * 4 * d + d + col] = dda;
                    parameter_sum[r * 4 * d + 2 * d + col] = ddq * nz[k] + dq[k] * dm;
                    parameter_sum[r * 4 * d + 3 * d + col] = ddq;
                }
            }
            mu = warp_sum(mu) * inverse_d;
            mv = warp_sum(mv) * inverse_d;
            mnu = warp_sum(mnu) * inverse_d;
            mnv = warp_sum(mnv) * inverse_d;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                int col = (Forward ? lane * K + k : k * 32 + lane);
                if (row < n && col < d) {
                    dx[row * d + col] = px[k] - rx * mu - rx * nx[k] * mnu;
                    dz[row * d + col] = pz[k] - rz * mv - rz * nz[k] * mnv;
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                int col = (Forward ? lane * K + k : k * 32 + lane);
                if (row < n && col < d) {
                    dx[row * d + col] = rx * px[k];
                    dz[row * d + col] = rz * pz[k];
                    if constexpr (Parameters) {
                        parameter_sum[r * 4 * d + col] = da[k] * nx[k];
                        parameter_sum[r * 4 * d + d + col] = da[k];
                        parameter_sum[r * 4 * d + 2 * d + col] = dq[k] * nz[k];
                        parameter_sum[r * 4 * d + 3 * d + col] = dq[k];
                    }
                }
            }
        }
    }

    // Fold per-row parameter gradients into one partial sum per block.
    if constexpr (Parameters) {
        __syncthreads();
        for (int i = thread; i < 4 * d; i += threads) {
            float sum = 0.f;
            for (int row = 0; row < blockDim.y; ++row)
                sum += parameter_sum[row * 4 * d + i];
            partial[(int64_t)blockIdx.x * 4 * d + i] = sum;
        }
    }
}

__global__ void gate_reduce_kernel(
    const float *__restrict__ x, float *y, int64_t n, int d, int rows) {
    int col = blockIdx.y * 32 + threadIdx.x;
    float value = 0;
    for (int64_t i = (int64_t)blockIdx.x * rows + threadIdx.y;
        i < n && i < (int64_t)(blockIdx.x + 1) * rows; i += blockDim.y)
        value += x[i * 4 * d + col];
    __shared__ float sums[8][32];
    sums[threadIdx.y][threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.y == 0) {
        for (int r = 1; r < 8; ++r)
            value += sums[r][threadIdx.x];
        y[(int64_t)blockIdx.x * 4 * d + col] = value;
    }
}

} // namespace

void gated_forward(const at::Tensor &core, const at::Tensor &gate, const at::Tensor &core_weight,
    const at::Tensor &core_bias, const at::Tensor &gate_weight, const at::Tensor &gate_bias,
    const at::Tensor &output, int64_t num_rows, int64_t channels, int64_t core_row_stride,
    int64_t core_col_stride, int64_t gate_row_stride, int64_t gate_col_stride, double eps,
    int64_t rows_per_block) {
    const c10::cuda::CUDAGuard guard(core.device());
    if (!num_rows)
        return;
    // Wider widths retain the same formula through native CUDA tensor operations.
    if (channels > 1024) {
        output.copy_(at::silu(at::layer_norm(core, {channels}, core_weight, core_bias, eps)) *
                     at::sigmoid(at::layer_norm(gate, {channels}, gate_weight, gate_bias, eps)));
        return;
    }
#define LAUNCH_GATE_FORWARD(K)                                                                     \
    gate_kernel<K, true, false, false><<<(num_rows + rows_per_block - 1) / rows_per_block,         \
        dim3(32, rows_per_block), 0, at::cuda::getCurrentCUDAStream()>>>(core.data_ptr<float>(),   \
        gate.data_ptr<float>(), core_weight.data_ptr<float>(), core_bias.data_ptr<float>(),        \
        gate_weight.data_ptr<float>(), gate_bias.data_ptr<float>(), nullptr, nullptr, nullptr,     \
        nullptr, nullptr, nullptr, nullptr, output.data_ptr<float>(), nullptr, nullptr, nullptr,   \
        num_rows, channels, eps, 1.f / static_cast<float>(channels), core_row_stride,              \
        core_col_stride, gate_row_stride, gate_col_stride)
    if (channels <= 32) {
        LAUNCH_GATE_FORWARD(1);
    } else if (channels <= 64) {
        LAUNCH_GATE_FORWARD(2);
    } else if (channels <= 128) {
        LAUNCH_GATE_FORWARD(4);
    } else if (channels <= 256) {
        LAUNCH_GATE_FORWARD(8);
    } else if (channels <= 512) {
        LAUNCH_GATE_FORWARD(16);
    } else {
        LAUNCH_GATE_FORWARD(32);
    }
#undef LAUNCH_GATE_FORWARD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gated_derivatives(const at::Tensor &core, const at::Tensor &gate,
    const at::Tensor &core_weight, const at::Tensor &core_bias, const at::Tensor &gate_weight,
    const at::Tensor &gate_bias, const at::Tensor &grad, const at::Tensor &direction_core,
    const at::Tensor &direction_gate, const at::Tensor &direction_core_weight,
    const at::Tensor &direction_core_bias, const at::Tensor &direction_gate_weight,
    const at::Tensor &direction_gate_bias, const at::Tensor &grad_core, const at::Tensor &grad_gate,
    const at::Tensor &grad_grad, const at::Tensor &partial, int64_t num_rows, int64_t channels,
    double eps, bool second_order, int64_t rows_per_block, bool parameter_grads) {
    const c10::cuda::CUDAGuard guard(core.device());
    if (!num_rows)
        return;
    // Wider widths retain the same formula through native CUDA tensor operations.
    if (channels > 1024) {
        auto x = core - core.select(1, 0).unsqueeze(1), z = gate - gate.select(1, 0).unsqueeze(1);
        x = x - x.mean(1, true);
        z = z - z.mean(1, true);
        auto rx = at::rsqrt(x.square().mean(1, true) + eps),
             rz = at::rsqrt(z.square().mean(1, true) + eps);
        auto nx = x * rx, nz = z * rz, a = nx * core_weight + core_bias,
             q = nz * gate_weight + gate_bias, s = at::sigmoid(a), t = at::sigmoid(q);
        auto f = a * s, ds = s * (1 + a * (1 - s)), dt = t * (1 - t), da = grad * ds * t,
             dq = grad * f * dt, ux = da * core_weight, uz = dq * gate_weight;
        auto mx = (ux * nx).mean(1, true), mz = (uz * nz).mean(1, true),
             px = ux - ux.mean(1, true) - nx * mx, pz = uz - uz.mean(1, true) - nz * mz;
        auto dw = da * nx, db = da, dv = dq * nz, dc = dq;
        if (second_order) {
            auto tx = (nx * direction_core).mean(1, true), tz = (nz * direction_gate).mean(1, true);
            auto dnx = rx * (direction_core - direction_core.mean(1, true) - nx * tx),
                 dnz = rz * (direction_gate - direction_gate.mean(1, true) - nz * tz);
            auto ad = dnx * core_weight + nx * direction_core_weight + direction_core_bias,
                 qd = dnz * gate_weight + nz * direction_gate_weight + direction_gate_bias;
            auto dda = grad * (s * (1 - s) * (2 + a * (1 - 2 * s)) * t * ad + ds * dt * qd),
                 ddq = grad * (ds * dt * ad + f * dt * (1 - 2 * t) * qd);
            auto dux = dda * core_weight + da * direction_core_weight,
                 duz = ddq * gate_weight + dq * direction_gate_weight;
            grad_core.copy_(rx * (dux - dux.mean(1, true) - dnx * mx -
                                     nx * (dux * nx + ux * dnx).mean(1, true)) -
                            rx * rx * tx * px);
            grad_gate.copy_(rz * (duz - duz.mean(1, true) - dnz * mz -
                                     nz * (duz * nz + uz * dnz).mean(1, true)) -
                            rz * rz * tz * pz);
            grad_grad.copy_(ds * t * ad + f * dt * qd);
            dw = dda * nx + da * dnx;
            db = dda;
            dv = ddq * nz + dq * dnz;
            dc = ddq;
        } else {
            grad_core.copy_(rx * px);
            grad_gate.copy_(rz * pz);
        }
        if (parameter_grads) {
            partial.zero_();
            partial.select(0, 0).copy_(at::stack({dw.sum(0), db.sum(0), dv.sum(0), dc.sum(0)}));
        }
        return;
    }
    if (channels > 768 && parameter_grads) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gate_kernel<32, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
        C10_CUDA_CHECK(cudaFuncSetAttribute(gate_kernel<32, false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    }
#define LAUNCH_GATE_DERIVATIVES(K, S, P)                                                           \
    gate_kernel<K, false, S, P><<<(num_rows + rows_per_block - 1) / rows_per_block,                \
        dim3(32, rows_per_block), P ? 4 * channels * rows_per_block * sizeof(float) : 0,           \
        at::cuda::getCurrentCUDAStream()>>>(core.data_ptr<float>(), gate.data_ptr<float>(),        \
        core_weight.data_ptr<float>(), core_bias.data_ptr<float>(), gate_weight.data_ptr<float>(), \
        gate_bias.data_ptr<float>(), grad.data_ptr<float>(), direction_core.data_ptr<float>(),     \
        direction_gate.data_ptr<float>(), direction_core_weight.data_ptr<float>(),                 \
        direction_core_bias.data_ptr<float>(), direction_gate_weight.data_ptr<float>(),            \
        direction_gate_bias.data_ptr<float>(), grad_core.data_ptr<float>(),                        \
        grad_gate.data_ptr<float>(), grad_grad.data_ptr<float>(), partial.data_ptr<float>(),       \
        num_rows, channels, eps, 1.f / static_cast<float>(channels), channels, 1, channels, 1)
#define DISPATCH_GATE_DERIVATIVES(K)                                                               \
    if (second_order) {                                                                            \
        LAUNCH_GATE_DERIVATIVES(K, true, true);                                                    \
    } else if (parameter_grads) {                                                                  \
        LAUNCH_GATE_DERIVATIVES(K, false, true);                                                   \
    } else {                                                                                       \
        LAUNCH_GATE_DERIVATIVES(K, false, false);                                                  \
    }
    if (channels <= 32) {
        DISPATCH_GATE_DERIVATIVES(1);
    } else if (channels <= 64) {
        DISPATCH_GATE_DERIVATIVES(2);
    } else if (channels <= 128) {
        DISPATCH_GATE_DERIVATIVES(4);
    } else if (channels <= 256) {
        DISPATCH_GATE_DERIVATIVES(8);
    } else if (channels <= 512) {
        DISPATCH_GATE_DERIVATIVES(16);
    } else {
        DISPATCH_GATE_DERIVATIVES(32);
    }
#undef DISPATCH_GATE_DERIVATIVES
#undef LAUNCH_GATE_DERIVATIVES
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gated_parameter_reduce(const at::Tensor &partial, const at::Tensor &output, int64_t num_rows,
    int64_t channels, int64_t rows_per_block) {
    const c10::cuda::CUDAGuard guard(partial.device());
    if (output.numel())
        gate_reduce_kernel<<<dim3(output.numel() / (4 * channels), 4 * channels / 32), dim3(32, 8),
            0, at::cuda::getCurrentCUDAStream()>>>(partial.data_ptr<float>(),
            output.data_ptr<float>(), num_rows, channels, rows_per_block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
            name=f"matris_gated_{key}",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--extended-lambda"],
            verbose=False,
        )
        extension.register_autograd()
        _EXTENSION = extension
    return _EXTENSION


_PROJECTION_EXTENSION = None


PROJECTION_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

std::vector<torch::Tensor> refine_line_project_combine_silu_backward_atom_reduce(
    const torch::Tensor &grad_core_act, const torch::Tensor &grad_gate_act,
    const torch::Tensor &core_edge, const torch::Tensor &core_atom,
    const torch::Tensor &core_target, const torch::Tensor &core_source,
    const torch::Tensor &core_bias, const torch::Tensor &gate_edge, const torch::Tensor &gate_atom,
    const torch::Tensor &gate_target, const torch::Tensor &gate_source,
    const torch::Tensor &gate_bias, const torch::Tensor &atom_index,
    const torch::Tensor &source_index, const torch::Tensor &target_index, int64_t atom_rows,
    int64_t node_rows);
std::vector<torch::Tensor> attn_line_project_combine_silu_backward(
    const torch::Tensor &grad_core_act, const torch::Tensor &grad_gate_act,
    const torch::Tensor &core_edge, const torch::Tensor &core_target,
    const torch::Tensor &core_source, const torch::Tensor &core_bias,
    const torch::Tensor &gate_edge, const torch::Tensor &gate_target,
    const torch::Tensor &gate_source, const torch::Tensor &gate_bias,
    const torch::Tensor &source_index, const torch::Tensor &target_index, int64_t node_rows);
void project_forward(const at::Tensor &core_edge, const std::optional<at::Tensor> &core_atom,
    const at::Tensor &core_target, const at::Tensor &core_source, const at::Tensor &core_bias,
    const at::Tensor &gate_edge, const std::optional<at::Tensor> &gate_atom,
    const at::Tensor &gate_target, const at::Tensor &gate_source, const at::Tensor &gate_bias,
    const std::optional<at::Tensor> &atom_index, const at::Tensor &source_index,
    const at::Tensor &target_index, const at::Tensor &core_output, const at::Tensor &gate_output,
    int64_t num_rows, int64_t channels, bool with_atom);
void project_double_backward(const at::Tensor &edge, const std::optional<at::Tensor> &atom,
    const at::Tensor &target, const at::Tensor &source, const at::Tensor &bias,
    const at::Tensor &grad, const at::Tensor &direction_edge,
    const std::optional<at::Tensor> &direction_atom, const at::Tensor &direction_target,
    const at::Tensor &direction_source, const at::Tensor &direction_bias,
    const std::optional<at::Tensor> &atom_index, const at::Tensor &source_index,
    const at::Tensor &target_index, const at::Tensor &grad_edge,
    const std::optional<at::Tensor> &grad_atom, const at::Tensor &grad_target,
    const at::Tensor &grad_source, const at::Tensor &grad_grad, int64_t num_rows, int64_t channels,
    bool with_atom);

namespace matris_ops {
using Tensor = at::Tensor;
using OptionalTensor = std::optional<Tensor>;

std::tuple<Tensor, OptionalTensor, Tensor, Tensor, Tensor, Tensor, OptionalTensor, Tensor, Tensor,
    Tensor>
project_backward(const Tensor &gc, const Tensor &gg, const Tensor &ce, const OptionalTensor &ca,
    const Tensor &ct, const Tensor &cs, const Tensor &cb, const Tensor &ge,
    const OptionalTensor &ga, const Tensor &gt, const Tensor &gs, const Tensor &gb,
    const OptionalTensor &ai, const Tensor &si, const Tensor &ti) {
    if (!ca) {
        auto grads = attn_line_project_combine_silu_backward(
            gc.contiguous(), gg.contiguous(), ce, ct, cs, cb, ge, gt, gs, gb, si, ti, ct.size(0));
        return {grads[0], std::nullopt, grads[1], grads[2], grads[0].sum(0), grads[3], std::nullopt,
            grads[4], grads[5], grads[3].sum(0)};
    }
    auto grads =
        refine_line_project_combine_silu_backward_atom_reduce(gc.contiguous(), gg.contiguous(), ce,
            *ca, ct, cs, cb, ge, *ga, gt, gs, gb, *ai, si, ti, ca->size(0), ct.size(0));
    return {grads[0], grads[1], grads[2], grads[3], grads[0].sum(0), grads[4], grads[5], grads[6],
        grads[7], grads[4].sum(0)};
}

std::tuple<Tensor, OptionalTensor, Tensor, Tensor, Tensor, Tensor, OptionalTensor, Tensor, Tensor,
    Tensor>
project_backward_meta(const Tensor &gc, const Tensor &gg, const Tensor &ce,
    const OptionalTensor &ca, const Tensor &ct, const Tensor &cs, const Tensor &cb,
    const Tensor &ge, const OptionalTensor &ga, const Tensor &gt, const Tensor &gs,
    const Tensor &gb, const OptionalTensor &ai, const Tensor &si, const Tensor &ti) {
    return {at::empty_symint(ce.sym_sizes(), ce.options()),
        ca ? OptionalTensor(at::empty_symint(ca->sym_sizes(), ca->options())) : std::nullopt,
        at::empty_symint(ct.sym_sizes(), ct.options()),
        at::empty_symint(cs.sym_sizes(), cs.options()),
        at::empty_symint(cb.sym_sizes(), cb.options()),
        at::empty_symint(ge.sym_sizes(), ge.options()),
        ga ? OptionalTensor(at::empty_symint(ga->sym_sizes(), ga->options())) : std::nullopt,
        at::empty_symint(gt.sym_sizes(), gt.options()),
        at::empty_symint(gs.sym_sizes(), gs.options()),
        at::empty_symint(gb.sym_sizes(), gb.options())};
}

std::tuple<Tensor, Tensor, Tensor, OptionalTensor, Tensor, Tensor, Tensor, Tensor, OptionalTensor,
    Tensor, Tensor, Tensor>
project_double_backward(const Tensor &gc, const Tensor &gg, const Tensor &ce,
    const OptionalTensor &ca, const Tensor &ct, const Tensor &cs, const Tensor &cb,
    const Tensor &ge, const OptionalTensor &ga, const Tensor &gt, const Tensor &gs,
    const Tensor &gb, const OptionalTensor &ai, const Tensor &si, const Tensor &ti,
    const Tensor &hce, const OptionalTensor &hca, const Tensor &hct, const Tensor &hcs,
    const Tensor &hcb, const Tensor &hge, const OptionalTensor &hga, const Tensor &hgt,
    const Tensor &hgs, const Tensor &hgb) {
    auto dce = at::empty_like(ce), dcg = at::empty_like(ce);
    auto dct = at::zeros_like(ct), dcs = at::zeros_like(cs);
    OptionalTensor dca = ca ? OptionalTensor(at::zeros_like(*ca)) : std::nullopt;
    ::project_double_backward(ce, ca, ct, cs, cb, gc.contiguous(), hce.contiguous(),
        hca ? OptionalTensor(hca->contiguous()) : std::nullopt, hct.contiguous(), hcs.contiguous(),
        hcb.contiguous(), ai, si, ti, dce, dca, dct, dcs, dcg, ce.size(0), ce.size(1),
        ca.has_value());
    auto dcb = dce.sum(0);
    auto dge = at::empty_like(ge), dgg = at::empty_like(ge);
    auto dgt = at::zeros_like(gt), dgs = at::zeros_like(gs);
    OptionalTensor dga = ga ? OptionalTensor(at::zeros_like(*ga)) : std::nullopt;
    ::project_double_backward(ge, ga, gt, gs, gb, gg.contiguous(), hge.contiguous(),
        hga ? OptionalTensor(hga->contiguous()) : std::nullopt, hgt.contiguous(), hgs.contiguous(),
        hgb.contiguous(), ai, si, ti, dge, dga, dgt, dgs, dgg, ge.size(0), ge.size(1),
        ga.has_value());
    auto dgb = dge.sum(0);
    return {dcg, dgg, dce, dca, dct, dcs, dcb, dge, dga, dgt, dgs, dgb};
}

std::tuple<Tensor, Tensor, Tensor, OptionalTensor, Tensor, Tensor, Tensor, Tensor, OptionalTensor,
    Tensor, Tensor, Tensor>
project_double_backward_meta(const Tensor &gc, const Tensor &gg, const Tensor &ce,
    const OptionalTensor &ca, const Tensor &ct, const Tensor &cs, const Tensor &cb,
    const Tensor &ge, const OptionalTensor &ga, const Tensor &gt, const Tensor &gs,
    const Tensor &gb, const OptionalTensor &ai, const Tensor &si, const Tensor &ti,
    const Tensor &hce, const OptionalTensor &hca, const Tensor &hct, const Tensor &hcs,
    const Tensor &hcb, const Tensor &hge, const OptionalTensor &hga, const Tensor &hgt,
    const Tensor &hgs, const Tensor &hgb) {
    return {at::empty_symint(gc.sym_sizes(), gc.options()),
        at::empty_symint(gg.sym_sizes(), gg.options()),
        at::empty_symint(ce.sym_sizes(), ce.options()),
        ca ? OptionalTensor(at::empty_symint(ca->sym_sizes(), ca->options())) : std::nullopt,
        at::empty_symint(ct.sym_sizes(), ct.options()),
        at::empty_symint(cs.sym_sizes(), cs.options()),
        at::empty_symint(cb.sym_sizes(), cb.options()),
        at::empty_symint(ge.sym_sizes(), ge.options()),
        ga ? OptionalTensor(at::empty_symint(ga->sym_sizes(), ga->options())) : std::nullopt,
        at::empty_symint(gt.sym_sizes(), gt.options()),
        at::empty_symint(gs.sym_sizes(), gs.options()),
        at::empty_symint(gb.sym_sizes(), gb.options())};
}

std::tuple<Tensor, Tensor> refine_project(const Tensor &core_edge, const Tensor &core_atom,
    const Tensor &core_target, const Tensor &core_source, const Tensor &core_bias,
    const Tensor &gate_edge, const Tensor &gate_atom, const Tensor &gate_target,
    const Tensor &gate_source, const Tensor &gate_bias, const Tensor &atom_index,
    const Tensor &source_index, const Tensor &target_index) {
    auto source = source_index.contiguous(), target = target_index.contiguous();
    auto core_act = at::empty_like(core_edge), gate_act = at::empty_like(gate_edge);
    auto atom = atom_index.contiguous();
    project_forward(core_edge, core_atom, core_target, core_source, core_bias, gate_edge, gate_atom,
        gate_target, gate_source, gate_bias, atom, source, target, core_act, gate_act,
        core_edge.size(0), core_edge.size(1), true);
    return {core_act, gate_act};
}

std::tuple<Tensor, Tensor> refine_project_meta(const Tensor &core_edge, const Tensor &core_atom,
    const Tensor &core_target, const Tensor &core_source, const Tensor &core_bias,
    const Tensor &gate_edge, const Tensor &gate_atom, const Tensor &gate_target,
    const Tensor &gate_source, const Tensor &gate_bias, const Tensor &atom_index,
    const Tensor &source_index, const Tensor &target_index) {
    return {at::empty_symint(core_edge.sym_sizes(), core_edge.options()),
        at::empty_symint(gate_edge.sym_sizes(), gate_edge.options())};
}

std::tuple<Tensor, Tensor> attn_project(const Tensor &core_edge, const Tensor &core_target,
    const Tensor &core_source, const Tensor &core_bias, const Tensor &gate_edge,
    const Tensor &gate_target, const Tensor &gate_source, const Tensor &gate_bias,
    const Tensor &source_index, const Tensor &target_index) {
    auto source = source_index.contiguous(), target = target_index.contiguous();
    auto core_act = at::empty_like(core_edge), gate_act = at::empty_like(gate_edge);
    project_forward(core_edge, core_edge, core_target, core_source, core_bias, gate_edge, gate_edge,
        gate_target, gate_source, gate_bias, source, source, target, core_act, gate_act,
        core_edge.size(0), core_edge.size(1), false);
    return {core_act, gate_act};
}

std::tuple<Tensor, Tensor> attn_project_meta(const Tensor &core_edge, const Tensor &core_target,
    const Tensor &core_source, const Tensor &core_bias, const Tensor &gate_edge,
    const Tensor &gate_target, const Tensor &gate_source, const Tensor &gate_bias,
    const Tensor &source_index, const Tensor &target_index) {
    return {at::empty_symint(core_edge.sym_sizes(), core_edge.options()),
        at::empty_symint(gate_edge.sym_sizes(), gate_edge.options())};
}
} // namespace matris_ops

TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def("project_backward(Tensor gc, Tensor gg, Tensor ce, Tensor? ca, Tensor ct, Tensor cs, "
          "Tensor cb, Tensor ge, Tensor? ga, Tensor gt, Tensor gs, Tensor gb, Tensor? ai, Tensor "
          "si, Tensor ti) -> (Tensor, Tensor?, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor, "
          "Tensor, Tensor)");
    m.def("project_double_backward(Tensor gc, Tensor gg, Tensor ce, Tensor? ca, Tensor ct, Tensor "
          "cs, Tensor cb, Tensor ge, Tensor? ga, Tensor gt, Tensor gs, Tensor gb, Tensor? ai, "
          "Tensor si, Tensor ti, Tensor hce, Tensor? hca, Tensor hct, Tensor hcs, Tensor hcb, "
          "Tensor hge, Tensor? hga, Tensor hgt, Tensor hgs, Tensor hgb) -> (Tensor, Tensor, "
          "Tensor, Tensor?, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor, Tensor, Tensor)");
    m.def("refine_project(Tensor core_edge, Tensor core_atom, Tensor core_target, Tensor "
          "core_source, Tensor core_bias, Tensor gate_edge, Tensor gate_atom, Tensor gate_target, "
          "Tensor gate_source, Tensor gate_bias, Tensor atom_index, Tensor source_index, Tensor "
          "target_index) -> (Tensor, Tensor)");
    m.def("attn_project(Tensor core_edge, Tensor core_target, Tensor core_source, Tensor "
          "core_bias, Tensor gate_edge, Tensor gate_target, Tensor gate_source, Tensor gate_bias, "
          "Tensor source_index, Tensor target_index) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("project_backward", TORCH_FN(matris_ops::project_backward));
    m.impl("project_double_backward", TORCH_FN(matris_ops::project_double_backward));
    m.impl("refine_project", TORCH_FN(matris_ops::refine_project));
    m.impl("attn_project", TORCH_FN(matris_ops::attn_project));
}

TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("project_backward", TORCH_FN(matris_ops::project_backward_meta));
    m.impl("project_double_backward", TORCH_FN(matris_ops::project_double_backward_meta));
    m.impl("refine_project", TORCH_FN(matris_ops::refine_project_meta));
    m.impl("attn_project", TORCH_FN(matris_ops::attn_project_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto register_gradient = py::module_::import("torch.library").attr("register_autograd");
        auto gated_mlp = py::module_::import("matris.model.op.gated_mlp");
        register_gradient("matris::project_backward",
            gated_mlp.attr("ProjectBackwardFunction").attr("backward"),
            py::arg("setup_context") =
                gated_mlp.attr("ProjectBackwardFunction").attr("setup_context"));
        register_gradient("matris::refine_project",
            gated_mlp.attr("RefineProjectFunction").attr("backward"),
            py::arg("setup_context") =
                gated_mlp.attr("RefineProjectFunction").attr("setup_context"));
        register_gradient("matris::attn_project",
            gated_mlp.attr("AttnProjectFunction").attr("backward"),
            py::arg("setup_context") = gated_mlp.attr("AttnProjectFunction").attr("setup_context"));
    });
    m.def("attn_line_project_combine_silu_backward", &attn_line_project_combine_silu_backward,
        "attn_line_project_combine_silu_backward");
    m.def("project_forward", &project_forward);
    m.def("project_double_backward", &project_double_backward);
}
"""


PROJECTION_CUDA_SOURCE = COMMON_SOURCE + r"""

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kProjectionThreads = 256;

__device__ __forceinline__ float sigmoidf_stable(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float silu_backward_from_x(float x) {
    float sig = sigmoidf_stable(x);
    return sig * (1.0f + x * (1.0f - sig));
}

__global__ void refine_line_project_combine_silu_backward_no_atom_kernel(
    const float *__restrict__ grad_core_act, const float *__restrict__ grad_gate_act,
    const float *__restrict__ core_edge, const float *__restrict__ core_atom,
    const float *__restrict__ core_target, const float *__restrict__ core_source,
    const float *__restrict__ core_bias, const float *__restrict__ gate_edge,
    const float *__restrict__ gate_atom, const float *__restrict__ gate_target,
    const float *__restrict__ gate_source, const float *__restrict__ gate_bias,
    const int64_t *__restrict__ atom_index, const int64_t *__restrict__ source_index,
    const int64_t *__restrict__ target_index, float *__restrict__ grad_core_edge,
    float *__restrict__ grad_core_target, float *__restrict__ grad_core_source,
    float *__restrict__ grad_gate_edge, float *__restrict__ grad_gate_target,
    float *__restrict__ grad_gate_source, int64_t rows, int kDim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t row = idx / kDim;
    int dim = static_cast<int>(idx - row * kDim);
    int64_t atom = atom_index[row];
    int64_t source = source_index[row];
    int64_t target = target_index[row];
    float core_hidden = core_edge[idx] + core_atom[atom * kDim + dim] +
                        core_target[target * kDim + dim] + core_source[source * kDim + dim] +
                        core_bias[dim];
    float gate_hidden = gate_edge[idx] + gate_atom[atom * kDim + dim] +
                        gate_target[target * kDim + dim] + gate_source[source * kDim + dim] +
                        gate_bias[dim];
    float gc = grad_core_act[idx] * silu_backward_from_x(core_hidden);
    float gg = grad_gate_act[idx] * silu_backward_from_x(gate_hidden);

    grad_core_edge[idx] = gc;
    grad_gate_edge[idx] = gg;
    atomicAdd(grad_core_target + target * kDim + dim, gc);
    atomicAdd(grad_core_source + source * kDim + dim, gc);
    atomicAdd(grad_gate_target + target * kDim + dim, gg);
    atomicAdd(grad_gate_source + source * kDim + dim, gg);
}

__device__ __forceinline__ int64_t lower_bound_atom_index(
    const int64_t *__restrict__ atom_index, int64_t rows, int64_t value) {
    int64_t lo = 0;
    int64_t hi = rows;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (atom_index[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void refine_line_project_combine_silu_atom_reduce_kernel(
    const float *__restrict__ grad_core_edge, const float *__restrict__ grad_gate_edge,
    const int64_t *__restrict__ atom_index, float *__restrict__ grad_core_atom,
    float *__restrict__ grad_gate_atom, int64_t rows, int64_t atom_rows, int kDim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = atom_rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t atom = idx / kDim;
    int dim = static_cast<int>(idx - atom * kDim);
    int64_t start = lower_bound_atom_index(atom_index, rows, atom);
    int64_t end = lower_bound_atom_index(atom_index, rows, atom + 1);
    float sum_core = 0.0f;
    float sum_gate = 0.0f;
    for (int64_t row = start; row < end; ++row) {
        int64_t offset = row * kDim + dim;
        sum_core += grad_core_edge[offset];
        sum_gate += grad_gate_edge[offset];
    }
    grad_core_atom[idx] = sum_core;
    grad_gate_atom[idx] = sum_gate;
}

__global__ void attn_line_project_combine_silu_backward_kernel(
    const float *__restrict__ grad_core_act, const float *__restrict__ grad_gate_act,
    const float *__restrict__ core_edge, const float *__restrict__ core_target,
    const float *__restrict__ core_source, const float *__restrict__ core_bias,
    const float *__restrict__ gate_edge, const float *__restrict__ gate_target,
    const float *__restrict__ gate_source, const float *__restrict__ gate_bias,
    const int64_t *__restrict__ source_index, const int64_t *__restrict__ target_index,
    float *__restrict__ grad_core_edge, float *__restrict__ grad_core_target,
    float *__restrict__ grad_core_source, float *__restrict__ grad_gate_edge,
    float *__restrict__ grad_gate_target, float *__restrict__ grad_gate_source, int64_t rows,
    int kDim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * kDim;
    if (idx >= total) {
        return;
    }
    int64_t row = idx / kDim;
    int dim = static_cast<int>(idx - row * kDim);
    int64_t source = source_index[row];
    int64_t target = target_index[row];
    float core_hidden = core_edge[idx] + core_target[target * kDim + dim] +
                        core_source[source * kDim + dim] + core_bias[dim];
    float gate_hidden = gate_edge[idx] + gate_target[target * kDim + dim] +
                        gate_source[source * kDim + dim] + gate_bias[dim];
    float gc = grad_core_act[idx] * silu_backward_from_x(core_hidden);
    float gg = grad_gate_act[idx] * silu_backward_from_x(gate_hidden);

    grad_core_edge[idx] = gc;
    grad_gate_edge[idx] = gg;
    atomicAdd(grad_core_target + target * kDim + dim, gc);
    atomicAdd(grad_core_source + source * kDim + dim, gc);
    atomicAdd(grad_gate_target + target * kDim + dim, gg);
    atomicAdd(grad_gate_source + source * kDim + dim, gg);
}

void check_projected(const at::Tensor &tensor, const char *name, int kDim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == kDim && kDim > 0 && kDim % 32 == 0, name,
        " must have shape [N, D], D a positive multiple of 32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_index_1d(const at::Tensor &tensor, int64_t rows, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == rows, name, " must have shape [rows]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::vector<at::Tensor> refine_line_project_combine_silu_backward_atom_reduce(
    const at::Tensor &grad_core_act, const at::Tensor &grad_gate_act, const at::Tensor &core_edge,
    const at::Tensor &core_atom, const at::Tensor &core_target, const at::Tensor &core_source,
    const at::Tensor &core_bias, const at::Tensor &gate_edge, const at::Tensor &gate_atom,
    const at::Tensor &gate_target, const at::Tensor &gate_source, const at::Tensor &gate_bias,
    const at::Tensor &atom_index, const at::Tensor &source_index, const at::Tensor &target_index,
    int64_t atom_rows, int64_t node_rows) {
    const c10::cuda::CUDAGuard guard(core_edge.device());
    const int kDim = core_edge.size(1);
    check_projected(grad_core_act, "grad_core_act", kDim);
    check_projected(grad_gate_act, "grad_gate_act", kDim);
    check_projected(core_edge, "core_edge", kDim);
    check_projected(core_atom, "core_atom", kDim);
    check_projected(core_target, "core_target", kDim);
    check_projected(core_source, "core_source", kDim);
    TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == at::kFloat &&
                    core_bias.is_contiguous() && core_bias.numel() == kDim,
        "core_bias must be contiguous CUDA float32 [D]");
    check_projected(gate_edge, "gate_edge", kDim);
    check_projected(gate_atom, "gate_atom", kDim);
    check_projected(gate_target, "gate_target", kDim);
    check_projected(gate_source, "gate_source", kDim);
    TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == at::kFloat &&
                    gate_bias.is_contiguous() && gate_bias.numel() == kDim,
        "gate_bias must be contiguous CUDA float32 [D]");
    int64_t rows = core_edge.size(0);
    TORCH_CHECK(grad_core_act.size(0) == rows && grad_gate_act.size(0) == rows,
        "grad row count must match projected rows");
    TORCH_CHECK(gate_edge.size(0) == rows, "gate_edge row count must match core_edge");
    TORCH_CHECK(
        core_target.size(0) == core_source.size(0), "core target/source row count must match");
    TORCH_CHECK(
        gate_target.size(0) == gate_source.size(0), "gate target/source row count must match");
    TORCH_CHECK(core_target.size(0) == gate_target.size(0),
        "core/gate node projection row count must match");
    TORCH_CHECK(
        core_atom.size(0) == gate_atom.size(0), "core/gate atom projection row count must match");
    TORCH_CHECK(atom_rows >= 0 && node_rows >= 0, "atom_rows/node_rows must be non-negative");
    check_index_1d(atom_index, rows, "atom_index");
    check_index_1d(source_index, rows, "source_index");
    check_index_1d(target_index, rows, "target_index");

    auto grad_core_edge = at::empty_like(core_edge);
    auto grad_gate_edge = at::empty_like(gate_edge);
    auto grad_core_atom = at::empty({atom_rows, kDim}, core_edge.options());
    auto grad_core_target = at::zeros({node_rows, kDim}, core_edge.options());
    auto grad_core_source = at::zeros({node_rows, kDim}, core_edge.options());
    auto grad_gate_atom = at::empty({atom_rows, kDim}, gate_edge.options());
    auto grad_gate_target = at::zeros({node_rows, kDim}, gate_edge.options());
    auto grad_gate_source = at::zeros({node_rows, kDim}, gate_edge.options());

    int64_t edge_total = rows * kDim;
    int edge_blocks = static_cast<int>((edge_total + kProjectionThreads - 1) / kProjectionThreads);
    refine_line_project_combine_silu_backward_no_atom_kernel<<<edge_blocks, kProjectionThreads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(grad_core_act.data_ptr<float>(),
        grad_gate_act.data_ptr<float>(), core_edge.data_ptr<float>(), core_atom.data_ptr<float>(),
        core_target.data_ptr<float>(), core_source.data_ptr<float>(), core_bias.data_ptr<float>(),
        gate_edge.data_ptr<float>(), gate_atom.data_ptr<float>(), gate_target.data_ptr<float>(),
        gate_source.data_ptr<float>(), gate_bias.data_ptr<float>(), atom_index.data_ptr<int64_t>(),
        source_index.data_ptr<int64_t>(), target_index.data_ptr<int64_t>(),
        grad_core_edge.data_ptr<float>(), grad_core_target.data_ptr<float>(),
        grad_core_source.data_ptr<float>(), grad_gate_edge.data_ptr<float>(),
        grad_gate_target.data_ptr<float>(), grad_gate_source.data_ptr<float>(), rows, kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t atom_total = atom_rows * kDim;
    int atom_blocks = static_cast<int>((atom_total + kProjectionThreads - 1) / kProjectionThreads);
    refine_line_project_combine_silu_atom_reduce_kernel<<<atom_blocks, kProjectionThreads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(grad_core_edge.data_ptr<float>(),
        grad_gate_edge.data_ptr<float>(), atom_index.data_ptr<int64_t>(),
        grad_core_atom.data_ptr<float>(), grad_gate_atom.data_ptr<float>(), rows, atom_rows, kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_core_edge,
        grad_core_atom,
        grad_core_target,
        grad_core_source,
        grad_gate_edge,
        grad_gate_atom,
        grad_gate_target,
        grad_gate_source,
    };
}

std::vector<at::Tensor> attn_line_project_combine_silu_backward(const at::Tensor &grad_core_act,
    const at::Tensor &grad_gate_act, const at::Tensor &core_edge, const at::Tensor &core_target,
    const at::Tensor &core_source, const at::Tensor &core_bias, const at::Tensor &gate_edge,
    const at::Tensor &gate_target, const at::Tensor &gate_source, const at::Tensor &gate_bias,
    const at::Tensor &source_index, const at::Tensor &target_index, int64_t node_rows) {
    const c10::cuda::CUDAGuard guard(core_edge.device());
    const int kDim = core_edge.size(1);
    check_projected(grad_core_act, "grad_core_act", kDim);
    check_projected(grad_gate_act, "grad_gate_act", kDim);
    check_projected(core_edge, "core_edge", kDim);
    check_projected(core_target, "core_target", kDim);
    check_projected(core_source, "core_source", kDim);
    TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == at::kFloat &&
                    core_bias.is_contiguous() && core_bias.numel() == kDim,
        "core_bias must be contiguous CUDA float32 [D]");
    check_projected(gate_edge, "gate_edge", kDim);
    check_projected(gate_target, "gate_target", kDim);
    check_projected(gate_source, "gate_source", kDim);
    TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == at::kFloat &&
                    gate_bias.is_contiguous() && gate_bias.numel() == kDim,
        "gate_bias must be contiguous CUDA float32 [D]");
    int64_t rows = core_edge.size(0);
    TORCH_CHECK(grad_core_act.size(0) == rows && grad_gate_act.size(0) == rows,
        "grad row count must match projected rows");
    TORCH_CHECK(gate_edge.size(0) == rows, "gate_edge row count must match core_edge");
    TORCH_CHECK(
        core_target.size(0) == core_source.size(0), "core target/source row count must match");
    TORCH_CHECK(core_target.size(0) == gate_target.size(0),
        "core/gate node projection row count must match");
    TORCH_CHECK(
        gate_target.size(0) == gate_source.size(0), "gate target/source row count must match");
    TORCH_CHECK(node_rows >= 0, "node_rows must be non-negative");
    check_index_1d(source_index, rows, "source_index");
    check_index_1d(target_index, rows, "target_index");

    auto grad_core_edge = at::empty_like(core_edge);
    auto grad_gate_edge = at::empty_like(gate_edge);
    auto grad_core_target = at::zeros({node_rows, kDim}, core_edge.options());
    auto grad_core_source = at::zeros({node_rows, kDim}, core_edge.options());
    auto grad_gate_target = at::zeros({node_rows, kDim}, gate_edge.options());
    auto grad_gate_source = at::zeros({node_rows, kDim}, gate_edge.options());
    int64_t total = rows * kDim;
    int blocks = static_cast<int>((total + kProjectionThreads - 1) / kProjectionThreads);
    attn_line_project_combine_silu_backward_kernel<<<blocks, kProjectionThreads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(grad_core_act.data_ptr<float>(),
        grad_gate_act.data_ptr<float>(), core_edge.data_ptr<float>(), core_target.data_ptr<float>(),
        core_source.data_ptr<float>(), core_bias.data_ptr<float>(), gate_edge.data_ptr<float>(),
        gate_target.data_ptr<float>(), gate_source.data_ptr<float>(), gate_bias.data_ptr<float>(),
        source_index.data_ptr<int64_t>(), target_index.data_ptr<int64_t>(),
        grad_core_edge.data_ptr<float>(), grad_core_target.data_ptr<float>(),
        grad_core_source.data_ptr<float>(), grad_gate_edge.data_ptr<float>(),
        grad_gate_target.data_ptr<float>(), grad_gate_source.data_ptr<float>(), rows, kDim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_core_edge,
        grad_core_target,
        grad_core_source,
        grad_gate_edge,
        grad_gate_target,
        grad_gate_source,
    };
}

using namespace matris_cuda;

namespace {
template <typename I, bool Atom>
__global__ void project_forward_kernel(const float *__restrict__ ce, const float *__restrict__ ca,
    const float *__restrict__ ct, const float *__restrict__ cs, const float *__restrict__ cb,
    const float *__restrict__ ge, const float *__restrict__ ga, const float *__restrict__ gt,
    const float *__restrict__ gs, const float *__restrict__ gb, const I *ai, const I *si,
    const I *ti, float *__restrict__ co, float *__restrict__ go, int64_t n, int d, int shift) {
    int width = d / 4;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int c = linear & ((1 << shift) - 1);
    int64_t row = linear >> shift;
    if (row >= n || c >= width)
        return;
    int64_t j = row * width + c;
    int64_t s = (int64_t)si[row] * width + c, t = (int64_t)ti[row] * width + c;
    float4 x = __ldcs(reinterpret_cast<const float4 *>(ce) + j);
    float4 z = __ldcs(reinterpret_cast<const float4 *>(ge) + j);
    if constexpr (Atom) {
        float4 v = reinterpret_cast<const float4 *>(ca)[(int64_t)ai[row] * width + c];
        x.x += v.x;
        x.y += v.y;
        x.z += v.z;
        x.w += v.w;
    }
    if constexpr (Atom) {
        float4 v = reinterpret_cast<const float4 *>(ga)[(int64_t)ai[row] * width + c];
        z.x += v.x;
        z.y += v.y;
        z.z += v.z;
        z.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(ct)[t];
        x.x += v.x;
        x.y += v.y;
        x.z += v.z;
        x.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(gt)[t];
        z.x += v.x;
        z.y += v.y;
        z.z += v.z;
        z.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(cs)[s];
        x.x += v.x;
        x.y += v.y;
        x.z += v.z;
        x.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(gs)[s];
        z.x += v.x;
        z.y += v.y;
        z.z += v.z;
        z.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(cb)[c];
        x.x += v.x;
        x.y += v.y;
        x.z += v.z;
        x.w += v.w;
    }
    {
        float4 v = reinterpret_cast<const float4 *>(gb)[c];
        z.x += v.x;
        z.y += v.y;
        z.z += v.z;
        z.w += v.w;
    }
    x.x *= sigmoid(x.x);
    x.y *= sigmoid(x.y);
    x.z *= sigmoid(x.z);
    x.w *= sigmoid(x.w);
    z.x *= sigmoid(z.x);
    z.y *= sigmoid(z.y);
    z.z *= sigmoid(z.z);
    z.w *= sigmoid(z.w);
    __stcs(reinterpret_cast<float4 *>(co) + j, x);
    __stcs(reinterpret_cast<float4 *>(go) + j, z);
}

__global__ void project_second_kernel(const float *e, const float *a, const float *t,
    const float *s, const float *b, const float *g, const float *he, const float *ha,
    const float *ht, const float *hs, const float *hb, Index ai, Index si, Index ti, float *de,
    float *da, float *dt, float *ds, float *dg, int64_t n, int d, bool atom) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n * d)
        return;
    int c = j % d;
    int64_t sj = si[j / d] * d + c, tj = ti[j / d] * d + c, aj = 0;
    float x = e[j], h = he[j];
    if (atom) {
        aj = ai[j / d] * d + c;
        x += a[aj];
        h += ha[aj];
    }
    x += t[tj];
    x += s[sj];
    x += b[c];
    h += ht[tj] + hs[sj] + hb[c];
    float sig = sigmoid(x), dx = g[j] * h * sig * (1 - sig) * (2 + x * (1 - 2 * sig));
    de[j] = dx;
    dg[j] = h * sig * (1 + x * (1 - sig));
    atomicAdd(dt + tj, dx);
    atomicAdd(ds + sj, dx);
    if (atom)
        atomicAdd(da + aj, dx);
}

} // namespace

void project_forward(const at::Tensor &core_edge, const std::optional<at::Tensor> &core_atom,
    const at::Tensor &core_target, const at::Tensor &core_source, const at::Tensor &core_bias,
    const at::Tensor &gate_edge, const std::optional<at::Tensor> &gate_atom,
    const at::Tensor &gate_target, const at::Tensor &gate_source, const at::Tensor &gate_bias,
    const std::optional<at::Tensor> &atom_index, const at::Tensor &source_index,
    const at::Tensor &target_index, const at::Tensor &core_output, const at::Tensor &gate_output,
    int64_t num_rows, int64_t channels, bool with_atom) {
    const c10::cuda::CUDAGuard guard(core_edge.device());
    int shift = 0;
    while ((1 << shift) < channels / 4)
        ++shift;
    if (!num_rows)
        return;
#define LAUNCH_PROJECT_FORWARD(I, A)                                                               \
    project_forward_kernel<I, A>                                                                   \
        <<<(num_rows * (1 << shift) + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(     \
            core_edge.data_ptr<float>(), core_atom ? core_atom->data_ptr<float>() : nullptr,       \
            core_target.data_ptr<float>(), core_source.data_ptr<float>(),                          \
            core_bias.data_ptr<float>(), gate_edge.data_ptr<float>(),                              \
            gate_atom ? gate_atom->data_ptr<float>() : nullptr, gate_target.data_ptr<float>(),     \
            gate_source.data_ptr<float>(), gate_bias.data_ptr<float>(),                            \
            atom_index ? atom_index->data_ptr<I>() : nullptr, source_index.data_ptr<I>(),          \
            target_index.data_ptr<I>(), core_output.data_ptr<float>(),                             \
            gate_output.data_ptr<float>(), num_rows, channels, shift);
    if (source_index.scalar_type() == at::kLong) {
        if (with_atom) {
            LAUNCH_PROJECT_FORWARD(int64_t, true);
        } else {
            LAUNCH_PROJECT_FORWARD(int64_t, false);
        }
    } else {
        if (with_atom) {
            LAUNCH_PROJECT_FORWARD(int32_t, true);
        } else {
            LAUNCH_PROJECT_FORWARD(int32_t, false);
        }
    }
#undef LAUNCH_PROJECT_FORWARD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_double_backward(const at::Tensor &edge, const std::optional<at::Tensor> &atom,
    const at::Tensor &target, const at::Tensor &source, const at::Tensor &bias,
    const at::Tensor &grad, const at::Tensor &direction_edge,
    const std::optional<at::Tensor> &direction_atom, const at::Tensor &direction_target,
    const at::Tensor &direction_source, const at::Tensor &direction_bias,
    const std::optional<at::Tensor> &atom_index, const at::Tensor &source_index,
    const at::Tensor &target_index, const at::Tensor &grad_edge,
    const std::optional<at::Tensor> &grad_atom, const at::Tensor &grad_target,
    const at::Tensor &grad_source, const at::Tensor &grad_grad, int64_t num_rows, int64_t channels,
    bool with_atom) {
    const c10::cuda::CUDAGuard guard(edge.device());
    if (num_rows)
        project_second_kernel<<<(num_rows * channels + 255) / 256, 256, 0,
            at::cuda::getCurrentCUDAStream()>>>(edge.data_ptr<float>(),
            atom ? atom->data_ptr<float>() : nullptr, target.data_ptr<float>(),
            source.data_ptr<float>(), bias.data_ptr<float>(), grad.data_ptr<float>(),
            direction_edge.data_ptr<float>(),
            direction_atom ? direction_atom->data_ptr<float>() : nullptr,
            direction_target.data_ptr<float>(), direction_source.data_ptr<float>(),
            direction_bias.data_ptr<float>(),
            atom_index ? index_of(*atom_index) : Index{nullptr, false}, index_of(source_index),
            index_of(target_index), grad_edge.data_ptr<float>(),
            grad_atom ? grad_atom->data_ptr<float>() : nullptr, grad_target.data_ptr<float>(),
            grad_source.data_ptr<float>(), grad_grad.data_ptr<float>(), num_rows, channels,
            with_atom);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@torch.compiler.assume_constant_result
def get_projection_extension():
    """Compile this operator family once and reuse its source-addressed cache."""
    global _PROJECTION_EXTENSION
    if _PROJECTION_EXTENSION is None:
        os.environ.setdefault("MAX_JOBS", "4")
        key = hashlib.sha256(
            (
                f"{sys.implementation.cache_tag}:{torch.__version__}:{torch.version.cuda}\n"
                + PROJECTION_CPP_SOURCE
                + "\n"
                + PROJECTION_CUDA_SOURCE
            ).encode()
        ).hexdigest()[:16]
        extension = load_inline(
            name=f"matris_projection_{key}",
            cpp_sources=PROJECTION_CPP_SOURCE,
            cuda_sources=PROJECTION_CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--extended-lambda"],
            verbose=False,
        )
        extension.register_autograd()
        _PROJECTION_EXTENSION = extension
    return _PROJECTION_EXTENSION


class GatedDoubleBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad,
        core,
        gate,
        core_weight,
        core_bias,
        gate_weight,
        gate_bias,
        eps,
        hx,
        hz,
        hp,
    ):
        get_extension()
        return torch.ops.matris.gated_double_backward(
            grad,
            core,
            gate,
            core_weight,
            core_bias,
            gate_weight,
            gate_bias,
            eps,
            hx,
            hz,
            hp,
        )


class GatedBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(grad, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps):
        get_extension()
        return torch.ops.matris.gated_backward(
            grad, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1])
        ctx.eps = inputs[-1]

    @staticmethod
    def backward(ctx, hx, hz, hp):
        dg, dx, dz, parameters = GatedDoubleBackwardFunction.apply(
            *ctx.saved_tensors, ctx.eps, hx, hz, hp
        )
        return (dg, dx, dz, *parameters.unbind(), None)


class GatedInputBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, grad, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
    ):
        get_extension()
        return torch.ops.matris.gated_input_backward(
            grad, core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
        )


class GatedTailFunction(torch.autograd.Function):
    @staticmethod
    def forward(core, gate, core_weight, core_bias, gate_weight, gate_bias, eps):
        get_extension()
        return torch.ops.matris.gated_tail(
            core, gate, core_weight, core_bias, gate_weight, gate_bias, eps
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1])
        ctx.eps = inputs[-1]

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled() or any(ctx.needs_input_grad[2:6]):
            dx, dz, parameters = GatedBackwardFunction.apply(
                grad, *ctx.saved_tensors, ctx.eps
            )
            return (dx, dz, *parameters.unbind(), None)
        dx, dz = GatedInputBackwardFunction.apply(grad, *ctx.saved_tensors, ctx.eps)
        return (dx, dz, None, None, None, None, None)


class ProjectDoubleBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gc,
        gg,
        ce,
        ca,
        ct,
        cs,
        cb,
        ge,
        ga,
        gt,
        gs,
        gb,
        ai,
        si,
        ti,
        hce,
        hca,
        hct,
        hcs,
        hcb,
        hge,
        hga,
        hgt,
        hgs,
        hgb,
    ):
        get_projection_extension()
        return torch.ops.matris.project_double_backward(
            gc,
            gg,
            ce,
            ca,
            ct,
            cs,
            cb,
            ge,
            ga,
            gt,
            gs,
            gb,
            ai,
            si,
            ti,
            hce,
            hca,
            hct,
            hcs,
            hcb,
            hge,
            hga,
            hgt,
            hgs,
            hgb,
        )


class ProjectBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(gc, gg, ce, ca, ct, cs, cb, ge, ga, gt, gs, gb, ai, si, ti):
        get_projection_extension()
        return torch.ops.matris.project_backward(
            gc, gg, ce, ca, ct, cs, cb, ge, ga, gt, gs, gb, ai, si, ti
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, *grads):
        return (
            *ProjectDoubleBackwardFunction.apply(*ctx.saved_tensors, *grads),
            None,
            None,
            None,
        )


class RefineProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        core_edge,
        core_atom,
        core_target,
        core_source,
        core_bias,
        gate_edge,
        gate_atom,
        gate_target,
        gate_source,
        gate_bias,
        atom_index,
        source_index,
        target_index,
    ):
        get_projection_extension()
        return torch.ops.matris.refine_project(
            core_edge,
            core_atom,
            core_target,
            core_source,
            core_bias,
            gate_edge,
            gate_atom,
            gate_target,
            gate_source,
            gate_bias,
            atom_index,
            source_index,
            target_index,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gc, gg):
        return (
            *ProjectBackwardFunction.apply(gc, gg, *ctx.saved_tensors),
            None,
            None,
            None,
        )


class AttnProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        core_edge,
        core_target,
        core_source,
        core_bias,
        gate_edge,
        gate_target,
        gate_source,
        gate_bias,
        source_index,
        target_index,
    ):
        get_projection_extension()
        return torch.ops.matris.attn_project(
            core_edge,
            core_target,
            core_source,
            core_bias,
            gate_edge,
            gate_target,
            gate_source,
            gate_bias,
            source_index,
            target_index,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gc, gg):
        (
            core_edge,
            core_target,
            core_source,
            core_bias,
            gate_edge,
            gate_target,
            gate_source,
            gate_bias,
            source_index,
            target_index,
        ) = ctx.saved_tensors
        grads = ProjectBackwardFunction.apply(
            gc,
            gg,
            core_edge,
            None,
            core_target,
            core_source,
            core_bias,
            gate_edge,
            None,
            gate_target,
            gate_source,
            gate_bias,
            None,
            source_index,
            target_index,
        )
        return (grads[0], *grads[2:6], *grads[7:], None, None)


gated_tail_op = GatedTailFunction.apply


refine_project_op = RefineProjectFunction.apply
attn_project_op = AttnProjectFunction.apply
