"""Per-GEMM TF32 with cuBLAS; no changes to PyTorch's global precision flags."""

import hashlib

import torch
from torch.utils.cpp_extension import load_inline

_EXTENSION = None

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>
at::Tensor tf32_mm_cuda(const at::Tensor&, const at::Tensor&);
TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def("tf32_mm(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("tf32_mm", TORCH_FN(tf32_mm_cuda));
}
TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("tf32_mm", [](const at::Tensor& a, const at::Tensor& b) {
        return at::empty_symint({a.sym_size(0), b.sym_size(1)}, a.options());
    });
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto function = py::module_::import("matris.model.op.gemm").attr("TF32MatmulFunction");
        py::module_::import("torch.library").attr("register_autograd")(
            "matris::tf32_mm", function.attr("backward"),
            py::arg("setup_context") = function.attr("setup_context"));
    });
}
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

at::Tensor tf32_mm_cuda(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "tf32_mm expects compatible matrices");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat
                && a.device() == b.device(), "tf32_mm expects FP32 on one CUDA device");
    const c10::cuda::CUDAGuard guard(a.device());
    // Retain transposed/column-sliced weights without materializing a copy.
    auto left = ((a.stride(1) == 1 && a.stride(0) >= a.size(1)) ||
                 (a.stride(0) == 1 && a.stride(1) >= a.size(0))) ? a : a.contiguous();
    auto right = ((b.stride(1) == 1 && b.stride(0) >= b.size(1)) ||
                  (b.stride(0) == 1 && b.stride(1) >= b.size(0))) ? b : b.contiguous();
    auto out = at::empty({a.size(0), b.size(1)}, a.options());
    if (out.numel() == 0) return out;
    if (a.size(1) == 0) return out.zero_();
    const bool left_row = left.stride(1) == 1;
    const bool right_row = right.stride(1) == 1;
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A B is column-major C^T = B^T A^T.
    // PyTorch's handle carries its current stream and graph-capture workspace.
    auto status = cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),
        right_row ? CUBLAS_OP_N : CUBLAS_OP_T,
        left_row ? CUBLAS_OP_N : CUBLAS_OP_T,
        b.size(1), a.size(0), a.size(1), &alpha,
        right.data_ptr<float>(), CUDA_R_32F,
        std::max<int64_t>(1, right_row ? right.stride(0) : right.stride(1)),
        left.data_ptr<float>(), CUDA_R_32F,
        std::max<int64_t>(1, left_row ? left.stride(0) : left.stride(1)),
        &beta, out.data_ptr<float>(), CUDA_R_32F, std::max<int64_t>(1, b.size(1)),
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "TF32 cuBLAS GEMM failed: ", int(status));
    return out;
}
"""


@torch.compiler.assume_constant_result
def get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        key = hashlib.sha256(
            (
                torch.__version__ + str(torch.version.cuda) + CPP_SOURCE + CUDA_SOURCE
            ).encode()
        ).hexdigest()[:16]
        extension = load_inline(
            name=f"matris_gemm_{key}",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublas"],
        )
        extension.register_autograd()
        _EXTENSION = extension
    return _EXTENSION


class TF32MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(a, b):
        get_extension()
        return torch.ops.matris.tf32_mm(a, b)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        da = TF32MatmulFunction.apply(grad, b.t()) if ctx.needs_input_grad[0] else None
        db = TF32MatmulFunction.apply(a.t(), grad) if ctx.needs_input_grad[1] else None
        return da, db
