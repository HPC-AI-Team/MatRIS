"""Precise ATen powers with an explicit compiler and autograd boundary."""

import hashlib
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

_EXTENSION = None

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <torch/library.h>

at::Tensor matris_pow(const at::Tensor &x, double exponent) {
    return at::pow(x, exponent);
}
at::Tensor matris_pow_meta(const at::Tensor &x, double exponent) {
    return at::empty_like(x);
}
TORCH_LIBRARY_FRAGMENT(matris, m) {
    m.def("pow(Tensor x, float exponent) -> Tensor");
}
TORCH_LIBRARY_IMPL(matris, CPU, m) {
    m.impl("pow", TORCH_FN(matris_pow));
}
TORCH_LIBRARY_IMPL(matris, CUDA, m) {
    m.impl("pow", TORCH_FN(matris_pow));
}
TORCH_LIBRARY_IMPL(matris, Meta, m) {
    m.impl("pow", TORCH_FN(matris_pow_meta));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_autograd", []() {
        namespace py = pybind11;
        auto function = py::module_::import("matris.model.op.elementwise").attr("PowFunction");
        py::module_::import("torch.library").attr("register_autograd")(
            "matris::pow", function.attr("backward"),
            py::arg("setup_context") = function.attr("setup_context"));
    });
}
"""


@torch.compiler.assume_constant_result
def get_extension():
    """Load the registered ATen wrapper once, keyed by source and ABI."""
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("MAX_JOBS", "4")
        key = hashlib.sha256(
            (f"{sys.implementation.cache_tag}:{torch.__version__}:{torch.version.cuda}\n" + CPP_SOURCE).encode()
        ).hexdigest()[:16]
        extension = load_inline(
            name=f"matris_elementwise_{key}", cpp_sources=CPP_SOURCE,
            extra_cflags=["-O3"], with_cuda=True, verbose=False,
        )
        extension.register_autograd()
        _EXTENSION = extension
    return _EXTENSION


class PowFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, exponent):
        get_extension()
        return torch.ops.matris.pow(x, exponent)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
        ctx.exponent = inputs[1]

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        if ctx.exponent == 0:
            return (torch.zeros_like(x), None)
        return (grad * (ctx.exponent * PowFunction.apply(x, ctx.exponent - 1)), None)
