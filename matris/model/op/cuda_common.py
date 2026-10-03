"""Shared CUDA types and primitives."""

CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <optional>

namespace matris_cuda {
struct Index {
    const void *data;
    bool wide;
    __device__ int64_t operator[](int64_t i) const {
        return wide ? static_cast<const int64_t *>(data)[i] : static_cast<const int32_t *>(data)[i];
    }
};
inline Index index_of(const at::Tensor &x) { return {x.data_ptr(), x.scalar_type() == at::kLong}; }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
} // namespace matris_cuda
"""
