"""CUDA source, lazy inline loading and Python interfaces for graph."""

from __future__ import annotations

import hashlib
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

_EXTENSION = None


CPP_SOURCE = r"""

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> graph_prepare_cuda(const at::Tensor &, const at::Tensor &,
                                           const at::Tensor &, const at::Tensor &, double);
std::vector<at::Tensor> graph_fill_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                        const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                        const at::Tensor &, double, int64_t);

at::Tensor graph_distances_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                const at::Tensor &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("distances", &graph_distances_cuda);
    m.def("prepare", &graph_prepare_cuda);
    m.def("fill", &graph_fill_cuda);
}
"""


CUDA_SOURCE = r"""

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <vector>

namespace {
constexpr int threads = 128;

// Compute periodic geometry in registers instead of materializing E-by-3
// gathered positions, converted images and displacement tensors.
__global__ void distances_kernel(const double *positions, const double *cell, const int *edges,
                                 const int *images, int ne, double *distances) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne)
        return;
    double squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        double shift = __dadd_rn(__dadd_rn(__dmul_rn(double(images[3 * e]), cell[axis]),
                                           __dmul_rn(double(images[3 * e + 1]), cell[3 + axis])),
                                 __dmul_rn(double(images[3 * e + 2]), cell[6 + axis]));
        double d =
            __dadd_rn(positions[3 * edges[ne + e] + axis], shift) - positions[3 * edges[e] + axis];
        squared = __dadd_rn(squared, __dmul_rn(d, d));
    }
    distances[e] = sqrt(squared);
}

// One warp per center. Keep only per-atom counts; d2u holds local representative
// ranks until the prefix sum is available. Reverse pairing includes the image.
__global__ void prepare_kernel(const int *edges, const int *ptr, const int *images,
                               const double *distances, int ne, int na, double cutoff, int *reverse,
                               int *d2u, int *rep_counts, int64_t *line_counts, int *status) {
    int center = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (center >= na)
        return;
    int begin = ptr[center], end = ptr[center + 1];
    int nrep = 0, nle = 0, nlt = 0;
    for (int base = begin; base < end; base += 32) {
        int e = base + lane, rev = -1;
        if (e < end) {
            int neighbor = edges[ne + e];
            for (int j = ptr[neighbor]; j < ptr[neighbor + 1]; ++j) {
                if (edges[ne + j] == center && images[3 * j] == -images[3 * e] &&
                    images[3 * j + 1] == -images[3 * e + 1] &&
                    images[3 * j + 2] == -images[3 * e + 2]) {
                    rev = j;
                    break;
                }
            }
            if (rev < 0 || rev == e)
                atomicOr(status, 2);
            reverse[e] = rev;
        }
        unsigned reps = __ballot_sync(0xffffffff, e < end && e < rev);
        unsigned le = __ballot_sync(0xffffffff, e < end && distances[e] <= cutoff);
        unsigned lt = __ballot_sync(0xffffffff, e < end && distances[e] < cutoff);
        if (e < end && e < rev)
            d2u[e] = nrep + __popc(reps & ((1u << lane) - 1));
        nrep += __popc(reps);
        nle += __popc(le);
        nlt += __popc(lt);
    }
    if (lane == 0) {
        rep_counts[center] = nrep;
        // First bond <= cutoff, second bond < cutoff, excluding the same edge.
        line_counts[center] = int64_t(nlt) * (nle - 1);
    }
}

__global__ void representatives_kernel(const int *edges, const int *reverse, const int *rep_prefix,
                                       int ne, int *d2u, int *u2d) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne || e >= reverse[e])
        return;
    int center = edges[e];
    int u = d2u[e] + (center == 0 ? 0 : rep_prefix[center - 1]);
    d2u[e] = u;
    u2d[u] = e;
}

// Stream ordered angle rows directly to their final storage. Ballots compute
// offsets within each warp, so neither per-edge prefixes nor angle workspace
// are needed. All representative IDs are finalized by the preceding kernel.
__global__ void fill_kernel(const int *ptr, const double *distances, const int *reverse,
                            const int64_t *line_prefix, int na, double cutoff, int *d2u,
                            int *line_graph) {
    int center = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (center >= na)
        return;
    int begin = ptr[center], end = ptr[center + 1];
    for (int e = begin + lane; e < end; e += 32)
        if (e > reverse[e])
            d2u[e] = d2u[reverse[e]];
    int64_t row = center == 0 ? 0 : line_prefix[center - 1];
    for (int e = begin; e < end; ++e) {
        if (distances[e] > cutoff)
            continue;
        int u = d2u[min(e, reverse[e])];
        for (int base = begin; base < end; base += 32) {
            int j = base + lane;
            bool valid = j < end && j != e && distances[j] < cutoff;
            unsigned mask = __ballot_sync(0xffffffff, valid);
            if (valid) {
                int64_t offset = 5 * (row + __popc(mask & ((1u << lane) - 1)));
                line_graph[offset] = center;
                line_graph[offset + 1] = u;
                line_graph[offset + 2] = e;
                line_graph[offset + 3] = d2u[min(j, reverse[j])];
                line_graph[offset + 4] = j;
            }
            row += __popc(mask);
        }
    }
}
} // namespace

std::vector<at::Tensor> graph_prepare_cuda(const at::Tensor &edges, const at::Tensor &ptr,
                                           const at::Tensor &images, const at::Tensor &distances,
                                           double cutoff) {
    const c10::cuda::CUDAGuard guard(edges.device());
    TORCH_CHECK(edges.dim() == 2 && edges.size(0) == 2 && ptr.dim() == 1 && ptr.numel() >= 2,
                "Invalid COO/CSR graph shape");
    int64_t ne = edges.size(1), na = ptr.numel() - 1;
    TORCH_CHECK(ne > 0 && ne <= INT_MAX && na <= INT_MAX, "Unsupported graph size");
    TORCH_CHECK(images.dim() == 2 && images.size(0) == ne && images.size(1) == 3 &&
                    distances.dim() == 1 && distances.numel() == ne,
                "Graph geometry shape mismatch");
    auto reverse = at::empty({ne}, edges.options());
    auto d2u = at::empty({ne}, edges.options());
    auto representatives = at::empty({na}, edges.options());
    auto counts = at::empty({na}, edges.options().dtype(at::kLong));
    auto status = at::zeros({1}, edges.options());
    prepare_kernel<<<(na + threads / 32 - 1) / (threads / 32), threads, 0,
                     c10::cuda::getCurrentCUDAStream()>>>(
        edges.data_ptr<int>(), ptr.data_ptr<int>(), images.data_ptr<int>(),
        distances.data_ptr<double>(), ne, na, cutoff, reverse.data_ptr<int>(), d2u.data_ptr<int>(),
        representatives.data_ptr<int>(), counts.data_ptr<int64_t>(), status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {reverse, d2u, representatives, counts, status};
}

std::vector<at::Tensor> graph_fill_cuda(const at::Tensor &edges, const at::Tensor &ptr,
                                        const at::Tensor &distances, const at::Tensor &reverse,
                                        const at::Tensor &d2u, const at::Tensor &rep_prefix,
                                        const at::Tensor &line_prefix, double cutoff,
                                        int64_t nline) {
    const c10::cuda::CUDAGuard guard(edges.device());
    int64_t ne = edges.size(1), na = ptr.numel() - 1;
    TORCH_CHECK(ne > 0 && ne <= INT_MAX && ne % 2 == 0 && nline >= 0, "Invalid graph output size");
    auto u2d = at::empty({ne / 2}, edges.options());
    auto line_graph = at::empty({nline, 5}, edges.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    representatives_kernel<<<(ne + threads - 1) / threads, threads, 0, stream>>>(
        edges.data_ptr<int>(), reverse.data_ptr<int>(), rep_prefix.data_ptr<int>(), ne,
        d2u.data_ptr<int>(), u2d.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    fill_kernel<<<(na + threads / 32 - 1) / (threads / 32), threads, 0, stream>>>(
        ptr.data_ptr<int>(), distances.data_ptr<double>(), reverse.data_ptr<int>(),
        line_prefix.data_ptr<int64_t>(), na, cutoff, d2u.data_ptr<int>(),
        line_graph.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {edges.transpose(0, 1), d2u, u2d, line_graph};
}

at::Tensor graph_distances_cuda(const at::Tensor &positions, const at::Tensor &cell,
                                const at::Tensor &edges, const at::Tensor &images) {
    const c10::cuda::CUDAGuard guard(edges.device());
    int64_t ne = edges.size(1);
    TORCH_CHECK(ne <= INT_MAX, "Unsupported graph size");
    auto distances = at::empty({ne}, positions.options());
    if (ne > 0) {
        distances_kernel<<<(ne + threads - 1) / threads, threads, 0,
                           c10::cuda::getCurrentCUDAStream()>>>(
            positions.data_ptr<double>(), cell.data_ptr<double>(), edges.data_ptr<int>(),
            images.data_ptr<int>(), ne, distances.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return distances;
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
            name=f"matris_graph_{key}",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--extended-lambda"],
            verbose=False,
        )
        _EXTENSION = extension
    return _EXTENSION


def gpu_neighbor_list(cart_coords, lattice, cutoff, *, device, pbc=True):
    """Return GPU COO edges, CSR pointers, images and float64 distances.

    Float64 geometry preserves cutoff decisions; feature tensors remain FP32.
    """
    # Load our CUDA runtime bindings before the Warp neighbor backend.
    extension = get_extension()
    try:
        from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
        from nvalchemiops.torch.neighbors import neighbor_list
    except ImportError as exc:
        raise ImportError(
            "Fast GPU graph construction needs nvalchemi-toolkit-ops. "
            "Install it with `pip install 'nvalchemi-toolkit-ops>=0.4.1,<0.5'`, "
            "or use mode='torch'."
        ) from exc
    positions = torch.asarray(
        cart_coords, dtype=torch.float64, device=device, copy=True
    ).contiguous()
    cell = torch.asarray(
        lattice, dtype=torch.float64, device=device, copy=True
    ).contiguous()
    pbc = torch.as_tensor(pbc, dtype=torch.bool, device=device).expand(3).contiguous()
    max_neighbors = 300
    while True:
        try:
            edges, ptr, shifts = neighbor_list(
                positions,
                cutoff + 1e-8,
                cell=cell.unsqueeze(0),
                pbc=pbc,
                return_neighbor_list=True,
                method="cell_list",
                max_neighbors=max_neighbors,
            )
            break
        except NeighborOverflowError as error:
            # The estimate is a buffer size, not a limit on the physical graph.
            max_neighbors = error.num_neighbors
    distances = extension.distances(positions, cell, edges, shifts)
    return (
        edges.contiguous(),
        ptr.contiguous(),
        shifts.contiguous(),
        distances.contiguous(),
    )


def build_graph_tensors_gpu(edges, ptr, images, distances, *, line_cutoff):
    """Pair reverse edges and assemble the full line graph on the current GPU.

    Only three shape/error scalars leave the device to validate the graph and
    allocate its variable-length line tensor; neighbor/graph arrays stay on GPU.
    """
    if edges.shape[1] == 0:
        return (
            edges.t().contiguous(), edges.new_empty(0),
            edges.new_empty(0), edges.new_empty((0, 5)),
        )
    extension = get_extension()
    reverse, d2u, rep_prefix, line_prefix, status = extension.prepare(
        edges, ptr, images, distances, float(line_cutoff)
    )
    torch.cumsum(rep_prefix, dim=0, out=rep_prefix)
    torch.cumsum(line_prefix, dim=0, out=line_prefix)
    error, nundirected, nline = torch.stack(
        (status[0].to(torch.int64), rep_prefix[-1].to(torch.int64), line_prefix[-1])
    ).tolist()
    if error & 2 or 2 * nundirected != edges.shape[1]:
        raise ValueError("Periodic neighbor list is missing a reverse edge.")
    return tuple(
        extension.fill(
            edges,
            ptr,
            distances,
            reverse,
            d2u,
            rep_prefix,
            line_prefix,
            float(line_cutoff),
            nline,
        )
    )
