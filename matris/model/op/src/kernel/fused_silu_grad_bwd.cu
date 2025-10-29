#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int VEC_SIZE = 4;
constexpr int THREADS_PER_BLOCK = 256;

__device__ __inline__ float4 sigmoid4(const float4& x) {
    return make_float4(
        1.0f / (1.0f + expf(-x.x)),
        1.0f / (1.0f + expf(-x.y)),
        1.0f / (1.0f + expf(-x.z)),
        1.0f / (1.0f + expf(-x.w))
    );
}

__global__ void silu_grad_bwd_kernel(
    const float* __restrict__ grad_grad_input,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_grad_output,
    float* __restrict__ grad_input,
    int64_t total_elements)
{
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    if (idx >= total_elements) return;
    
    const float4 ggi = *reinterpret_cast<const float4*>(grad_grad_input + idx);
    const float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
    const float4 x = *reinterpret_cast<const float4*>(input + idx);

    const float4 sig_x = sigmoid4(x);
    float4 one_minus_sig;
    one_minus_sig.x = 1.0f - sig_x.x;
    one_minus_sig.y = 1.0f - sig_x.y;
    one_minus_sig.z = 1.0f - sig_x.z;
    one_minus_sig.w = 1.0f - sig_x.w;

    float4 x_one_minus_sig;
    x_one_minus_sig.x = x.x * one_minus_sig.x;
    x_one_minus_sig.y = x.y * one_minus_sig.y;
    x_one_minus_sig.z = x.z * one_minus_sig.z;
    x_one_minus_sig.w = x.w * one_minus_sig.w;

    float4 term;
    term.x = sig_x.x * (1.0f + x_one_minus_sig.x);
    term.y = sig_x.y * (1.0f + x_one_minus_sig.y);
    term.z = sig_x.z * (1.0f + x_one_minus_sig.z);
    term.w = sig_x.w * (1.0f + x_one_minus_sig.w);

    float4 grad_grad_out;
    grad_grad_out.x = ggi.x * term.x;
    grad_grad_out.y = ggi.y * term.y;
    grad_grad_out.z = ggi.z * term.z;
    grad_grad_out.w = ggi.w * term.w;

    float4 sig_x_2; // 2 * sig_x
    sig_x_2.x = 2.0f * sig_x.x;
    sig_x_2.y = 2.0f * sig_x.y;
    sig_x_2.z = 2.0f * sig_x.z;
    sig_x_2.w = 2.0f * sig_x.w;

    float4 one_minus_2sig;
    one_minus_2sig.x = 1.0f - sig_x_2.x;
    one_minus_2sig.y = 1.0f - sig_x_2.y;
    one_minus_2sig.z = 1.0f - sig_x_2.z;
    one_minus_2sig.w = 1.0f - sig_x_2.w;

    float4 x_one_minus_2sig;
    x_one_minus_2sig.x = x.x * one_minus_2sig.x;
    x_one_minus_2sig.y = x.y * one_minus_2sig.y;
    x_one_minus_2sig.z = x.z * one_minus_2sig.z;
    x_one_minus_2sig.w = x.w * one_minus_2sig.w;

    float4 part;
    part.x = 2.0f + x_one_minus_2sig.x;
    part.y = 2.0f + x_one_minus_2sig.y;
    part.z = 2.0f + x_one_minus_2sig.z;
    part.w = 2.0f + x_one_minus_2sig.w;

    float4 sig_deriv;
    sig_deriv.x = sig_x.x * one_minus_sig.x;
    sig_deriv.y = sig_x.y * one_minus_sig.y;
    sig_deriv.z = sig_x.z * one_minus_sig.z;
    sig_deriv.w = sig_x.w * one_minus_sig.w;

    float4 d_term_dx;
    d_term_dx.x = sig_deriv.x * part.x;
    d_term_dx.y = sig_deriv.y * part.y;
    d_term_dx.z = sig_deriv.z * part.z;
    d_term_dx.w = sig_deriv.w * part.w;

    float4 grad_in;
    grad_in.x = ggi.x * go.x * d_term_dx.x;
    grad_in.y = ggi.y * go.y * d_term_dx.y;
    grad_in.z = ggi.z * go.z * d_term_dx.z;
    grad_in.w = ggi.w * go.w * d_term_dx.w;

    *reinterpret_cast<float4*>(grad_grad_output + idx) = grad_grad_out;
    *reinterpret_cast<float4*>(grad_input + idx) = grad_in;
}

void launch_silu_grad_bwd_kernel(
    const float* grad_grad_input,
    const float* grad_output,
    const float* input,
    float* grad_grad_output,
    float* grad_input,
    int feas_num,
    int embed_dim)
{
    const int64_t total_elements = static_cast<int64_t>(feas_num) * embed_dim;
    const int64_t grid_size = (total_elements + THREADS_PER_BLOCK * VEC_SIZE - 1) / (THREADS_PER_BLOCK * VEC_SIZE);
    
    silu_grad_bwd_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        grad_grad_input,
        grad_output,
        input,
        grad_grad_output,
        grad_input,
        total_elements
    );
}