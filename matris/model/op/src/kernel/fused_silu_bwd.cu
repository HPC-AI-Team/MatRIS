#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void silu_bwd_kernel(const float* dgrad, const float* input, float* output, int num_float4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_float4) return;

    float4 x4 = reinterpret_cast<const float4*>(input)[idx];
    float4 g4 = reinterpret_cast<const float4*>(dgrad)[idx];
    float4 out4;

    auto process = [](float g_val, float x_val) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        float term = 1.0f + x_val * (1.0f - sigmoid_x);
        return g_val * (sigmoid_x * term);
    };

    out4.x = process(g4.x, x4.x);
    out4.y = process(g4.y, x4.y);
    out4.z = process(g4.z, x4.z);
    out4.w = process(g4.w, x4.w);

    reinterpret_cast<float4*>(output)[idx] = out4;
}

void launch_silu_bwd_kernel(const float* dgrad, 
                            const float* input, 
                            float* output, 
                            int feas_num, 
                            int embed_dim) 
{
    int num_elements = feas_num * embed_dim;
    int num_float4 = num_elements / 4;
    int block_size = 256;
    int grid_size = (num_float4 + block_size - 1) / block_size;

    silu_bwd_kernel<<<grid_size, block_size>>>(dgrad, input, output, num_float4);
}
