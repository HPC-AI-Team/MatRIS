#include <torch/extension.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

using namespace std;

void launch_silu_bwd_kernel(const float *dgrad, const float *input, float *output, int feas_num, int embed_dim);

void launch_silu_grad_bwd_kernel(const float *grad_grad_input, const float *grad_output, const float *input, 
                                float *grad_grad_output, float *grad_input,
                                int feas_num, int embed_dim);

torch::Tensor fused_SiLU_Bwd(const torch::Tensor &dgrad, const torch::Tensor &input)
{  
    const at::cuda::OptionalCUDAGuard device_guard(device_of(dgrad)); 
    int64_t embed_dim = input.size(-1);    
    int64_t feas_num = input.size(-2);
    
    torch::Tensor output = torch::empty({feas_num, embed_dim}, torch::TensorOptions().dtype(torch::kFloat).device(input.device()));

    launch_silu_bwd_kernel(
        static_cast<float*>(dgrad.data_ptr()),
        static_cast<float*>(input.data_ptr()), 
        static_cast<float*>(output.data_ptr()),
        feas_num,
        embed_dim
    );
    return output;
}

std::vector<torch::Tensor> fused_SiLU_Grad_Bwd(const torch::Tensor &grad_grad_input, const torch::Tensor &grad_output,
                                                const torch::Tensor &input)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_grad_input));
    int64_t embed_dim = input.size(-1);
    int64_t feas_num = input.size(-2);
    //int64_t total_elements = feas_num * embed_dim;
    torch::Tensor grad_grad_output = torch::empty({feas_num, embed_dim}, torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
    torch::Tensor grad_input = torch::empty({feas_num, embed_dim}, torch::TensorOptions().dtype(torch::kFloat).device(input.device()));

    launch_silu_grad_bwd_kernel(
        static_cast<float*>(grad_grad_input.data_ptr()),
        static_cast<float*>(grad_output.data_ptr()),
        static_cast<float*>(input.data_ptr()),
        //output
        static_cast<float*>(grad_grad_output.data_ptr()),
        static_cast<float*>(grad_input.data_ptr()),
        feas_num, embed_dim 
    );
    return {grad_grad_output, grad_input};
}