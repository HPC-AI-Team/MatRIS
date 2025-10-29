#ifndef OP_SRC_OPDECLARE_H_
#define OP_SRC_OPDECLARE_H_

#include <torch/extension.h>


torch::Tensor fused_SiLU_Bwd(const torch::Tensor &dgrad, const torch::Tensor &input);

std::vector<torch::Tensor> fused_SiLU_Grad_Bwd(const torch::Tensor &grad_grad_input, const torch::Tensor &grad_output,
                                                const torch::Tensor &input);


#endif  // OP_SRC_OPDECLARE_H_