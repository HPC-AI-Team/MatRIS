#include "Opdefine.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd, "fuse_silu_bwd");
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd, "fuse_silu_grad_bwd"); 
}

TORCH_LIBRARY(matris_op, m)
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd);
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd);  
}
