import torch
from torch.autograd import Function
from torch.profiler import profile, ProfilerActivity,record_function

fusion_envelop_string = """
    template <typename T> T envelop_fwd_kernel(T r_scaled, T a, T b, T c, T p) 
    {
        return ( 1 + a * pow(r_scaled, p) + b * pow(r_scaled, (p + 1)) + c * pow(r_scaled, (p + 2)) );
    }
"""
#return ( 1 + a * r_scaled ** p + b * r_scaled ** (p + 1) + c * r_scaled ** (p + 2) );
envelop_fwd_func = torch.cuda.jiterator._create_jit_fn(fusion_envelop_string)

fusion_envelop_bwd_string = """
    template <typename T> T envelop_bwd_kernel(T grad_output, T r_scaled, T a, T b, T c, T p) 
    {
        T grad_r_scaled = (
            a * p * pow(r_scaled, (p - 1))
            + b * (p + 1) * pow(r_scaled, p)
            + c * (p + 2) * pow(r_scaled, (p + 1))
        );
        return grad_output * grad_r_scaled; 
    }
"""
envelop_bwd_func = torch.cuda.jiterator._create_jit_fn(fusion_envelop_bwd_string)

fusion_envelop_gradbwd_string = """
    template <typename T> T envelop_bwd_kernel(T d_grad_r_scaled, T grad_output, T r_scaled, T a, T b, T c, T p, T &grad_grad_output, T &grad_r_scaled) 
    {
        grad_grad_output = (
            a * p * pow(r_scaled, (p - 1))
            + b * (p + 1) * pow(r_scaled, p)
            + c * (p + 2) * pow(r_scaled, (p + 1))
        );
        grad_grad_output = d_grad_r_scaled * grad_grad_output;

        T grad_r_scaled_term = (
            a * p * (p - 1) * pow(r_scaled, (p - 2))
            + b * (p + 1) * p * pow(r_scaled, (p - 1))
            + c * (p + 2) * (p + 1) * pow(r_scaled, p)
        );
        grad_r_scaled = d_grad_r_scaled * grad_output * grad_r_scaled_term;
    }
"""
envelop_gradbwd_func = torch.cuda.jiterator._create_multi_output_jit_fn(fusion_envelop_gradbwd_string, num_outputs=2)


class envelop_bwd(Function):
    @staticmethod
    def forward(ctx, grad_output: torch.Tensor, r_scaled: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: torch.Tensor): 
        """
        # Compute grad_r_scaled
        grad_r_scaled_part = (
            a * p * r_scaled ** (p - 1)
            + b * (p + 1) * r_scaled ** p
            + c * (p + 2) * r_scaled ** (p + 1)
        )
        
        grad_r_scaled = grad_output * grad_r_scaled_part
        """ 
        grad_r_scaled = envelop_bwd_func(grad_output, r_scaled, a, b, c, p)
        # Save for backward
        ctx.save_for_backward(grad_output, r_scaled)
        ctx.a = a
        ctx.b = b
        ctx.c = c
        ctx.p = p

        return grad_r_scaled

    @staticmethod
    def backward(ctx, d_grad_r_scaled):
        grad_output, r_scaled = ctx.saved_tensors
        a = ctx.a
        b = ctx.b
        c = ctx.c
        p = ctx.p
        """
        # Compute gradient w.r.t grad_output
        grad_grad_output = (
            a * p * r_scaled ** (p - 1)
            + b * (p + 1) * r_scaled ** p
            + c * (p + 2) * r_scaled ** (p + 1)
        )
        grad_grad_output = d_grad_r_scaled * grad_grad_output  # chain rule

        # Compute gradient w.r.t r_scaled
        grad_r_scaled_term = (
            a * p * (p - 1) * r_scaled ** (p - 2)
            + b * (p + 1) * p * r_scaled ** (p - 1)
            + c * (p + 2) * (p + 1) * r_scaled ** p
        )
        grad_r_scaled = d_grad_r_scaled * grad_output * grad_r_scaled_term  # chain rule
        """
        grad_grad_output, grad_r_scaled = envelop_gradbwd_func(d_grad_r_scaled, grad_output, r_scaled, a, b, c, p)
        # Return gradients in the order of forward's inputs
        return grad_grad_output, grad_r_scaled, None, None, None, None

    
class envelop_fwd(Function):
    @staticmethod
    def forward(ctx, r_scaled: torch.Tensor, a: torch.Tensor, b:torch.Tensor, c: torch.Tensor, p:torch.Tensor): 
        output = envelop_fwd_func(r_scaled, a, b, c, p)
        ctx.save_for_backward(r_scaled)
        ctx.a = a
        ctx.b = b
        ctx.c = c
        ctx.p = p
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (r_scaled,) = ctx.saved_tensors
        a = ctx.a
        b = ctx.b
        c = ctx.c
        p = ctx.p

        grad_r_scaled = envelop_bwd.apply(grad_output, r_scaled, a, b, c, p)
        
        return grad_r_scaled, None, None, None, None
    

def fusion_env_val(r_scaled:torch.Tensor, a:torch.Tensor, b:torch.Tensor, c:torch.Tensor, p:torch.Tensor):
    if r_scaled.device.type == "cuda":
        return envelop_fwd.apply(r_scaled, a, b, c, p) 
    else:
        return ( 1 + a * r_scaled ** p + b * r_scaled ** (p + 1) + c * r_scaled ** (p + 2) )

