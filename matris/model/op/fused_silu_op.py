import torch
from torch.autograd import Function
from torch.profiler import profile, ProfilerActivity,record_function

silu_bwd_string = """
    template <typename T> T silu_bwd_kernel(T dgrad, T x) 
    { 
        T sigmoid_x = 1/(1 + exp(-x));
        
        T grad_input = dgrad * (1 + x*(1-sigmoid_x)) * sigmoid_x;
        
        return grad_input;
    }
"""
silu_bwd_func = torch.cuda.jiterator._create_jit_fn(silu_bwd_string)

silu_gradbwd_string = """
    template <typename T> T silu_grad_bwd_kernel(T grad_grad_input, T grad_output, T x, T &grad_grad_output, T &grad_x) 
    {   
        T sigmoid_x = 1/(1 + exp(-x));
        T term = sigmoid_x * (1 + x * (1 - sigmoid_x));
        grad_grad_output = grad_grad_input * term;
        
        T sigmoid_derivative = sigmoid_x * (1 - sigmoid_x);
        T d_term_dx = sigmoid_derivative * (2 + x * (1 - 2 * sigmoid_x));
        grad_x = grad_grad_input * grad_output * d_term_dx; 
        
    }
"""
silu_gradbwd_func = torch.cuda.jiterator._create_multi_output_jit_fn(silu_gradbwd_string, num_outputs=2)


class fusion_silu_bwd(Function):
    @staticmethod
    def forward(ctx, grad_output, x):
        ctx.save_for_backward(x, grad_output)
        """
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (1 + x * (1 - sigmoid_x)) * sigmoid_x
        """
        with record_function("silu bwd"): 
            grad_input = silu_bwd_func(grad_output, x) 
        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        """
        sigmoid_x = torch.sigmoid(x)
        term = sigmoid_x * (1 + x * (1 - sigmoid_x))
        grad_grad_output = grad_grad_input * term
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        d_term_dx = sigmoid_derivative * (2 + x * (1 - 2 * sigmoid_x))
        
        grad_x = grad_grad_input * grad_output * d_term_dx
        """
        x, grad_output = ctx.saved_tensors
            
        with record_function("silu grad bwd"):  
            grad_grad_output, grad_x = silu_gradbwd_func(grad_grad_input, grad_output, x)  
        return grad_grad_output, grad_x

class fusion_silu_fwd(Function):
    @staticmethod
    def forward(ctx, x):
        
        out = torch.nn.functional.silu(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = fusion_silu_bwd.apply(grad_output, x)
        return grad_input

def fused_silu(input_feas):
    if input_feas.device.type == "cuda":
        return fusion_silu_fwd.apply(input_feas)
    else:
        return torch.nn.functional.silu(input_feas)

# out = fused_silu(input)