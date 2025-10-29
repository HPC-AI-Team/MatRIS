import torch
from torch.autograd import Function
from torch.profiler import profile, ProfilerActivity,record_function


sigmoid_bwd_string = """
    template <typename T> T sigmoid_bwd_kernel(T dgrad, T output) 
    { 
        return dgrad * output * (1 - output); 
    }
"""
sigmoid_bwd_func = torch.cuda.jiterator._create_jit_fn(sigmoid_bwd_string)

sigmoid_gradbwd_string = """
    template <typename T> T sigmoid_grad_bwd_kernel(T grad_output, T dgrad, T output, T &grad_dgrad, T &grad_output_val) 
    { 
        grad_dgrad = grad_output * output * (1 - output);
        grad_output_val = grad_output * dgrad * (1 - 2 * output); 
    }
"""
sigmoid_gradbwd_func = torch.cuda.jiterator._create_multi_output_jit_fn(sigmoid_gradbwd_string, num_outputs=2)


class sigmoid_bwd(Function):
    @staticmethod
    def forward(ctx, dgrad, output):
        #grad_feas = dgrad * output * (1 - output)
        with record_function("sigmoid bwd"):  
            grad_feas = sigmoid_bwd_func(dgrad, output)
        ctx.save_for_backward(dgrad, output)
        #ctx.save_for_backward()
        return grad_feas

    @staticmethod
    def backward(ctx, grad_output):
        dgrad, output = ctx.saved_tensors
        """
        grad_dgrad = grad_output * output * (1 - output)
        grad_output_val = grad_output * dgrad * (1 - 2 * output)
        """
        with record_function("sigmoid grad bwd"):  
            grad_dgrad, grad_output_val = sigmoid_gradbwd_func(grad_output, dgrad, output) 
        return grad_dgrad, grad_output_val

class sigmoid_fwd(Function):
    @staticmethod
    def forward(ctx, feas): 
        output = torch.nn.functional.sigmoid(feas)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, dgrad):
        output, = ctx.saved_tensors
        grad_feas = sigmoid_bwd.apply(dgrad, output)
        return grad_feas

def fused_sigmoid(input_feas):
    if input_feas.device.type == "cuda":
        return sigmoid_fwd.apply(input_feas)
    else:
        return torch.nn.functional.sigmoid(input_feas)
    