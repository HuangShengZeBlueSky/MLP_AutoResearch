import torch.nn as nn
import torch
from torch.autograd import Function


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class MyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in',nonlinearity='relu')
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.maximum(torch.zeros_like(input), input)


    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input


class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()

    def forward(self, input):
        return ReLUFunction.apply(input)



class BatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5, momentum=0.1, running_mean=None, running_var=None, training=True):
        if training == True:
            sample_mean = input.mean(dim=0)
            sample_var = input.var(dim=0) + eps
            
            running_mean.mul_(1-momentum).add_(momentum * sample_mean)
            running_var.mul_(1-momentum).add_(momentum * sample_var)
            
            input_ = (input - sample_mean) / torch.sqrt(sample_var)
            out = gamma * input_ + beta
            ctx.save_for_backward(input_, gamma, sample_var)
        else:
            input_ = (input - running_mean) / torch.sqrt(running_var)
            out = gamma * input_ + beta
        return out


    @staticmethod
    def backward(ctx, grad_output):
        input_, gamma, var = ctx.saved_tensors
        N, D = grad_output.shape
        grad_gamma = torch.sum(input_ * grad_output, dim=0)
        grad_beta = torch.sum(grad_output, dim=0)
        
        grad_input_ = gamma * grad_output
        grad_input = (grad_input_ - 1 / N * torch.sum(grad_input_, dim=0) - 1 / N * input_ * torch.sum(grad_input_ * input_, dim=0)) / torch.sqrt(var)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None


class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MyBatchNorm1d, self).__init__()
        self.training = True        
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.empty(num_features))
        self.beta = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, input):
        return BatchNormFunction.apply(
            input,
            self.gamma,
            self.beta,
            self.eps,
            self.momentum,
            self.running_mean,
            self.running_var,
            self.training
        )





