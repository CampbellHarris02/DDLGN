# difflogic/difflogic.py
import torch
#import difflogic_cuda
import difflogic_metal
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor

########################################################################################################################

class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'mps',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        if implementation is None:
            if device == 'cuda':
                implementation = 'cuda'
            elif device in ['cpu', 'mps']:
                implementation = 'python'

        self.implementation = implementation

        assert self.implementation in ['cuda', 'python'], self.implementation
        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for training.'
            if self.device == 'cuda':
                return self.forward_cuda_eval(x)
            elif self.device == 'mps':
                return self.forward_metal_eval(x)
            else:
                raise NotImplementedError(f"PackBitsTensor only supported for 'cuda' and 'mps', got {self.device}")
        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x
    '''
    def forward_cuda(self, x):
        assert x.ndim == 2
        assert x.device.type == 'cuda'

        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim

        a, b = self.indices
        w = (torch.nn.functional.softmax(self.weights, dim=-1) if self.training
             else torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)).to(x.dtype)

        return LogicLayerCudaFunction.apply(
            x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
        ).transpose(0, 1)
    
    def forward_cuda_eval(self, x: PackBitsTensor):
        assert not self.training
        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)
        return x

    '''

    def forward_metal_eval(self, x: PackBitsTensor):
        assert not self.training
        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_metal.eval(x.t, a, b, w)
        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c].reshape(2, self.out_dim)
            return c[0].to(torch.int64).to(device), c[1].to(torch.int64).to(device)
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

########################################################################################################################

class GroupSum(torch.nn.Module):
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)

########################################################################################################################
'''
class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y) \
            if ctx.needs_input_grad[0] else None
        grad_w = difflogic_cuda.backward_w(x, a, b, grad_y) if ctx.needs_input_grad[3] else None
        return grad_x, None, None, grad_w, None, None, None
'''