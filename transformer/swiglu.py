import torch
from torch import nn
from torch.nn.functional import silu

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=-1)
        return silu(x0) * x1
