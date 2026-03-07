"""
Differentiable stochastic computing linear layer.
In SC, values in [0,1] are represented as bitstreams; multiplication is AND (expectation = x*y).
This layer maps input to [0,1] (unipolar) then does linear; weight stays full range. Fully differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticLinear(nn.Linear):
    """
    Linear layer with stochastic-computing-style input: activations mapped to [0,1] (unipolar)
    so that in hardware they could be bitstreams. Forward: out = linear(sigmoid(x), weight, bias).
    Same signature as nn.Linear; casts weight to input dtype in forward like nanochat's Linear.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        x_u = torch.sigmoid(x)  # unipolar [0,1] for SC interpretation
        return F.linear(x_u, weight, bias)
