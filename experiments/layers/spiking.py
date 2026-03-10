"""
Spiking activation for use in MLP (and optionally elsewhere).
Single-step LIF-style threshold with surrogate gradient for backprop.
Inspired by SpikeLM's ElasticBiSpiking and SpikeGPT's LIF; simplified for (B, T, C) without extra time dimension.

Uses a custom torch.autograd.Function; torch.compile can hang on it. If --variant=spiking hangs, use --no-compile.
"""

import torch
import torch.nn as nn
import math


class _SurrogateSpiking(torch.autograd.Function):
    """Forward: binary or ternary spike; backward: surrogate gradient (ATan-style)."""

    @staticmethod
    def forward(ctx, x, threshold, alpha, ternary):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.alpha = alpha
        ctx.ternary = ternary
        if ternary:
            # ternary: -1, 0, 1
            out = torch.sign(x) * (x.abs() >= threshold).to(x.dtype)
        else:
            # binary: 0 or 1 (ReLU-like spike)
            out = (x >= threshold).to(x.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        t, alpha, ternary = ctx.threshold, ctx.alpha, ctx.ternary
        # Surrogate: gradient proportional to 1 / (1 + (alpha * x)^2) or similar smooth approximation
        if ternary:
            # derivative of sign-like: nonzero only near zero
            grad_input = grad_output * alpha / (1 + (alpha * x).pow(2))
        else:
            # derivative of step: smooth bump
            grad_input = grad_output * alpha / (1 + (alpha * (x - t)).pow(2))
        return grad_input, None, None, None


def spiking_forward(x, threshold=0.0, alpha=2.0, ternary=False):
    return _SurrogateSpiking.apply(x, threshold, alpha, ternary)


class SpikingActivation(nn.Module):
    """
    Spiking activation: threshold input to binary (0/1) or ternary (-1/0/1) with surrogate gradient.
    Surrogate uses ATan-style smooth gradient. Optional learnable scale for output magnitude (elastic bi-spiking style).
    """

    def __init__(self, threshold=0.0, alpha=4.0, ternary=False, learnable_scale=True, dim=None):
        super().__init__()
        if dim is not None:
            # Per-channel learnable thresholds; assumes last dimension = dim
            self.threshold = nn.Parameter(torch.full((dim,), float(threshold)))
        else:
            self.threshold = threshold
        self.ternary = ternary
        self.alpha = alpha  # surrogate smoothness (fixed)
        self.scale = nn.Parameter(torch.tensor(1.0)) if learnable_scale else 1.0
        self.learnable_scale = learnable_scale

    def forward(self, x):
        spike = _SurrogateSpiking.apply(x, self.threshold, self.alpha, self.ternary)
        gate = torch.sigmoid(x)
        s = self.scale.clamp(min=0.1) if self.learnable_scale and isinstance(self.scale, torch.Tensor) else self.scale
        return spike * gate * s
