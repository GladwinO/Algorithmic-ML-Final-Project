"""MLP models with sigmoid activations for SG vs true-gradient comparison.

The forward pass uses a steep sigmoid (beta_f). The surrogate version replaces
the backward pass derivative with that of a flatter sigmoid (beta_sg), exactly
as in Gygax & Zenke (2024), Section 4.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SurrogateActivation(torch.autograd.Function):
    """Forward: sigmoid(beta_f * x). Backward: surrogate derivative using beta_sg."""

    @staticmethod
    def forward(ctx, x, beta_f, beta_sg):
        ctx.save_for_backward(x)
        ctx.beta_f = beta_f
        ctx.beta_sg = beta_sg
        return torch.sigmoid(beta_f * x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta_sg = ctx.beta_sg
        sg = torch.sigmoid(beta_sg * x)
        surrogate_deriv = beta_sg * sg * (1 - sg)
        return grad_output * surrogate_deriv, None, None


def surrogate_activation(x, beta_f, beta_sg):
    return SurrogateActivation.apply(x, beta_f, beta_sg)


def true_activation(x, beta_f):
    return torch.sigmoid(beta_f * x)


class MLPTrue(nn.Module):
    """MLP using the true (steep) sigmoid throughout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, beta_f: float):
        super().__init__()
        self.beta_f = beta_f
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = true_activation(layer(x), self.beta_f)
        return self.layers[-1](x)


class MLPSurrogate(nn.Module):
    """MLP using surrogate-gradient sigmoid throughout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        beta_f: float,
        beta_sg: float,
    ):
        super().__init__()
        self.beta_f = beta_f
        self.beta_sg = beta_sg
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = surrogate_activation(layer(x), self.beta_f, self.beta_sg)
        return self.layers[-1](x)
