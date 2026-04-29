"""Gradient comparison metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def get_flat_gradients(model) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().flatten())
    return torch.cat(grads)


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    return F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()


def sign_agreement(g1: torch.Tensor, g2: torch.Tensor) -> float:
    return (torch.sign(g1) == torch.sign(g2)).float().mean().item()


def relative_magnitude(g1: torch.Tensor, g2: torch.Tensor) -> float:
    return (torch.norm(g2) / (torch.norm(g1) + 1e-8)).item()
