"""Rigorous Hessian primitives shared by all curvature analyses.

Both estimators take an explicit ``torch.Generator`` so that random probe
vectors are drawn from an RNG stream that is *isolated* from the global
torch RNG used for model initialisation and SGD. Without this isolation,
adding a curvature probe perturbs the training trajectory through the
RNG stream alone (independent of any GPU non-determinism).

- ``hutchinson_trace``: tr(H) ≈ (1/m) Σ z_iᵀ H z_i with Rademacher z_i.
- ``lanczos_top_eigs``: top-k eigenvalues of the *symmetric* Hessian via
  Lanczos iteration with full reorthogonalisation (the gold-standard
  matrix-free top-eigenvalue method; far more reliable than power
  iteration once you want more than λ₁ or you need rapid convergence on
  near-degenerate spectra).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _flat_grad(loss: torch.Tensor, params, create_graph: bool):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def _hvp(flat_grad: torch.Tensor, params, v: torch.Tensor) -> torch.Tensor:
    """Hessian-vector product Hv via grad(g·v, params)."""
    gv = (flat_grad * v).sum()
    parts = torch.autograd.grad(gv, params, retain_graph=True)
    return torch.cat([p.reshape(-1) for p in parts])


def _clear_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def hutchinson_trace(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 5,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Stochastic estimate of tr(H) using Rademacher probes."""
    params = [p for p in model.parameters() if p.requires_grad]
    _clear_grads(model)

    pred = model(X)
    loss = loss_fn(pred, y)
    flat_g = _flat_grad(loss, params, create_graph=True)

    estimates = []
    for _ in range(n_samples):
        z = torch.randint(
            0, 2, flat_g.shape,
            device=flat_g.device, dtype=flat_g.dtype,
            generator=generator,
        )
        z = z * 2 - 1
        Hz = _hvp(flat_g, params, z)
        estimates.append(float((z * Hz).sum().item()))

    _clear_grads(model)
    return float(sum(estimates) / len(estimates))


def lanczos_top_eigs(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_iter: int = 30,
    k: int = 5,
    generator: Optional[torch.Generator] = None,
    tol: float = 1e-9,
) -> List[float]:
    """Top-``k`` eigenvalues of the Hessian via Lanczos with full reorthog.

    Returns a list of length ``min(k, n_iter)`` sorted descending.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    _clear_grads(model)

    pred = model(X)
    loss = loss_fn(pred, y)
    flat_g = _flat_grad(loss, params, create_graph=True)

    n = flat_g.numel()
    device, dtype = flat_g.device, flat_g.dtype

    # Initial unit vector from the isolated generator.
    v = torch.randn(n, device=device, dtype=dtype, generator=generator)
    v = v / v.norm()

    V = [v]
    alphas: List[float] = []
    betas: List[float] = []

    for j in range(n_iter):
        w = _hvp(flat_g, params, V[j])
        alpha = float(torch.dot(w, V[j]).item())
        alphas.append(alpha)

        # Three-term recurrence then full reorthog (twice for numerical
        # stability against drift in finite precision).
        w = w - alpha * V[j]
        if j > 0:
            w = w - betas[-1] * V[j - 1]
        for _pass in range(2):
            for vk in V:
                w = w - torch.dot(w, vk) * vk

        beta = float(w.norm().item())
        if beta < tol:
            break
        betas.append(beta)
        V.append(w / beta)

    _clear_grads(model)

    # Build the (j+1) x (j+1) tridiagonal T and eigendecompose it. T's
    # eigenvalues are Ritz values converging to the extreme eigenvalues of H.
    m = len(alphas)
    T = torch.zeros(m, m, dtype=torch.float64)
    for i, a in enumerate(alphas):
        T[i, i] = a
    # The last beta describes the magnitude of the (m+1)-th Lanczos vector
    # outside our truncated basis; it does not enter T.
    for i, b in enumerate(betas[: m - 1]):
        T[i, i + 1] = b
        T[i + 1, i] = b
    eigs = torch.linalg.eigvalsh(T).tolist()
    eigs.sort(reverse=True)
    return eigs[:k]
