"""Improvements #3 and #4: multi-scale loss landscape and Hessian-eigvec
visualisation plane.

Loads the same scout-then-retrain pipeline as ``landscape_visualization``
(reuses its functions) but produces:

  - a 3 (epochs) x 4 (ranges) grid of filter-normalised landscapes at
    range_val in {0.05, 0.1, 0.5, 1.0} so we see the local geometry the
    optimizer actually explores, not just the global bowl
    (``figures/landscape_multiscale.png``)

  - a 1 x 3 figure where the visualisation plane *is* the top-2 Hessian
    eigenvector plane (filter-normalised), so the arrows from
    ``landscape_eigenvalue_overlay.png`` become the axes themselves
    (``figures/landscape_hessplane.png``)

The point of the second figure is that random projections of high-dim
unit vectors are O(1/sqrt(n)) so the eigenvector arrows in the original
overlay are tiny by construction; using them as the plane removes that
issue.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh

from collapse_detection import detect_collapse
from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.metrics import cosine_similarity, get_flat_gradients
from sg_experiment.models import MLPSurrogate, MLPTrue

import landscape_visualization as base

FIG_DIR = "figures"
RESOLUTION = 41               # finer than base 25
RANGES = [0.05, 0.1, 0.5, 1.0]
HESS_RANGE = 0.3              # range along Hessian-eigvec plane
HESS_RESOLUTION = 35


def _train_with_ckpts(seed: int, ckpt_lo: int, ckpt_hi: int):
    cfg = base._cfg(seed)
    return base._train_with_checkpoints(cfg, ckpt_lo, ckpt_hi)


def _shape_split(flat: torch.Tensor, model) -> List[torch.Tensor]:
    out, idx = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(flat[idx:idx + n].view_as(p).detach().clone())
        idx += n
    return out


def _multi_scale_grid(model, loss_fn, X, y,
                      ckpts: Dict[int, Dict],
                      target_eps: List[int],
                      d1: List[torch.Tensor], d2: List[torch.Tensor],
                      ranges: List[float], resolution: int):
    """Return loss_grids[i_ep][i_range] -> 2D array."""
    grids: List[List[np.ndarray]] = []
    for ep in target_eps:
        model.load_state_dict(ckpts[ep])
        cw = [p.detach().clone() for p in model.parameters()]
        row = []
        for r in ranges:
            g = base.plot_loss_landscape(model, loss_fn, X, y, cw, d1, d2,
                                         resolution=resolution, range_val=r)
            row.append(g)
        grids.append(row)
    return grids


def _filter_normalize_eigvec(vec_np: np.ndarray, model) -> List[torch.Tensor]:
    flat = torch.from_numpy(vec_np.astype(np.float32)).to(
        next(model.parameters()).device)
    parts = _shape_split(flat, model)
    weights = [p.detach() for p in model.parameters()]
    return base.filter_normalize(parts, weights)


def _make_hess_planes(model, loss_fn, X, y, ckpts, target_eps):
    """Return per-ep tuple (d1, d2, lambda1, lambda2)."""
    out = []
    for ep in target_eps:
        model.load_state_dict(ckpts[ep])
        vals, vecs = base.top_hessian_eigvecs(model, loss_fn, X, y, k=2)
        d1 = _filter_normalize_eigvec(vecs[:, 0], model)
        d2 = _filter_normalize_eigvec(vecs[:, 1], model)
        out.append((d1, d2, float(vals[0]), float(vals[1])))
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== IMPROVEMENTS 3 & 4: multi-scale + Hessian-plane landscape ===")
    print(f"Using device: {DEVICE}")

    print("\nScouting for a cleanly collapsing seed...")
    seed, c0 = base._pick_clean_seed()
    print(f"Selected seed = {seed}, scout collapse epoch = {c0['epoch']}")

    print(f"Re-training seed {seed} with checkpoints "
          f"[{base.CKPT_LO}, {base.CKPT_HI}]...")
    history, ckpts = _train_with_ckpts(seed, base.CKPT_LO, base.CKPT_HI)
    c = detect_collapse(history["cosine_sim"])
    if c is None:
        raise RuntimeError("retrain produced no collapse")
    ce = c["epoch"]
    target_eps = [max(base.CKPT_LO, min(base.CKPT_HI, ce + d))
                  for d in (-20, 0, 20)]
    print(f"Collapse epoch = {ce}, target epochs = {target_eps}")

    work = MLPTrue(2, 64, 4, 50).to(DEVICE)
    work.load_state_dict(ckpts[ce])
    center_w = [p.detach().clone() for p in work.parameters()]
    d1, d2 = base._make_directions(center_w, seed=20260429)

    X, y = get_dataset("moons")
    loss_fn = nn.BCEWithLogitsLoss()

    # ---------------- (3) multi-scale random plane ----------------
    print(f"\n[#3] multi-scale grids: ranges={RANGES}  res={RESOLUTION}")
    grids = _multi_scale_grid(work, loss_fn, X, y, ckpts, target_eps,
                              d1, d2, RANGES, RESOLUTION)

    fig, axes = plt.subplots(len(target_eps), len(RANGES),
                             figsize=(4 * len(RANGES), 3.6 * len(target_eps)))
    for i_ep, ep in enumerate(target_eps):
        for j_r, r in enumerate(RANGES):
            ax = axes[i_ep, j_r]
            grid = grids[i_ep][j_r]
            xs = np.linspace(-r, r, RESOLUTION)
            cf = ax.contourf(xs, xs, grid, levels=20, cmap="viridis")
            ax.scatter([0], [0], color="red", s=40, edgecolor="white", zorder=5)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\beta$")
            ax.set_title(f"epoch {ep}, range ±{r}\n"
                         f"loss [{grid.min():.3f}, {grid.max():.3f}]")
            ax.set_aspect("equal")
            plt.colorbar(cf, ax=ax, fraction=0.045)
    fig.suptitle(f"Multi-scale loss landscape (seed {seed}, collapse @ {ce})",
                 fontsize=12)
    plt.tight_layout()
    out_ms = os.path.join(FIG_DIR, "landscape_multiscale.png")
    plt.savefig(out_ms, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_ms}")

    # ---------------- (4) Hessian-eigvec plane ----------------
    print(f"\n[#4] Hessian-eigvec plane: range=±{HESS_RANGE}  "
          f"res={HESS_RESOLUTION}")
    hplanes = _make_hess_planes(work, loss_fn, X, y, ckpts, target_eps)

    fig, axes = plt.subplots(1, len(target_eps),
                             figsize=(5 * len(target_eps), 4.8))
    for ax, ep, (hd1, hd2, l1, l2) in zip(axes, target_eps, hplanes):
        work.load_state_dict(ckpts[ep])
        cw = [p.detach().clone() for p in work.parameters()]
        grid = base.plot_loss_landscape(work, loss_fn, X, y, cw, hd1, hd2,
                                        resolution=HESS_RESOLUTION,
                                        range_val=HESS_RANGE)
        xs = np.linspace(-HESS_RANGE, HESS_RANGE, HESS_RESOLUTION)
        cf = ax.contourf(xs, xs, grid, levels=20, cmap="viridis")
        ax.scatter([0], [0], color="red", s=70, edgecolor="white", zorder=10)
        ax.set_xlabel(rf"$\alpha$ along $v_1$ ($\lambda_1$={l1:.1f})")
        ax.set_ylabel(rf"$\beta$  along $v_2$ ($\lambda_2$={l2:.1f})")
        ax.set_title(f"epoch {ep}\nloss [{grid.min():.3f}, {grid.max():.3f}]")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, fraction=0.045)
    fig.suptitle(f"Loss surface in the top-2 Hessian eigenvector plane "
                 f"(seed {seed}, collapse @ {ce})", fontsize=12)
    plt.tight_layout()
    out_hp = os.path.join(FIG_DIR, "landscape_hessplane.png")
    plt.savefig(out_hp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_hp}")


if __name__ == "__main__":
    main()
