"""ANALYSIS 3: Loss landscape visualisation around the collapse epoch.

Implements filter normalisation (Li et al., 2018) and renders the loss
surface in a 2D random plane through the trained weights at three time
points: collapse_epoch ± 20. The same two random directions are used for
all three plots so the plots are directly comparable.

Top Hessian eigenvectors are obtained by wrapping the Hessian-vector
product as a ``scipy.sparse.linalg.LinearOperator`` and calling ``eigsh``;
the two largest eigenvectors are projected onto the same two random
directions and overlaid as arrows in the second figure.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh

from collapse_detection import detect_collapse
from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.metrics import (
    cosine_similarity,
    get_flat_gradients,
)
from sg_experiment.models import MLPSurrogate, MLPTrue

FIG_DIR = "figures"
RESOLUTION = 25
RANGE_VAL = 1.0
CKPT_LO = 180
CKPT_HI = 300


def _cfg(seed: int) -> Dict:
    return {
        "hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
        "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed,
    }


# ------------------------- training & checkpoints -------------------------


def _train_with_checkpoints(config: Dict, ckpt_lo: int, ckpt_hi: int
                            ) -> Tuple[Dict[str, List], Dict[int, Dict]]:
    torch.manual_seed(config["seed"])
    X, y = get_dataset(config["dataset"])

    true_net = MLPTrue(2, config["hidden_dim"], config["num_layers"],
                       config["beta_f"]).to(DEVICE)
    surr_net = MLPSurrogate(2, config["hidden_dim"], config["num_layers"],
                            config["beta_f"], config["beta_sg"]).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_surr = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])

    history: Dict[str, List] = {
        "epoch": [], "cosine_sim": [], "true_loss": [], "grad_norm_true": [],
    }
    ckpts: Dict[int, Dict] = {}

    for epoch in range(config["n_epochs"]):
        opt_true.zero_grad()
        loss_true = loss_fn(true_net(X), y)
        loss_true.backward()
        g_true = get_flat_gradients(true_net)

        surr_net.load_state_dict(true_net.state_dict())
        opt_surr.zero_grad()
        loss_surr = loss_fn(surr_net(X), y)
        loss_surr.backward()
        g_surr = get_flat_gradients(surr_net)

        history["epoch"].append(epoch)
        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["true_loss"].append(float(loss_true.item()))
        history["grad_norm_true"].append(float(torch.norm(g_true).item()))

        if ckpt_lo <= epoch <= ckpt_hi:
            ckpts[epoch] = {
                k: v.detach().clone()
                for k, v in true_net.state_dict().items()
            }

        opt_true.step()

    return history, ckpts


# ------------------------- direction & landscape -------------------------


def filter_normalize(direction: List[torch.Tensor],
                     weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per Li et al. 2018 for FC layers: rescale each row of direction to
    have the same L2 norm as the corresponding row of the weight matrix.
    Biases are zeroed (the original implementation's default for FC layers).
    """
    out: List[torch.Tensor] = []
    for d, w in zip(direction, weights):
        if w.dim() == 1:  # bias
            out.append(torch.zeros_like(w))
            continue
        d_n = d.clone()
        for i in range(w.shape[0]):
            wn = w[i].norm()
            dn = d_n[i].norm()
            if dn > 1e-12:
                d_n[i] = d_n[i] * (wn / dn)
            else:
                d_n[i].zero_()
        out.append(d_n)
    return out


def _state_to_param_list(state_dict, model: MLPTrue) -> List[torch.Tensor]:
    """Materialise model state in the parameter order of model.parameters()."""
    out = []
    for name, _ in model.named_parameters():
        out.append(state_dict[name])
    return out


def _set_params(model: MLPTrue, params: List[torch.Tensor]):
    with torch.no_grad():
        for p, new in zip(model.parameters(), params):
            p.copy_(new)


def _make_directions(weights: List[torch.Tensor], seed: int
                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    g = torch.Generator(device=weights[0].device)
    g.manual_seed(seed)
    raw1 = [torch.randn(w.shape, generator=g, device=w.device) for w in weights]
    raw2 = [torch.randn(w.shape, generator=g, device=w.device) for w in weights]
    d1 = filter_normalize(raw1, weights)
    d2 = filter_normalize(raw2, weights)
    return d1, d2


def plot_loss_landscape(model: MLPTrue, loss_fn, X, y,
                        center_weights: List[torch.Tensor],
                        d1: List[torch.Tensor], d2: List[torch.Tensor],
                        resolution: int = RESOLUTION,
                        range_val: float = RANGE_VAL) -> np.ndarray:
    """Evaluate loss on an alpha x beta grid around center_weights using d1,d2."""
    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)
    grid = np.zeros((resolution, resolution), dtype=float)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                params = [w + a * dd1 + b * dd2
                          for w, dd1, dd2 in zip(center_weights, d1, d2)]
                _set_params(model, params)
                loss = loss_fn(model(X), y)
                grid[j, i] = float(loss.item())

    _set_params(model, center_weights)
    return grid


# ------------------------- Hessian eigenvectors -------------------------


def _hvp_at(model: MLPTrue, loss_fn, X, y, vec: torch.Tensor) -> torch.Tensor:
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    Hv = torch.autograd.grad(flat, params, grad_outputs=vec, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in Hv]).detach()


def top_hessian_eigvecs(model: MLPTrue, loss_fn, X, y, k: int = 2):
    n = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device

    def matvec(v_np):
        v = torch.from_numpy(v_np.astype(np.float64)).to(device, dtype=torch.float32)
        Hv = _hvp_at(model, loss_fn, X, y, v)
        return Hv.cpu().numpy().astype(np.float64)

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals, vecs = eigsh(op, k=k, which="LA", maxiter=300, tol=1e-4)
    # eigsh returns smallest-to-largest; flip.
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs  # vecs columns are eigenvectors


def _project_eigvec_to_dirs(vec_np: np.ndarray,
                            d1_flat: torch.Tensor,
                            d2_flat: torch.Tensor) -> Tuple[float, float]:
    v = torch.from_numpy(vec_np.astype(np.float32)).to(d1_flat.device)
    a = float(torch.dot(v, d1_flat / (d1_flat.norm() + 1e-12)).item())
    b = float(torch.dot(v, d2_flat / (d2_flat.norm() + 1e-12)).item())
    return a, b


# ------------------------- driver -------------------------


def _flatten(plist: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in plist])


def _pick_clean_seed(max_seed: int = 30) -> Tuple[int, Dict]:
    """Find the first seed whose collapse epoch sits comfortably inside
    [CKPT_LO+20, CKPT_HI-20] so ±20 windows are entirely captured."""
    for s in range(max_seed):
        torch.manual_seed(s)
        X, y = get_dataset("moons")
        true_net = MLPTrue(2, 64, 4, 50).to(DEVICE)
        surr_net = MLPSurrogate(2, 64, 4, 50, 5).to(DEVICE)
        surr_net.load_state_dict(true_net.state_dict())
        loss_fn = nn.BCEWithLogitsLoss()
        opt_t = torch.optim.SGD(true_net.parameters(), lr=0.01)
        opt_s = torch.optim.SGD(surr_net.parameters(), lr=0.01)
        cos_hist = []
        for epoch in range(300):
            opt_t.zero_grad()
            loss_fn(true_net(X), y).backward()
            g_true = get_flat_gradients(true_net)
            surr_net.load_state_dict(true_net.state_dict())
            opt_s.zero_grad()
            loss_fn(surr_net(X), y).backward()
            g_surr = get_flat_gradients(surr_net)
            cos_hist.append(cosine_similarity(g_true, g_surr))
            opt_t.step()
        c = detect_collapse(cos_hist)
        if c is not None and CKPT_LO + 20 <= c["epoch"] <= CKPT_HI - 20:
            print(f"  scout seed {s}: collapse @ {c['epoch']} (chosen)")
            return s, c
        else:
            print(f"  scout seed {s}: "
                  f"{('collapse @ ' + str(c['epoch'])) if c else 'no collapse'}"
                  f"  (skip)")
    raise RuntimeError("no clean collapsed seed found")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== ANALYSIS 3: LOSS LANDSCAPE ===")
    print(f"Using device: {DEVICE}")

    print("\nScouting for a cleanly collapsing seed...")
    seed, c0 = _pick_clean_seed()
    print(f"Selected seed = {seed}, scout collapse epoch = {c0['epoch']}")

    # Retrain with checkpoints
    print(f"\nRe-running seed {seed} with checkpoints "
          f"[{CKPT_LO}, {CKPT_HI}]...")
    cfg = _cfg(seed)
    history, ckpts = _train_with_checkpoints(cfg, CKPT_LO, CKPT_HI)
    c = detect_collapse(history["cosine_sim"])
    if c is None:
        raise RuntimeError("retrain did not produce a collapse "
                           "(this is the GPU-non-determinism bite again)")
    ce = c["epoch"]
    target_eps = [ce - 20, ce, ce + 20]
    target_eps = [max(CKPT_LO, min(CKPT_HI, e)) for e in target_eps]
    print(f"Collapse epoch in retrain = {ce}, target epochs = {target_eps}")

    # Build a working model & build directions from the CENTER (=collapse) ckpt.
    work = MLPTrue(2, cfg["hidden_dim"], cfg["num_layers"],
                   cfg["beta_f"]).to(DEVICE)
    work.load_state_dict(ckpts[ce])
    center_w = [p.detach().clone() for p in work.parameters()]

    d1, d2 = _make_directions(center_w, seed=20260429)

    # Loss / data once
    X, y = get_dataset("moons")
    loss_fn = nn.BCEWithLogitsLoss()

    # ---------- Headline: filled contours at ce-20, ce, ce+20 ----------
    print("\nComputing landscapes (resolution="
          f"{RESOLUTION}x{RESOLUTION}, range=±{RANGE_VAL})...")
    grids: List[np.ndarray] = []
    cos_at: List[float] = []
    for ep in target_eps:
        work.load_state_dict(ckpts[ep])
        cw = [p.detach().clone() for p in work.parameters()]
        grid = plot_loss_landscape(work, loss_fn, X, y, cw, d1, d2)
        grids.append(grid)
        cos_at.append(history["cosine_sim"][ep])
        print(f"  epoch {ep}: loss range "
              f"[{grid.min():.3f}, {grid.max():.3f}]   "
              f"cos sim @ epoch = {cos_at[-1]:.3f}")

    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)
    levels = np.linspace(vmin, vmax, 20)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = [-RANGE_VAL, RANGE_VAL, -RANGE_VAL, RANGE_VAL]
    for ax, grid, ep, cs in zip(axes, grids, target_eps, cos_at):
        cf = ax.contourf(np.linspace(-RANGE_VAL, RANGE_VAL, RESOLUTION),
                         np.linspace(-RANGE_VAL, RANGE_VAL, RESOLUTION),
                         grid, levels=levels, cmap="viridis")
        ax.scatter([0], [0], color="red", s=80, edgecolor="white",
                   zorder=10, label="weights")
        ax.set_xlabel(r"$\alpha$ (along $d_1$)")
        ax.set_ylabel(r"$\beta$ (along $d_2$)")
        ax.set_title(f"epoch {ep}\ncos sim = {cs:.3f}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02, label="loss")
    fig.suptitle(f"Loss landscape (filter-normalised) around collapse "
                 f"(seed {seed}, collapse @ epoch {ce})", fontsize=12)
    out1 = os.path.join(FIG_DIR, "landscape_pre_during_post_collapse.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---------- Eigenvector overlay ----------
    print("\nComputing top-2 Hessian eigenvectors at each checkpoint "
          "(LinearOperator + eigsh)...")
    d1_flat = _flatten(d1)
    d2_flat = _flatten(d2)
    eigs_per_ep: List[Tuple[np.ndarray, np.ndarray]] = []
    for ep in target_eps:
        work.load_state_dict(ckpts[ep])
        vals, vecs = top_hessian_eigvecs(work, loss_fn, X, y, k=2)
        eigs_per_ep.append((vals, vecs))
        print(f"  epoch {ep}: top eigenvalues = "
              f"[{vals[0]:.3f}, {vals[1]:.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, grid, ep, cs, (vals, vecs) in zip(
        axes, grids, target_eps, cos_at, eigs_per_ep
    ):
        cs_lines = ax.contour(
            np.linspace(-RANGE_VAL, RANGE_VAL, RESOLUTION),
            np.linspace(-RANGE_VAL, RANGE_VAL, RESOLUTION),
            grid, levels=15, cmap="viridis", linewidths=0.9,
        )
        ax.clabel(cs_lines, inline=True, fontsize=7, fmt="%.2f")
        ax.scatter([0], [0], color="red", s=70, edgecolor="white", zorder=10)
        # Project eigenvectors onto (d1, d2) plane and scale by sign(eig)*log(|eig|+1)
        for k_eig in range(2):
            a, b = _project_eigvec_to_dirs(vecs[:, k_eig], d1_flat, d2_flat)
            scale = np.sign(vals[k_eig]) * np.log(abs(vals[k_eig]) + 1.0) * 0.05
            ax.arrow(0, 0, a * scale, b * scale,
                     width=0.01, head_width=0.05,
                     color=("crimson" if k_eig == 0 else "navy"),
                     length_includes_head=True,
                     label=fr"$v_{k_eig+1}$ ($\lambda$={vals[k_eig]:.1f})")
        ax.set_xlim(-RANGE_VAL, RANGE_VAL)
        ax.set_ylim(-RANGE_VAL, RANGE_VAL)
        ax.set_aspect("equal")
        ax.set_title(f"epoch {ep}   cos sim = {cs:.3f}")
        ax.set_xlabel(r"$\alpha$ (along $d_1$)")
        ax.set_ylabel(r"$\beta$ (along $d_2$)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Loss-surface contours + top-2 Hessian eigenvectors "
                 f"projected onto the visualisation plane (seed {seed})",
                 fontsize=11)
    out2 = os.path.join(FIG_DIR, "landscape_eigenvalue_overlay.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # ---------- Numeric summary ----------
    print("\n=== Checkpoint summary ===")
    print(f"{'epoch':<8}{'loss':<14}{'cos_sim':<12}"
          f"{'||g_true||':<14}{'lambda_max':<12}")
    print("-" * 60)
    for ep, (vals, _) in zip(target_eps, eigs_per_ep):
        print(f"{ep:<8d}"
              f"{history['true_loss'][ep]:<14.5f}"
              f"{history['cosine_sim'][ep]:<12.4f}"
              f"{history['grad_norm_true'][ep]:<14.4f}"
              f"{vals[0]:<12.3f}")


if __name__ == "__main__":
    main()
