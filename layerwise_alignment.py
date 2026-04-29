"""ANALYSIS 2: Layer-wise gradient alignment decomposition.

For each Linear layer in the network, separately concatenate that layer's
weight and bias gradients into a per-layer gradient vector, and report
cosine similarity / sign agreement against the corresponding true-gradient
slice. The point is to localise the collapse: does the deepest hidden
layer drift first and pull the others with it (consistent with the
chain-length finding) or does collapse appear in all layers at once?
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collapse_detection import detect_collapse
from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.metrics import cosine_similarity, sign_agreement
from sg_experiment.models import MLPSurrogate, MLPTrue

FIG_DIR = "figures"
N_SEEDS = 10
DROP_THRESH = 0.6


def _cfg(seed: int) -> Dict:
    return {
        "hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
        "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed,
    }


def _per_layer_grads(net) -> List[torch.Tensor]:
    """One flat gradient tensor per Linear layer (weight ++ bias)."""
    out: List[torch.Tensor] = []
    for layer in net.layers:
        parts = []
        if layer.weight.grad is not None:
            parts.append(layer.weight.grad.detach().cpu().flatten())
        if layer.bias is not None and layer.bias.grad is not None:
            parts.append(layer.bias.grad.detach().cpu().flatten())
        out.append(torch.cat(parts) if parts else torch.zeros(0))
    return out


def _train_layerwise(config: Dict) -> Dict[str, List]:
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

    n_layers = len(true_net.layers)  # hidden + output  (= 5 when num_layers=4)
    history: Dict[str, List] = {
        "epoch": [], "cosine_sim_total": [],
        "cos_per_layer": [], "sign_per_layer": [],
    }

    for epoch in range(config["n_epochs"]):
        opt_true.zero_grad()
        loss_true = loss_fn(true_net(X), y)
        loss_true.backward()
        g_true_layers = _per_layer_grads(true_net)

        surr_net.load_state_dict(true_net.state_dict())
        opt_surr.zero_grad()
        loss_surr = loss_fn(surr_net(X), y)
        loss_surr.backward()
        g_surr_layers = _per_layer_grads(surr_net)

        cos_layers, sign_layers = [], []
        for gt, gs in zip(g_true_layers, g_surr_layers):
            if gt.numel() == 0:
                cos_layers.append(float("nan"))
                sign_layers.append(float("nan"))
            else:
                cos_layers.append(cosine_similarity(gt, gs))
                sign_layers.append(sign_agreement(gt, gs))

        # Total cosine across the whole flat gradient (for collapse detection).
        flat_t = torch.cat(g_true_layers)
        flat_s = torch.cat(g_surr_layers)

        history["epoch"].append(epoch)
        history["cosine_sim_total"].append(cosine_similarity(flat_t, flat_s))
        history["cos_per_layer"].append(cos_layers)
        history["sign_per_layer"].append(sign_layers)

        opt_true.step()

    history["n_layers"] = n_layers
    return history


# ------------------------- plots -------------------------


def _plot_layerwise_all_seeds(histories, collapses, out_path):
    n_layers = histories[0]["n_layers"]
    # Restrict figure to hidden layers (exclude final readout) for clarity.
    n_hidden = n_layers - 1
    fig, axes = plt.subplots(n_hidden, 1, figsize=(9, 2.4 * n_hidden),
                             sharex=True)
    epochs = np.asarray(histories[0]["epoch"])
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_hidden))
    detected = [c["epoch"] for c in collapses if c is not None]
    mean_collapse = float(np.mean(detected)) if detected else None

    for li in range(n_hidden):
        ax = axes[li]
        per_seed = np.stack([
            [h["cos_per_layer"][e][li] for e in range(len(h["epoch"]))]
            for h in histories
        ])
        for run in per_seed:
            ax.plot(epochs, run, color="lightgray", linewidth=0.7, alpha=0.7)
        ax.plot(epochs, np.nanmean(per_seed, axis=0), color=palette[li],
                linewidth=2, label=f"mean (layer {li})")
        if mean_collapse is not None:
            ax.axvline(mean_collapse, color="red", linestyle="--",
                       label=f"mean collapse ≈ {mean_collapse:.0f}")
        ax.set_ylabel(f"layer {li} cos")
        ax.set_ylim(-0.2, 1.05)
        ax.set_title(f"Per-layer cosine similarity, hidden layer {li} "
                     f"(0=input-side)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_layerwise_collapsed(histories, collapses, out_path):
    coll_idx = [i for i, c in enumerate(collapses) if c is not None]
    if not coll_idx:
        print("[layerwise] no collapses; skipping collapsed-overlay plot")
        return
    n_layers = histories[0]["n_layers"]
    n_hidden = n_layers - 1
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.asarray(histories[0]["epoch"])
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_hidden))

    for li in range(n_hidden):
        per_seed = np.stack([
            [histories[i]["cos_per_layer"][e][li]
             for e in range(len(histories[i]["epoch"]))]
            for i in coll_idx
        ])
        ax.plot(epochs, np.nanmean(per_seed, axis=0), color=palette[li],
                linewidth=2.2, label=f"hidden layer {li}")
        if per_seed.shape[0] > 1:
            sd = np.nanstd(per_seed, axis=0, ddof=1)
            ax.fill_between(epochs,
                            np.nanmean(per_seed, axis=0) - sd,
                            np.nanmean(per_seed, axis=0) + sd,
                            color=palette[li], alpha=0.15)
    eps = [collapses[i]["epoch"] for i in coll_idx]
    ax.axvline(float(np.mean(eps)), color="red", linestyle="--", alpha=0.7,
               label=f"mean collapse ≈ {np.mean(eps):.0f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Layer-wise cosine similarity (mean across {len(coll_idx)} "
                 f"collapsed seeds)")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_propagation_heatmap(histories, collapses, out_path,
                              ep_lo=200, ep_hi=300):
    coll_idx = [i for i, c in enumerate(collapses) if c is not None]
    if not coll_idx:
        print("[layerwise] no collapses; skipping heatmap")
        return
    n_hidden = histories[0]["n_layers"] - 1
    n = len(coll_idx)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows),
                             squeeze=False)

    for k, idx in enumerate(coll_idx):
        h = histories[idx]
        c = collapses[idx]
        eps = np.asarray(h["epoch"])
        mask = (eps >= ep_lo) & (eps < ep_hi)
        cos_per = np.asarray(h["cos_per_layer"])[mask, :n_hidden]  # [E, L]
        ax = axes[k // cols, k % cols]
        im = ax.imshow(cos_per.T, aspect="auto", origin="lower",
                       extent=[ep_lo, ep_hi, -0.5, n_hidden - 0.5],
                       cmap="RdYlBu", vmin=-0.2, vmax=1.0)
        ax.axvline(c["epoch"], color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"seed {idx}, collapse @ {c['epoch']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hidden layer (0=input-side)")
        ax.set_yticks(range(n_hidden))
        plt.colorbar(im, ax=ax, label="cos sim")
    for k in range(len(coll_idx), rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle("Collapse propagation across layers (per collapsed seed)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== ANALYSIS 2: LAYERWISE ALIGNMENT ===")
    print(f"Using device: {DEVICE}")
    print(f"Running {N_SEEDS} seeds...")

    histories: List[Dict] = []
    collapses: List[Optional[Dict]] = []
    for s in range(N_SEEDS):
        print(f"  seed {s} ... ", end="", flush=True)
        h = _train_layerwise(_cfg(s))
        c = detect_collapse(h["cosine_sim_total"])
        histories.append(h)
        collapses.append(c)
        print(f"final cos={np.mean(h['cosine_sim_total'][-50:]):.3f}  "
              f"collapse={'ep ' + str(c['epoch']) if c else 'none'}")

    _plot_layerwise_all_seeds(
        histories, collapses,
        os.path.join(FIG_DIR, "layerwise_cosine_all_seeds.png"))
    _plot_layerwise_collapsed(
        histories, collapses,
        os.path.join(FIG_DIR, "layerwise_cosine_collapsed_seeds.png"))
    _plot_propagation_heatmap(
        histories, collapses,
        os.path.join(FIG_DIR, "collapse_propagation_heatmap.png"))

    n_hidden = histories[0]["n_layers"] - 1
    print("\n=== First epoch each hidden layer drops below "
          f"cos < {DROP_THRESH} (collapsed seeds) ===")
    header = f"{'seed':<5}{'collapse_ep':<13}" + "".join(
        f"L{li}_drop_ep ".ljust(14) for li in range(n_hidden))
    print(header)
    print("-" * len(header))
    for s, (h, c) in enumerate(zip(histories, collapses)):
        if c is None:
            continue
        cos_per = np.asarray(h["cos_per_layer"])  # [E, L_total]
        eps = np.asarray(h["epoch"])
        row = f"{s:<5}{c['epoch']:<13}"
        for li in range(n_hidden):
            below = np.where(cos_per[:, li] < DROP_THRESH)[0]
            below = below[below >= 50]  # ignore very early random init
            ep_drop = int(eps[below[0]]) if len(below) else -1
            row += f"{ep_drop:<14}"
        print(row)


if __name__ == "__main__":
    main()
