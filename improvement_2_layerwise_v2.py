"""Improvement #2: layer-wise alignment drop table using the *same*
collapse-detection criterion that's applied to the global cosine
similarity, but per layer with its own per-layer baseline.

Original analysis used a fixed cosine = 0.6 cutoff which is meaningless
for layers whose cos sim is ~0.1 throughout training (it just reports
"epoch 50" for everything). We instead use ``detect_collapse`` itself
on each layer's own cos-sim trajectory and report the per-layer
drop epoch.
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


def _cfg(seed):
    return {"hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
            "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed}


def _per_layer_grads(net):
    out = []
    for layer in net.layers:
        parts = []
        if layer.weight.grad is not None:
            parts.append(layer.weight.grad.detach().cpu().flatten())
        if layer.bias is not None and layer.bias.grad is not None:
            parts.append(layer.bias.grad.detach().cpu().flatten())
        out.append(torch.cat(parts) if parts else torch.zeros(0))
    return out


def _train(config):
    torch.manual_seed(config["seed"])
    X, y = get_dataset(config["dataset"])
    true_net = MLPTrue(2, 64, 4, 50).to(DEVICE)
    surr_net = MLPSurrogate(2, 64, 4, 50, 5).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())
    loss_fn = nn.BCEWithLogitsLoss()
    opt_t = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_s = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])
    n_layers = len(true_net.layers)
    history: Dict[str, List] = {
        "epoch": [], "cos_total": [],
        "cos_per_layer": [],
        "grad_norm_per_layer": [],
    }
    for epoch in range(config["n_epochs"]):
        opt_t.zero_grad()
        loss_fn(true_net(X), y).backward()
        gt_layers = _per_layer_grads(true_net)
        surr_net.load_state_dict(true_net.state_dict())
        opt_s.zero_grad()
        loss_fn(surr_net(X), y).backward()
        gs_layers = _per_layer_grads(surr_net)
        cos_layers, gnorm_layers = [], []
        for gt, gs in zip(gt_layers, gs_layers):
            cos_layers.append(cosine_similarity(gt, gs))
            gnorm_layers.append(float(torch.norm(gt).item()))
        history["epoch"].append(epoch)
        history["cos_total"].append(cosine_similarity(
            torch.cat(gt_layers), torch.cat(gs_layers)))
        history["cos_per_layer"].append(cos_layers)
        history["grad_norm_per_layer"].append(gnorm_layers)
        opt_t.step()
    history["n_layers"] = n_layers
    return history


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== IMPROVEMENT #2: per-layer drop detection (same algorithm "
          "as global collapse) ===")
    print(f"Using device: {DEVICE}")

    histories: List[Dict] = []
    collapses: List[Optional[Dict]] = []
    for s in range(N_SEEDS):
        print(f"  seed {s} ...", end="", flush=True)
        h = _train(_cfg(s))
        c = detect_collapse(h["cos_total"])
        histories.append(h); collapses.append(c)
        print(f"  global collapse: "
              f"{'ep ' + str(c['epoch']) if c else 'none'}")

    n_hidden = histories[0]["n_layers"] - 1  # exclude readout

    # Apply detect_collapse to each layer's per-epoch cos sim.
    print("\n=== Per-layer drop epochs (detect_collapse on each layer) ===")
    header = (f"{'seed':<5}{'global':<10}"
              + "".join(f"L{li}_drop ".ljust(11) for li in range(n_hidden))
              + "".join(f"L{li}_meanC ".ljust(11) for li in range(n_hidden)))
    print(header)
    print("-" * len(header))
    layer_drops: List[List[Optional[int]]] = []
    layer_means: List[List[float]] = []
    for s, h in enumerate(histories):
        cos_arr = np.asarray(h["cos_per_layer"])  # [E, L]
        row_drop = []
        row_mean = []
        for li in range(n_hidden):
            ev = detect_collapse(cos_arr[:, li].tolist(),
                                 window=20, threshold=0.15, min_epoch=50)
            row_drop.append(ev["epoch"] if ev else None)
            row_mean.append(float(np.nanmean(cos_arr[:, li])))
        layer_drops.append(row_drop)
        layer_means.append(row_mean)
        gs = collapses[s]
        gs_str = str(gs["epoch"]) if gs else "none"
        drop_strs = "".join(
            (str(d) if d is not None else "-").ljust(11) for d in row_drop
        )
        mean_strs = "".join(f"{m:>+.3f}".ljust(11) for m in row_mean)
        print(f"{s:<5}{gs_str:<10}{drop_strs}{mean_strs}")

    # Aggregate ordering: among collapsed seeds, what is the median epoch
    # at which each layer drops? (None counted as "did not drop".)
    print("\n=== Median per-layer drop epoch among COLLAPSED seeds ===")
    coll_idx = [i for i, c in enumerate(collapses) if c is not None]
    print(f"  collapsed seeds: {coll_idx}")
    for li in range(n_hidden):
        vals = [layer_drops[i][li] for i in coll_idx
                if layer_drops[i][li] is not None]
        if vals:
            print(f"  layer {li}: median={int(np.median(vals))}  "
                  f"n_dropped={len(vals)}/{len(coll_idx)}  "
                  f"min={min(vals)} max={max(vals)}")
        else:
            print(f"  layer {li}: never drops by detect_collapse criterion")

    # Per-layer mean gradient L2 norm — confirms the §5.2 claim that
    # output-side layer dominates total gradient magnitude.
    print("\n=== Mean per-layer gradient L2 norm "
          "(averaged over epochs, across seeds) ===")
    n_layers = histories[0]["n_layers"]
    for li in range(n_layers):
        gn = np.mean([
            np.mean([h["grad_norm_per_layer"][e][li]
                     for e in range(len(h["epoch"]))])
            for h in histories
        ])
        tag = "readout" if li == n_layers - 1 else f"hidden L{li}"
        print(f"  {tag:<12}  mean ||g||_2 = {gn:.4f}")

    # Visual: per-layer mean cos sim alongside the global collapse epoch.
    epochs = np.asarray(histories[0]["epoch"])
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_hidden))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for li in range(n_hidden):
        per_seed = np.stack([
            [h["cos_per_layer"][e][li] for e in range(len(h["epoch"]))]
            for h in histories
        ])
        ax.plot(epochs, np.nanmean(per_seed, axis=0), color=palette[li],
                linewidth=2, label=f"hidden layer {li}")
        sd = np.nanstd(per_seed, axis=0, ddof=1)
        ax.fill_between(epochs,
                        np.nanmean(per_seed, axis=0) - sd,
                        np.nanmean(per_seed, axis=0) + sd,
                        color=palette[li], alpha=0.12)
    if any(c is not None for c in collapses):
        m = float(np.mean([c["epoch"] for c in collapses if c is not None]))
        ax.axvline(m, color="red", linestyle="--",
                   label=f"global mean collapse ≈ {m:.0f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-layer cosine similarity")
    ax.set_title("Per-layer cosine similarity (mean ± std across 10 seeds)")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "layerwise_v2_mean_per_layer.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
