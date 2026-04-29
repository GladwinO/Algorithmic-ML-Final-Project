"""Improvement #1: re-do the saturation analysis with a derivative-magnitude
criterion instead of an output-magnitude criterion.

Original analysis declared a unit "saturated" if sigmoid(beta_f * x) > 0.99
or < 0.01. With beta_f=50 that corresponds to |x| > ~0.092, where the true
sigmoid derivative is still beta_f * sigma * (1-sigma) ~ 50 * 0.01 * 0.99
~ 0.5, which is not small in absolute terms (max derivative is beta_f/4 = 12.5,
so 0.5 is 4% of max). The output threshold therefore overcounts.

The right notion of saturation, *for the cosine-similarity question*, is:
"the true derivative is so small that it can no longer match the surrogate
derivative". We measure the *fraction of units whose true derivative is
below a small fraction of the max derivative*, for several thresholds, and
also report the actual mean true / surrogate derivative ratio per layer.
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
from sg_experiment.metrics import cosine_similarity, get_flat_gradients
from sg_experiment.models import MLPSurrogate, MLPTrue

FIG_DIR = "figures"
N_SEEDS = 10
# Fractions of the maximum true derivative (beta_f/4 = 12.5 with beta_f=50).
DERIV_THRESH_FRACS = [0.01, 0.05, 0.10]


def _cfg(seed: int) -> Dict:
    return {"hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
            "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed}


def _measure(true_net: MLPTrue, X: torch.Tensor, beta_f: float, beta_sg: float
             ) -> Dict[str, List[float]]:
    """Per layer: list of saturation fractions (one per threshold) and mean
    true/SG derivative ratio."""
    max_true_deriv = beta_f / 4.0
    sat_per_thresh: List[List[float]] = [[] for _ in DERIV_THRESH_FRACS]
    ratio_per_layer: List[float] = []
    with torch.no_grad():
        h = X
        for layer in true_net.layers[:-1]:
            pre = layer(h)
            sig = torch.sigmoid(beta_f * pre)
            true_d = beta_f * sig * (1 - sig)
            sg = torch.sigmoid(beta_sg * pre)
            sg_d = beta_sg * sg * (1 - sg)
            for k, frac in enumerate(DERIV_THRESH_FRACS):
                mask = true_d < frac * max_true_deriv
                sat_per_thresh[k].append(float(mask.float().mean().item()))
            ratio = (true_d.mean() / (sg_d.mean() + 1e-12)).item()
            ratio_per_layer.append(float(ratio))
            h = sig
    return {"sat": sat_per_thresh, "ratio": ratio_per_layer}


def _train(config: Dict) -> Dict[str, List]:
    torch.manual_seed(config["seed"])
    X, y = get_dataset(config["dataset"])
    true_net = MLPTrue(2, 64, 4, config["beta_f"]).to(DEVICE)
    surr_net = MLPSurrogate(2, 64, 4, config["beta_f"], config["beta_sg"]).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())
    loss_fn = nn.BCEWithLogitsLoss()
    opt_t = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_s = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])
    history: Dict[str, List] = {
        "epoch": [], "cosine_sim": [],
        "sat_per_thresh_per_layer": [],   # epoch -> [n_thresh][n_layer]
        "ratio_per_layer": [],            # epoch -> [n_layer]
    }
    for epoch in range(config["n_epochs"]):
        opt_t.zero_grad()
        loss_fn(true_net(X), y).backward()
        g_t = get_flat_gradients(true_net)
        surr_net.load_state_dict(true_net.state_dict())
        opt_s.zero_grad()
        loss_fn(surr_net(X), y).backward()
        g_s = get_flat_gradients(surr_net)
        m = _measure(true_net, X, config["beta_f"], config["beta_sg"])
        history["epoch"].append(epoch)
        history["cosine_sim"].append(cosine_similarity(g_t, g_s))
        history["sat_per_thresh_per_layer"].append(m["sat"])
        history["ratio_per_layer"].append(m["ratio"])
        opt_t.step()
    return history


def _plot_overview(histories, collapses, out_path):
    n_thresh = len(DERIV_THRESH_FRACS)
    n_layers = len(histories[0]["sat_per_thresh_per_layer"][0][0])
    fig, axes = plt.subplots(n_thresh, n_layers,
                             figsize=(3 * n_layers, 2.6 * n_thresh),
                             sharex=True, sharey=True)
    epochs = np.asarray(histories[0]["epoch"])
    detected = [c["epoch"] for c in collapses if c is not None]
    mean_collapse = float(np.mean(detected)) if detected else None

    for k, frac in enumerate(DERIV_THRESH_FRACS):
        for li in range(n_layers):
            ax = axes[k, li]
            arr = np.stack([
                [h["sat_per_thresh_per_layer"][e][k][li]
                 for e in range(len(h["epoch"]))]
                for h in histories
            ])
            for run in arr:
                ax.plot(epochs, run, color="lightgray", linewidth=0.7, alpha=0.7)
            ax.plot(epochs, arr.mean(0), color="navy", linewidth=2)
            if mean_collapse is not None:
                ax.axvline(mean_collapse, color="red", linestyle="--",
                           alpha=0.7)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.set_title(f"hidden layer {li}")
            if li == 0:
                ax.set_ylabel(f"thresh = {frac:.0%}\n of max deriv\nfraction")
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    fig.suptitle("Fraction of units with true derivative below threshold "
                 "(by layer, by threshold)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ratio(histories, collapses, out_path):
    n_layers = len(histories[0]["ratio_per_layer"][0])
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.asarray(histories[0]["epoch"])
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))
    detected = [c["epoch"] for c in collapses if c is not None]
    mean_collapse = float(np.mean(detected)) if detected else None
    for li in range(n_layers):
        arr = np.stack([
            [h["ratio_per_layer"][e][li] for e in range(len(h["epoch"]))]
            for h in histories
        ])
        ax.plot(epochs, arr.mean(0), color=palette[li], linewidth=2,
                label=f"layer {li}")
        sd = arr.std(0, ddof=1)
        ax.fill_between(epochs, arr.mean(0) - sd, arr.mean(0) + sd,
                        color=palette[li], alpha=0.15)
    if mean_collapse is not None:
        ax.axvline(mean_collapse, color="red", linestyle="--",
                   label=f"mean collapse ≈ {mean_collapse:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean(true_deriv) / mean(SG_deriv)")
    ax.set_title("Per-layer mean true / surrogate derivative ratio "
                 "(log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== IMPROVEMENT #1: derivative-magnitude saturation ===")
    print(f"Using device: {DEVICE}")
    print(f"DERIV_THRESH_FRACS = {DERIV_THRESH_FRACS}  "
          f"(of max true deriv = beta_f/4)\n")
    histories: List[Dict] = []
    collapses: List[Optional[Dict]] = []
    for s in range(N_SEEDS):
        print(f"  seed {s} ...", end="", flush=True)
        h = _train(_cfg(s))
        c = detect_collapse(h["cosine_sim"])
        histories.append(h)
        collapses.append(c)
        # last-epoch saturation summary at 5% threshold
        sat5 = np.mean(h["sat_per_thresh_per_layer"][-1][1])
        ratio_last = np.mean(h["ratio_per_layer"][-1])
        print(f"  final cos={np.mean(h['cosine_sim'][-50:]):.3f}  "
              f"sat@5%={sat5:.3f}  true/SG={ratio_last:.3e}  "
              f"collapse={'ep ' + str(c['epoch']) if c else 'none'}")

    _plot_overview(histories, collapses,
                   os.path.join(FIG_DIR, "saturation_v2_by_threshold.png"))
    _plot_ratio(histories, collapses,
                os.path.join(FIG_DIR, "saturation_v2_deriv_ratio.png"))

    # Per-seed summary on the 5% threshold (middle of DERIV_THRESH_FRACS).
    print("\n=== Per-seed saturation @ 5% threshold (mean across layers) ===")
    print(f"{'seed':<5}{'collapse_ep':<13}{'sat_pre':<11}{'sat_post':<11}"
          f"{'ratio_pre':<13}{'ratio_post':<13}{'corr(sat,cos)':<14}")
    print("-" * 80)
    for s, (h, c) in enumerate(zip(histories, collapses)):
        sat = np.asarray(
            [np.mean(h["sat_per_thresh_per_layer"][e][1])
             for e in range(len(h["epoch"]))])
        ratio = np.asarray(
            [np.mean(h["ratio_per_layer"][e])
             for e in range(len(h["epoch"]))])
        cos = np.asarray(h["cosine_sim"])
        corr = float(np.corrcoef(sat, cos)[0, 1])
        if c is not None:
            ep = c["epoch"]
            sp = sat[max(0, ep - 50):ep].mean()
            sq = sat[ep:ep + 50].mean()
            rp = ratio[max(0, ep - 50):ep].mean()
            rq = ratio[ep:ep + 50].mean()
            ep_s = str(ep)
        else:
            sp, sq = sat[180:230].mean(), sat[230:280].mean()
            rp, rq = ratio[180:230].mean(), ratio[230:280].mean()
            ep_s = "none"
        print(f"{s:<5}{ep_s:<13}{sp:<11.4f}{sq:<11.4f}"
              f"{rp:<13.3e}{rq:<13.3e}{corr:<14.3f}")

    # Pooled correlation across thresholds.
    print("\n=== Global Pearson r(sat_at_threshold, cos sim) "
          "over all (seed, epoch) ===")
    for k, frac in enumerate(DERIV_THRESH_FRACS):
        xs, ys = [], []
        for h in histories:
            sats = np.asarray(
                [np.mean(h["sat_per_thresh_per_layer"][e][k])
                 for e in range(len(h["epoch"]))])
            xs.append(sats)
            ys.append(np.asarray(h["cosine_sim"]))
        x = np.concatenate(xs); y = np.concatenate(ys)
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"  threshold = {frac:.0%} of max deriv:  r = {r:.3f}")


if __name__ == "__main__":
    main()
