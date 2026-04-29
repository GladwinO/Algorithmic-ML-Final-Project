"""ANALYSIS 1: Neuron saturation tracking.

Hypothesis: with beta_f=50, hidden-unit sigmoids saturate during training.
A saturated true sigmoid (sigmoid(50*x) ~ 0 or 1) has near-zero derivative,
while the surrogate (sigmoid(5*x)) is much flatter and has non-trivial
derivative everywhere. The systematic disagreement at saturated units
should drive the global cosine-similarity collapse.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collapse_detection import detect_collapse
from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.metrics import (
    cosine_similarity,
    get_flat_gradients,
    sign_agreement,
    relative_magnitude,
)
from sg_experiment.models import MLPSurrogate, MLPTrue

FIG_DIR = "figures"
N_SEEDS = 10
SAT_HIGH = 0.99
SAT_LOW = 0.01


def _cfg(seed: int) -> Dict:
    return {
        "hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
        "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed,
    }


def _measure_saturation(true_net: MLPTrue, X: torch.Tensor,
                        beta_f: float, beta_sg: float
                        ) -> Tuple[List[float], List[float]]:
    """Return (sat_frac_per_layer, mean_abs_sg_minus_true_deriv_at_sat).

    Pre-activations of the hidden layers are computed by running the same
    forward pass as MLPTrue but recording the linear outputs before the
    nonlinearity for each hidden Linear layer.
    """
    sat_fracs: List[float] = []
    deriv_diffs: List[float] = []
    with torch.no_grad():
        h = X
        for layer in true_net.layers[:-1]:
            pre = layer(h)
            sig = torch.sigmoid(beta_f * pre)
            sat_mask = (sig > SAT_HIGH) | (sig < SAT_LOW)
            sat_fracs.append(float(sat_mask.float().mean().item()))

            # Per-element derivatives of the two backward passes.
            true_deriv = beta_f * sig * (1 - sig)
            sg = torch.sigmoid(beta_sg * pre)
            sg_deriv = beta_sg * sg * (1 - sg)
            if sat_mask.any():
                diff = (sg_deriv - true_deriv).abs()[sat_mask]
                deriv_diffs.append(float(diff.mean().item()))
            else:
                deriv_diffs.append(0.0)

            # advance true forward for next layer
            h = sig
    return sat_fracs, deriv_diffs


def _train_with_saturation(config: Dict) -> Dict[str, List]:
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

    n_hidden = config["num_layers"]
    history: Dict[str, List] = {
        "epoch": [], "cosine_sim": [], "true_loss": [], "grad_norm_true": [],
        "sat_per_layer": [],            # list of [n_hidden] per epoch
        "deriv_diff_per_layer": [],     # list of [n_hidden] per epoch
    }

    for epoch in range(config["n_epochs"]):
        true_net.train()
        opt_true.zero_grad()
        pred_true = true_net(X)
        loss_true = loss_fn(pred_true, y)
        loss_true.backward()
        g_true = get_flat_gradients(true_net)

        surr_net.load_state_dict(true_net.state_dict())
        surr_net.train()
        opt_surr.zero_grad()
        pred_surr = surr_net(X)
        loss_surr = loss_fn(pred_surr, y)
        loss_surr.backward()
        g_surr = get_flat_gradients(surr_net)

        sat, dd = _measure_saturation(true_net, X, config["beta_f"],
                                      config["beta_sg"])

        history["epoch"].append(epoch)
        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["true_loss"].append(float(loss_true.item()))
        history["grad_norm_true"].append(float(torch.norm(g_true).item()))
        history["sat_per_layer"].append(sat)
        history["deriv_diff_per_layer"].append(dd)

        opt_true.step()

    return history


# ------------------------- plots -------------------------


def _plot_saturation_over_training(histories, collapses, out_path):
    n_hidden = len(histories[0]["sat_per_layer"][0])
    fig, axes = plt.subplots(n_hidden, 1, figsize=(9, 2.6 * n_hidden),
                             sharex=True)
    epochs = np.asarray(histories[0]["epoch"])

    detected = [c["epoch"] for c in collapses if c is not None]
    mean_collapse = float(np.mean(detected)) if detected else None

    for li in range(n_hidden):
        ax = axes[li]
        all_sats = np.stack(
            [[h["sat_per_layer"][e][li] for e in range(len(h["epoch"]))]
             for h in histories]
        )
        for run in all_sats:
            ax.plot(epochs, run, color="lightgray", linewidth=0.7, alpha=0.7)
        ax.plot(epochs, all_sats.mean(0), color="navy", linewidth=2,
                label="mean across seeds")
        if mean_collapse is not None:
            ax.axvline(mean_collapse, color="red", linestyle="--",
                       label=f"mean collapse ≈ {mean_collapse:.0f}")
        ax.set_ylabel(f"layer {li} sat. frac")
        ax.set_title(f"Hidden layer {li}: fraction of saturated units")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_saturation_vs_cosine(histories, out_path):
    xs, ys = [], []
    for h in histories:
        sats = np.asarray(h["sat_per_layer"]).mean(axis=1)
        cos = np.asarray(h["cosine_sim"])
        xs.append(sats); ys.append(cos)
    x = np.concatenate(xs); y = np.concatenate(ys)
    coef = float(np.corrcoef(x, y)[0, 1])
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=4, alpha=0.25, color="steelblue")
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, m * xx + b, color="crimson", linewidth=2,
            label=f"fit: y = {m:.2f}x + {b:.2f}")
    ax.set_xlabel("Mean saturation fraction (across hidden layers)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Saturation vs cosine similarity   (Pearson r = {coef:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return coef


def _plot_collapsed_vs_not_bars(histories, collapses, out_path, window=50):
    coll_pre, coll_post, ncoll_anchor = [], [], []
    anchor_eps = [c["epoch"] for c in collapses if c is not None]
    if not anchor_eps:
        print("[saturation] no collapses; skipping bar plot")
        return
    median_anchor = int(np.median(anchor_eps))

    for h, c in zip(histories, collapses):
        sats = np.asarray(h["sat_per_layer"]).mean(axis=1)
        if c is not None:
            ep = c["epoch"]
            pre = sats[max(0, ep - window):ep]
            post = sats[ep:ep + window]
            coll_pre.append(float(pre.mean()))
            coll_post.append(float(post.mean()))
        else:
            window_vals = sats[max(0, median_anchor - window):
                               median_anchor + window]
            ncoll_anchor.append(float(window_vals.mean()))

    means = [
        float(np.mean(coll_pre)) if coll_pre else 0.0,
        float(np.mean(coll_post)) if coll_post else 0.0,
        float(np.mean(ncoll_anchor)) if ncoll_anchor else 0.0,
    ]
    sems = [
        float(np.std(coll_pre, ddof=1) / np.sqrt(len(coll_pre)))
            if len(coll_pre) > 1 else 0.0,
        float(np.std(coll_post, ddof=1) / np.sqrt(len(coll_post)))
            if len(coll_post) > 1 else 0.0,
        float(np.std(ncoll_anchor, ddof=1) / np.sqrt(len(ncoll_anchor)))
            if len(ncoll_anchor) > 1 else 0.0,
    ]
    labels = [
        f"collapsed: {window}ep PRE\n(n={len(coll_pre)})",
        f"collapsed: {window}ep POST\n(n={len(coll_post)})",
        f"non-collapsed:\n±{window}ep around\nmedian collapse\n(n={len(ncoll_anchor)})",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=sems, capsize=8,
                  color=["#d35400", "#c0392b", "#2c3e50"])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean saturation fraction")
    ax.set_title("Saturation around collapse: collapsed vs non-collapsed seeds")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== ANALYSIS 1: SATURATION ===")
    print(f"Using device: {DEVICE}")
    print(f"Running {N_SEEDS} seeds...")

    histories: List[Dict] = []
    collapses: List[Optional[Dict]] = []
    for s in range(N_SEEDS):
        print(f"  seed {s} ... ", end="", flush=True)
        h = _train_with_saturation(_cfg(s))
        c = detect_collapse(h["cosine_sim"])
        histories.append(h)
        collapses.append(c)
        sat_final = float(np.mean(h["sat_per_layer"][-1]))
        print(f"final cos={np.mean(h['cosine_sim'][-50:]):.3f}  "
              f"final sat={sat_final:.3f}  "
              f"collapse={'ep ' + str(c['epoch']) if c else 'none'}")

    _plot_saturation_over_training(
        histories, collapses,
        os.path.join(FIG_DIR, "saturation_over_training.png"))
    coef = _plot_saturation_vs_cosine(
        histories, os.path.join(FIG_DIR, "saturation_vs_cosine.png"))
    _plot_collapsed_vs_not_bars(
        histories, collapses,
        os.path.join(FIG_DIR, "saturation_collapsed_vs_not.png"))

    print("\n=== Per-seed saturation summary ===")
    print(f"{'seed':<5}{'collapse_ep':<13}{'sat_pre':<11}{'sat_post':<11}"
          f"{'increased?':<12}{'corr(sat,cos)':<14}")
    print("-" * 66)
    for s, (h, c) in enumerate(zip(histories, collapses)):
        sats = np.asarray(h["sat_per_layer"]).mean(axis=1)
        cos = np.asarray(h["cosine_sim"])
        corr = float(np.corrcoef(sats, cos)[0, 1])
        if c is not None:
            ep = c["epoch"]
            pre = sats[max(0, ep - 50):ep].mean()
            post = sats[ep:ep + 50].mean()
            inc = "yes" if post > pre else "no"
            ep_s = str(ep)
        else:
            pre = sats[180:230].mean()
            post = sats[230:280].mean()
            inc = "n/a"
            ep_s = "none"
        print(f"{s:<5}{ep_s:<13}{pre:<11.4f}{post:<11.4f}{inc:<12}{corr:<14.3f}")

    print(f"\nGlobal Pearson r(sat, cos) across all (seed, epoch) = {coef:.3f}")


if __name__ == "__main__":
    main()
