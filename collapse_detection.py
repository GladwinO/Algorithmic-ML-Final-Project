"""Detect & characterise the late-training cosine-similarity collapse seen in
deep (esp. 4-layer) sigmoid MLPs trained with surrogate-gradient updates.

Reproduces the 4-layer moons run for 10 seeds, also tracks the true-gradient
L2 norm and a flatness proxy each epoch, finds the collapse epoch per seed,
and produces three diagnostic figures.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.metrics import (
    cosine_similarity,
    get_flat_gradients,
    relative_magnitude,
    sign_agreement,
)
from sg_experiment.models import MLPSurrogate, MLPTrue

FIG_DIR = "figures"


# ------------------------- core experiment -------------------------


def run_experiment_with_norms(config: Dict) -> Dict[str, List[float]]:
    """Same loop as `sg_experiment.experiment.run_experiment` but additionally
    records per-epoch true-gradient L2 norm and a flatness proxy."""
    torch.manual_seed(config.get("seed", 0))

    X, y = get_dataset(config["dataset"])

    true_net = MLPTrue(2, config["hidden_dim"], config["num_layers"], config["beta_f"]).to(DEVICE)
    surr_net = MLPSurrogate(
        2,
        config["hidden_dim"],
        config["num_layers"],
        config["beta_f"],
        config["beta_sg"],
    ).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_surr = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])

    history: Dict[str, List[float]] = {
        "cosine_sim": [],
        "sign_agreement": [],
        "relative_magnitude": [],
        "true_loss": [],
        "surr_loss": [],
        "grad_norm_true": [],
        "grad_norm_over_loss": [],
        "epoch": [],
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

        gnorm = float(torch.norm(g_true).item())
        loss_val = float(loss_true.item())

        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["sign_agreement"].append(sign_agreement(g_true, g_surr))
        history["relative_magnitude"].append(relative_magnitude(g_true, g_surr))
        history["true_loss"].append(loss_val)
        history["surr_loss"].append(float(loss_surr.item()))
        history["grad_norm_true"].append(gnorm)
        history["grad_norm_over_loss"].append(gnorm / (loss_val + 1e-8))
        history["epoch"].append(epoch)

        opt_true.step()

    return history


# ------------------------- collapse detection -------------------------


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) rolling mean: out[t] = mean(x[max(0, t-w+1) : t+1]).

    Avoids the boundary artifact of `np.convolve(..., mode='same')`, which at
    the trailing edge divides by ``window`` even though only half a window of
    real samples contributes. That artifact otherwise produces a fake
    "collapse" near the last epoch in every run.
    """
    n = len(x)
    if window <= 1 or n == 0:
        return x.copy()
    out = np.empty(n, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    for t in range(n):
        lo = max(0, t - window + 1)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def detect_collapse(
    cosine_sim_history: List[float],
    window: int = 20,
    threshold: float = 0.15,
    min_epoch: int = 50,
) -> Optional[Dict]:
    """Find the first epoch after ``min_epoch`` where the smoothed cosine sim
    drops by more than ``threshold`` versus the rolling mean of the previous
    ``window`` epochs *and* the drop persists for at least ``window`` epochs
    (so trailing-edge wiggles don't count).
    """
    arr = np.asarray(cosine_sim_history, dtype=float)
    smoothed = _rolling_mean(arr, window)
    n = len(arr)
    for t in range(max(min_epoch, window), n - window):
        prev_mean = float(smoothed[t - 1])  # trailing mean ending at t-1
        post_mean = float(arr[t:t + window].mean())  # actual values after t
        if prev_mean - post_mean > threshold:
            return {
                "epoch": int(t),
                "before": prev_mean,
                "after": post_mean,
                "drop": prev_mean - post_mean,
            }
    return None


# ------------------------- plots -------------------------


def _plot_runs(histories, collapses, out_path):
    epochs = np.asarray(histories[0]["epoch"])
    cos = np.stack([h["cosine_sim"] for h in histories])
    gnorm = np.stack([h["grad_norm_true"] for h in histories])
    loss = np.stack([h["true_loss"] for h in histories])

    detected = [c["epoch"] for c in collapses if c is not None]
    if detected:
        mean_ep = float(np.mean(detected))
        std_ep = float(np.std(detected, ddof=1)) if len(detected) > 1 else 0.0
    else:
        mean_ep, std_ep = float("nan"), float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    for arr, ax, title, ylabel in [
        (cos, axes[0], "Cosine similarity (SG vs true)", "cos sim"),
        (gnorm, axes[1], "True gradient L2 norm", "||g_true||"),
        (loss, axes[2], "Training loss (BCE)", "loss"),
    ]:
        for run in arr:
            ax.plot(epochs, run, color="lightgray", linewidth=0.8, alpha=0.7)
        ax.plot(epochs, arr.mean(axis=0), color="navy", linewidth=2, label="mean")
        if detected:
            ax.axvline(mean_ep, color="red", linestyle="--", alpha=0.8,
                       label=f"mean collapse epoch ≈ {mean_ep:.0f}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[0].text(
        0.02, 0.05,
        f"Mean collapse epoch: {mean_ep:.1f} ± {std_ep:.1f}  "
        f"({len(detected)}/{len(histories)} runs)",
        transform=axes[0].transAxes, fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_distribution(collapses, out_path):
    detected = [c["epoch"] for c in collapses if c is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    if detected:
        ax.hist(detected, bins=10, color="steelblue", edgecolor="black")
        m = float(np.mean(detected))
        s = float(np.std(detected, ddof=1)) if len(detected) > 1 else 0.0
        ax.axvline(m, color="red", linestyle="--", label=f"mean = {m:.1f}")
        ax.axvspan(m - s, m + s, color="red", alpha=0.15, label=f"±1 std ({s:.1f})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No collapses detected",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Detected collapse epoch")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of collapse epochs across seeds")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _pre_post_stats(histories, collapses, window=20):
    """For each run with a detected collapse, take 20-epoch windows immediately
    before/after the collapse epoch and average the three quantities."""
    rows = []  # list of (cos_pre, cos_post, gn_pre, gn_post, loss_pre, loss_post)
    for h, c in zip(histories, collapses):
        if c is None:
            continue
        ep = c["epoch"]
        cos = np.asarray(h["cosine_sim"])
        gn = np.asarray(h["grad_norm_true"])
        ls = np.asarray(h["true_loss"])
        n = len(cos)
        pre_lo, pre_hi = max(0, ep - window), ep
        post_lo, post_hi = ep, min(n, ep + window)
        rows.append((
            cos[pre_lo:pre_hi].mean(), cos[post_lo:post_hi].mean(),
            gn[pre_lo:pre_hi].mean(), gn[post_lo:post_hi].mean(),
            ls[pre_lo:pre_hi].mean(), ls[post_lo:post_hi].mean(),
        ))
    return np.asarray(rows) if rows else None


def _plot_pre_post(stats, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    if stats is None:
        for ax in axes:
            ax.text(0.5, 0.5, "No collapses detected",
                    ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    titles = ["Cosine similarity", "True gradient norm", "Training loss"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        pre = stats[:, 2 * k]
        post = stats[:, 2 * k + 1]
        means = [pre.mean(), post.mean()]
        errs = [
            pre.std(ddof=1) if len(pre) > 1 else 0.0,
            post.std(ddof=1) if len(post) > 1 else 0.0,
        ]
        ax.bar(["pre-collapse", "post-collapse"], means, yerr=errs,
               color=["seagreen", "indianred"], capsize=8, edgecolor="black")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------- main -------------------------


def _base_4layer_cfg(seed: int) -> Dict:
    return {
        "hidden_dim": 64,
        "num_layers": 4,
        "beta_f": 50,
        "beta_sg": 5,
        "dataset": "moons",
        "n_epochs": 300,
        "lr": 0.01,
        "seed": seed,
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    n_seeds = 10
    print(f"\nRunning depth=4 moons sweep across {n_seeds} seeds...")
    histories, collapses = [], []
    for s in range(n_seeds):
        h = run_experiment_with_norms(_base_4layer_cfg(s))
        c = detect_collapse(h["cosine_sim"])
        histories.append(h)
        collapses.append(c)

    # Figures.
    _plot_runs(histories, collapses,
               os.path.join(FIG_DIR, "collapse_detection_4layer.png"))
    _plot_distribution(collapses,
                       os.path.join(FIG_DIR, "collapse_epoch_distribution.png"))
    pre_post = _pre_post_stats(histories, collapses, window=20)
    _plot_pre_post(pre_post,
                   os.path.join(FIG_DIR, "pre_post_collapse_comparison.png"))

    # Per-seed summary table.
    print("\n=== Per-seed collapse summary (depth=4, moons) ===")
    print(f"{'seed':<5}{'collapse_epoch':<16}{'cos_pre':<10}"
          f"{'cos_post':<10}{'gn_pre':<10}{'gn_post':<10}{'gn change':<10}")
    print("-" * 71)
    for s, h, c in zip(range(n_seeds), histories, collapses):
        if c is None:
            print(f"{s:<5}{'none':<16}{'-':<10}{'-':<10}{'-':<10}{'-':<10}{'-':<10}")
            continue
        ep = c["epoch"]
        gn = np.asarray(h["grad_norm_true"])
        cos = np.asarray(h["cosine_sim"])
        gn_pre = float(gn[max(0, ep - 20):ep].mean())
        gn_post = float(gn[ep:min(len(gn), ep + 20)].mean())
        cos_pre = float(cos[max(0, ep - 20):ep].mean())
        cos_post = float(cos[ep:min(len(cos), ep + 20)].mean())
        change = "↑" if gn_post > gn_pre else "↓"
        print(f"{s:<5}{ep:<16d}{cos_pre:<10.3f}{cos_post:<10.3f}"
              f"{gn_pre:<10.4f}{gn_post:<10.4f}{change:<10}")

    detected = [c for c in collapses if c is not None]
    if detected:
        eps = [c["epoch"] for c in detected]
        print(
            f"\nCollapse detected in {len(detected)}/{n_seeds} runs.  "
            f"epoch = {np.mean(eps):.1f} ± {np.std(eps, ddof=1):.1f}"
        )
    else:
        print("\nNo collapse detected in any seed.")

    # Same detector, depths 1-3, single seed each.
    print("\n=== Collapse check at depths 1-3 (single seed=0) ===")
    for d in [1, 2, 3]:
        cfg = _base_4layer_cfg(0)
        cfg["num_layers"] = d
        h = run_experiment_with_norms(cfg)
        c = detect_collapse(h["cosine_sim"])
        if c is None:
            print(f"  depth={d}: no collapse detected")
        else:
            print(
                f"  depth={d}: collapse at epoch {c['epoch']}  "
                f"({c['before']:.3f} -> {c['after']:.3f}, drop {c['drop']:.3f})"
            )


if __name__ == "__main__":
    main()
