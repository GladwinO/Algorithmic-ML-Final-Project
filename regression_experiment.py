"""Regression experiment: depth vs target-ruggedness interaction sweep.

Extends the existing classification experiments with a 1D regression task
where the target is a sum of sine waves whose count controls landscape
ruggedness. Produces a 4x4 (depth x n_frequencies) heatmap of SG-vs-true
gradient agreement.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sg_experiment.device import DEVICE
from sg_experiment.metrics import (
    cosine_similarity,
    get_flat_gradients,
    relative_magnitude,
    sign_agreement,
)
from sg_experiment.models import MLPSurrogate, MLPTrue
from sg_experiment.plots import plot_metrics_over_training

FIG_DIR = "figures"


def get_rugged_regression_dataset(
    n_samples: int = 1000,
    n_frequencies: int = 1,
    x_range=(-2, 2),
    seed: int = 42,
):
    """1D regression with target = sum of n_frequencies random sines.

    Higher ``n_frequencies`` → more rugged target → more nonconvex loss
    landscape. Output is normalised by ``n_frequencies`` so loss scales stay
    comparable across configurations.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 1))

    frequencies = rng.uniform(2, 15, size=n_frequencies)
    phases = rng.uniform(0, 2 * np.pi, size=n_frequencies)
    amplitudes = rng.uniform(0.5, 1.5, size=n_frequencies)

    y = np.zeros_like(X)
    for f, p, a in zip(frequencies, phases, amplitudes):
        y += a * np.sin(f * X + p)
    y = y / n_frequencies

    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    return X, y


def run_experiment_regression(config: dict) -> dict:
    """Per-epoch true vs surrogate gradient comparison on the rugged
    regression task. Mirrors ``sg_experiment.experiment.run_experiment``."""
    torch.manual_seed(config.get("seed", 0))

    X, y = get_rugged_regression_dataset(
        n_samples=1000,
        n_frequencies=config["n_frequencies"],
        seed=config.get("dataset_seed", 42),
    )

    true_net = MLPTrue(
        input_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        beta_f=config["beta_f"],
    ).to(DEVICE)
    surr_net = MLPSurrogate(
        input_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        beta_f=config["beta_f"],
        beta_sg=config["beta_sg"],
    ).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.MSELoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=config["lr"])

    history = {
        "cosine_sim": [],
        "sign_agreement": [],
        "relative_magnitude": [],
        "true_loss": [],
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
        for p in surr_net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        pred_surr = surr_net(X)
        loss_surr = loss_fn(pred_surr, y)
        loss_surr.backward()
        g_surr = get_flat_gradients(surr_net)

        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["sign_agreement"].append(sign_agreement(g_true, g_surr))
        history["relative_magnitude"].append(relative_magnitude(g_true, g_surr))
        history["true_loss"].append(loss_true.item())
        history["epoch"].append(epoch)

        opt_true.step()

    return history


# --- Experiment 5: ruggedness sweep at fixed depth ---
ruggedness_configs = [
    {
        "hidden_dim": 64,
        "num_layers": 3,
        "beta_f": 50,
        "beta_sg": 5,
        "n_frequencies": nf,
        "n_epochs": 500,
        "lr": 0.01,
        "seed": 0,
    }
    for nf in [1, 3, 5, 10]
]
ruggedness_labels = ["1 freq (smooth)", "3 freqs", "5 freqs", "10 freqs (rugged)"]

# --- Experiment 6: 2D depth x ruggedness sweep ---
DEPTHS = [1, 2, 3, 4]
FREQS = [1, 3, 5, 10]
interaction_configs = [
    {
        "hidden_dim": 64,
        "num_layers": d,
        "beta_f": 50,
        "beta_sg": 5,
        "n_frequencies": nf,
        "n_epochs": 500,
        "lr": 0.01,
        "seed": 0,
        "_label": f"depth={d}, freqs={nf}",
    }
    for d in DEPTHS
    for nf in FREQS
]


def _build_heatmap(histories):
    cos = np.zeros((len(DEPTHS), len(FREQS)))
    sign = np.zeros((len(DEPTHS), len(FREQS)))
    idx = 0
    for i, _ in enumerate(DEPTHS):
        for j, _ in enumerate(FREQS):
            h = histories[idx]
            cos[i, j] = float(np.mean(h["cosine_sim"][-50:]))
            sign[i, j] = float(np.mean(h["sign_agreement"][-50:]))
            idx += 1
    return cos, sign


def _plot_heatmaps(cos, sign, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cos, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(FREQS)))
    axes[0].set_xticklabels([f"{f} freqs" for f in FREQS])
    axes[0].set_yticks(range(len(DEPTHS)))
    axes[0].set_yticklabels([f"{d} layers" for d in DEPTHS])
    axes[0].set_title("Cosine Similarity: Depth x Ruggedness")
    axes[0].set_xlabel("Target Ruggedness")
    axes[0].set_ylabel("Network Depth")
    for i in range(len(DEPTHS)):
        for j in range(len(FREQS)):
            axes[0].text(
                j, i, f"{cos[i,j]:.2f}", ha="center", va="center", fontsize=11
            )
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sign, cmap="RdYlGn", vmin=0.5, vmax=1, aspect="auto")
    axes[1].set_xticks(range(len(FREQS)))
    axes[1].set_xticklabels([f"{f} freqs" for f in FREQS])
    axes[1].set_yticks(range(len(DEPTHS)))
    axes[1].set_yticklabels([f"{d} layers" for d in DEPTHS])
    axes[1].set_title("Sign Agreement: Depth x Ruggedness")
    axes[1].set_xlabel("Target Ruggedness")
    axes[1].set_ylabel("Network Depth")
    for i in range(len(DEPTHS)):
        for j in range(len(FREQS)):
            axes[1].text(
                j, i, f"{sign[i,j]:.2f}", ha="center", va="center", fontsize=11
            )
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _summarise(cos, sign):
    flat_cos = cos.flatten()
    i_max, i_min = int(np.argmax(flat_cos)), int(np.argmin(flat_cos))
    di_max, fi_max = divmod(i_max, len(FREQS))
    di_min, fi_min = divmod(i_min, len(FREQS))

    print("\n=== Heatmap summary ===")
    print(
        f"Best cosine sim:  depth={DEPTHS[di_max]}, "
        f"freqs={FREQS[fi_max]}  -> {cos[di_max, fi_max]:.3f}"
    )
    print(
        f"Worst cosine sim: depth={DEPTHS[di_min]}, "
        f"freqs={FREQS[fi_min]}  -> {cos[di_min, fi_min]:.3f}"
    )

    # Crude additive prediction: predict bottom-right from top-right + bottom-left
    # relative to top-left baseline.
    top_left = cos[0, 0]
    top_right = cos[0, -1]
    bot_left = cos[-1, 0]
    bot_right = cos[-1, -1]
    additive_pred = top_right + bot_left - top_left  # add the two deltas to baseline
    interaction_gap = additive_pred - bot_right
    print(
        f"\nDepth-only effect (top-left -> bot-left): {top_left:.3f} -> {bot_left:.3f}"
        f"  (delta {bot_left - top_left:+.3f})"
    )
    print(
        f"Ruggedness-only effect (top-left -> top-right): "
        f"{top_left:.3f} -> {top_right:.3f}"
        f"  (delta {top_right - top_left:+.3f})"
    )
    print(
        f"Observed bot-right: {bot_right:.3f}  vs additive prediction: "
        f"{additive_pred:.3f}  (interaction gap {interaction_gap:+.3f})"
    )

    if abs(interaction_gap) < 0.05:
        verdict = (
            "INDEPENDENT EFFECTS — depth and ruggedness each degrade SG fidelity, "
            "but they do not compound."
        )
    elif interaction_gap > 0.05:
        verdict = (
            "INTERACTION EFFECT — deep + rugged is meaningfully worse than the sum "
            "of the two effects alone."
        )
    else:
        verdict = (
            "SUB-ADDITIVE / depth or ruggedness saturates the degradation; the "
            "combined corner is no worse than either alone."
        )

    # Compare row-variance vs column-variance to detect "depth dominates"
    row_range = float(cos.max(axis=1).mean() - cos.min(axis=1).mean())
    col_range = float(cos.max(axis=0).mean() - cos.min(axis=0).mean())
    print(
        f"\nMean within-row range (ruggedness effect at fixed depth):  {row_range:.3f}"
    )
    print(
        f"Mean within-col range (depth effect at fixed ruggedness):    {col_range:.3f}"
    )
    if col_range > 2 * row_range:
        verdict += "\nFurther: DEPTH DOMINATES — ruggedness is a much smaller axis."
    elif row_range > 2 * col_range:
        verdict += "\nFurther: RUGGEDNESS DOMINATES — depth is a much smaller axis."

    print(f"\nVerdict: {verdict}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    n_seeds_rugg = 5
    print(f"=== Experiment 5: ruggedness sweep at depth=3 ({n_seeds_rugg} seeds) ===")
    rugg_hist = []
    for cfg, label in zip(ruggedness_configs, ruggedness_labels):
        per_seed = []
        for s in range(n_seeds_rugg):
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            cfg_s["dataset_seed"] = 42 + s
            per_seed.append(run_experiment_regression(cfg_s))
        agg = {
            k: list(np.mean([h[k] for h in per_seed], axis=0))
            for k in per_seed[0]
        }
        rugg_hist.append(agg)
        print(
            f"  {label:<22s} final cos={np.mean(agg['cosine_sim'][-50:]):.3f}  "
            f"sign={np.mean(agg['sign_agreement'][-50:]):.3f}  "
            f"loss={agg['true_loss'][-1]:.4f}"
        )

    plot_metrics_over_training(
        rugg_hist,
        ruggedness_labels,
        metric="cosine_sim",
        title="Effect of Target Ruggedness on SG Alignment",
        out_dir=FIG_DIR,
    )
    plot_metrics_over_training(
        rugg_hist,
        ruggedness_labels,
        metric="sign_agreement",
        title="Effect of Target Ruggedness on Sign Agreement",
        out_dir=FIG_DIR,
    )

    n_seeds = 5
    print(
        f"\n=== Experiment 6: 2D depth x ruggedness sweep "
        f"({len(interaction_configs)} cells x {n_seeds} seeds) ==="
    )
    inter_hist = []
    for cfg in interaction_configs:
        per_seed = []
        for s in range(n_seeds):
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            cfg_s["dataset_seed"] = 42 + s
            per_seed.append(run_experiment_regression(cfg_s))
        # Aggregate: average each metric series across seeds element-wise
        agg = {
            k: list(np.mean([h[k] for h in per_seed], axis=0))
            for k in per_seed[0]
        }
        inter_hist.append(agg)
        print(
            f"  {cfg['_label']:<20s}  "
            f"cos={np.mean(agg['cosine_sim'][-50:]):.3f}  "
            f"sign={np.mean(agg['sign_agreement'][-50:]):.3f}  "
            f"(over {n_seeds} seeds)"
        )

    cos, sign = _build_heatmap(inter_hist)
    out_path = os.path.join(FIG_DIR, "heatmap_depth_vs_ruggedness.png")
    _plot_heatmaps(cos, sign, out_path)
    print(f"\nSaved heatmap to {out_path}")

    _summarise(cos, sign)


if __name__ == "__main__":
    main()
