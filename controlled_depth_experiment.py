"""Controlled-depth experiment.

Repeats the depth sweep but holds the total trainable parameter count
approximately constant across depths by adjusting hidden width. This isolates
the effect of backprop chain length from the effect of network capacity, which
were confounded in the original depth experiment.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from sg_experiment.device import DEVICE
from sg_experiment.experiment import run_experiment
from sg_experiment.models import MLPTrue
from sg_experiment.plots import plot_metrics_over_training

FIG_DIR = "figures"
RESULTS_DIR = "results"
UNCONTROLLED_PATH = os.path.join(RESULTS_DIR, "all_histories.json")


# ---------- helpers ----------


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _params_for(num_layers: int, h: int, input_dim: int, output_dim: int) -> int:
    """Closed-form param count for a sigmoid MLP with `num_layers` hidden layers
    of width `h`, mirroring `MLPTrue` in `sg_experiment.models`."""
    p = input_dim * h + h  # first layer weights + bias
    p += (num_layers - 1) * (h * h + h)  # remaining hidden layers
    p += h * output_dim + output_dim  # output layer
    return p


def get_controlled_width(
    num_layers: int,
    target_params: int,
    input_dim: int,
    output_dim: int,
    h_min: int = 8,
    h_max: int = 4000,
) -> int:
    """Brute-force search over integer widths in [h_min, h_max] for the width
    whose param count is closest to `target_params`."""
    best_h, best_diff = h_min, float("inf")
    for h in range(h_min, h_max + 1):
        diff = abs(_params_for(num_layers, h, input_dim, output_dim) - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


# ---------- build configs ----------


def build_controlled_configs():
    # Reference: same as uncontrolled depth experiment (depth=4, width=64, moons).
    ref_net = MLPTrue(input_dim=2, hidden_dim=64, num_layers=4, beta_f=50)
    target_params = count_parameters(ref_net)
    print(f"Reference (depth=4, width=64) param count: {target_params}")

    configs, labels = [], []
    for d in [1, 2, 3, 4]:
        w = get_controlled_width(d, target_params, input_dim=2, output_dim=1)
        # Build the network just to verify the actual count.
        actual_params = count_parameters(MLPTrue(2, w, d, beta_f=50))
        print(
            f"  depth={d}: width={w:>4d}  -> {actual_params} params "
            f"(target {target_params}, diff {actual_params - target_params:+d})"
        )
        cfg = {
            "hidden_dim": w,
            "num_layers": d,
            "beta_f": 50,
            "beta_sg": 5,
            "dataset": "moons",
            "n_epochs": 300,
            "lr": 0.01,
            "seed": 0,
            "_label": f"depth={d}, width={w}, params={actual_params}",
        }
        configs.append(cfg)
        labels.append(cfg["_label"])
    return configs, labels, target_params


# ---------- comparison figure ----------


def _final_cos(history, window=50):
    return float(np.mean(history["cosine_sim"][-window:]))


def _plot_comparison(unc_histories, unc_labels, ctrl_histories, ctrl_labels,
                     target_params: int, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for h, label in zip(unc_histories, unc_labels):
        axes[0].plot(h["epoch"], h["cosine_sim"], label=label)
    axes[0].set_title("Uncontrolled Depth (fixed width=64)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cosine similarity (SG vs true gradient)")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for h, label in zip(ctrl_histories, ctrl_labels):
        axes[1].plot(h["epoch"], h["cosine_sim"], label=label)
    axes[1].set_title(f"Controlled Depth (fixed param count ≈ {target_params})")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- main ----------


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # 1. Build & report controlled configs.
    configs, labels, target_params = build_controlled_configs()

    # 2. Run controlled experiment (single seed to mirror original depth sweep).
    print("\nRunning controlled-depth sweep...")
    histories = []
    for cfg, label in zip(configs, labels):
        print(f"  {label} ... ", end="", flush=True)
        h = run_experiment(cfg)
        histories.append(h)
        print(f"final cos={_final_cos(h):.3f}  loss={h['true_loss'][-1]:.4f}")

    # 3. Standalone plots.
    plot_metrics_over_training(
        histories, labels,
        metric="cosine_sim",
        title="Controlled Depth SG Alignment",
        out_dir=FIG_DIR,
    )
    plot_metrics_over_training(
        histories, labels,
        metric="sign_agreement",
        title="Controlled Depth Sign Agreement",
        out_dir=FIG_DIR,
    )

    # 4. Comparison figure: load original uncontrolled depth histories.
    if not os.path.exists(UNCONTROLLED_PATH):
        print(
            f"\n[skip comparison] {UNCONTROLLED_PATH} not found; "
            f"run `python main.py` first to populate it."
        )
        return

    with open(UNCONTROLLED_PATH) as f:
        all_results = json.load(f)
    unc = all_results["depth"]
    unc_histories = unc["histories"]
    unc_labels = unc["labels"]

    cmp_path = os.path.join(FIG_DIR, "depth_controlled_vs_uncontrolled_comparison.png")
    _plot_comparison(unc_histories, unc_labels, histories, labels,
                     target_params, cmp_path)
    print(f"\nSaved comparison figure to {cmp_path}")

    # 5. Summary table.
    print("\n=== Final cosine similarity (mean of last 50 epochs) ===")
    print(f"{'Depth':<6}| {'Uncontrolled cos_sim':<22}| "
          f"{'Controlled cos_sim':<20}| Width (controlled)")
    print("-" * 75)
    for i, d in enumerate([1, 2, 3, 4]):
        unc_cos = _final_cos(unc_histories[i])
        ctrl_cos = _final_cos(histories[i])
        width = configs[i]["hidden_dim"]
        print(f"{d:<6}| {unc_cos:<22.3f}| {ctrl_cos:<20.3f}| {width}")


if __name__ == "__main__":
    main()
