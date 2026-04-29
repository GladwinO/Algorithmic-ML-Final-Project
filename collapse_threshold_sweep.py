"""Threshold-sensitivity analysis for the cosine-similarity collapse detector.

For a fixed set of depth=4 moons runs, sweeps the ``threshold`` argument of
``detect_collapse`` and reports both how many seeds get flagged and how
time-localised the detected events are (std of detected epoch).

Produces ``figures/collapse_threshold_sensitivity.png`` with two stacked
subplots:
  - top:    number of seeds flagged vs threshold
  - bottom: std of detected collapse epoch vs threshold (low std = real
            time-localised event; high std = noise)

The "elbow" between the two regimes empirically justifies the choice of
threshold used in `collapse_detection.py`.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from collapse_detection import (
    _base_4layer_cfg,
    detect_collapse,
    run_experiment_with_norms,
)
from sg_experiment.device import DEVICE

FIG_DIR = "figures"
N_SEEDS = 10
THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
CHOSEN_THRESHOLD = 0.15


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Running {N_SEEDS} depth=4 moons seeds for threshold sweep...")

    runs = [run_experiment_with_norms(_base_4layer_cfg(s)) for s in range(N_SEEDS)]
    cos_histories = [h["cosine_sim"] for h in runs]

    n_caught, mean_ep, std_ep, mean_drop = [], [], [], []
    print(
        f"\n{'thresh':<8}{'n_caught':<10}{'mean_ep':<10}{'std_ep':<10}{'mean_drop':<10}"
    )
    print("-" * 48)
    for thr in THRESHOLDS:
        hits = [
            d
            for d in (detect_collapse(c, threshold=thr) for c in cos_histories)
            if d is not None
        ]
        n = len(hits)
        n_caught.append(n)
        if n:
            eps = [d["epoch"] for d in hits]
            drops = [d["drop"] for d in hits]
            mean_ep.append(float(np.mean(eps)))
            std_ep.append(float(np.std(eps, ddof=1)) if n > 1 else 0.0)
            mean_drop.append(float(np.mean(drops)))
            print(
                f"{thr:<8.2f}{n:<10d}{mean_ep[-1]:<10.1f}"
                f"{std_ep[-1]:<10.1f}{mean_drop[-1]:<10.3f}"
            )
        else:
            mean_ep.append(np.nan)
            std_ep.append(np.nan)
            mean_drop.append(np.nan)
            print(f"{thr:<8.2f}{0:<10d}{'-':<10}{'-':<10}{'-':<10}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(THRESHOLDS, n_caught, "o-", color="navy")
    axes[0].set_ylabel(f"Seeds flagged (out of {N_SEEDS})")
    axes[0].set_title("Threshold sensitivity of collapse detector (depth=4, moons)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        CHOSEN_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"chosen threshold = {CHOSEN_THRESHOLD}",
    )
    axes[0].legend()

    axes[1].plot(THRESHOLDS, std_ep, "o-", color="darkorange")
    axes[1].set_xlabel("Detection threshold (drop in cosine sim)")
    axes[1].set_ylabel("Std of detected epoch (across seeds)")
    axes[1].set_title("Low std = consistent event;  high std = noise")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(CHOSEN_THRESHOLD, color="red", linestyle="--", alpha=0.7)
    # Annotate the elbow.
    axes[1].axvspan(0.12, 0.15, color="gray", alpha=0.15, label="regime change")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "collapse_threshold_sensitivity.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
