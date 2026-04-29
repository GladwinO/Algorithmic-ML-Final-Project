"""Improvement #5: beta_sg sweep on the collapse experiment.

Tests the prediction implied by sec 5.2 of REPORT.md: if the late-training
collapse is the failure of the output-side layer's alignment, then varying
beta_sg should systematically shift the collapse rate and timing. Smaller
beta_sg (flatter surrogate, closer to the true gradient on average) should
push collapse later or eliminate it; larger beta_sg should push it earlier.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

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
RESULTS_DIR = "results"
N_SEEDS = 10
BETAS_SG = [2, 3, 5, 7, 10]


def _train(beta_sg: float, seed: int) -> List[float]:
    torch.manual_seed(seed)
    X, y = get_dataset("moons")
    true_net = MLPTrue(2, 64, 4, 50).to(DEVICE)
    surr_net = MLPSurrogate(2, 64, 4, 50, beta_sg).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())
    loss_fn = nn.BCEWithLogitsLoss()
    opt_t = torch.optim.SGD(true_net.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(surr_net.parameters(), lr=0.01)
    cos_hist: List[float] = []
    for _ in range(300):
        opt_t.zero_grad()
        loss_fn(true_net(X), y).backward()
        g_t = get_flat_gradients(true_net)
        surr_net.load_state_dict(true_net.state_dict())
        opt_s.zero_grad()
        loss_fn(surr_net(X), y).backward()
        g_s = get_flat_gradients(surr_net)
        cos_hist.append(cosine_similarity(g_t, g_s))
        opt_t.step()
    return cos_hist


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"beta_sg sweep over {BETAS_SG}, {N_SEEDS} seeds each.\n")

    summary: Dict[float, Dict] = {}
    for beta in BETAS_SG:
        eps, finals = [], []
        for s in range(N_SEEDS):
            cos = _train(beta, s)
            c = detect_collapse(cos)
            ep = c["epoch"] if c else None
            eps.append(ep)
            finals.append(float(np.mean(cos[-50:])))
            print(f"  beta_sg={beta:>2}  seed {s}: "
                  f"final cos={finals[-1]:.3f}  "
                  f"collapse={'ep ' + str(ep) if ep is not None else 'none'}")
        n_coll = sum(1 for e in eps if e is not None)
        ep_arr = np.array([e for e in eps if e is not None])
        summary[beta] = {
            "collapse_rate": n_coll / N_SEEDS,
            "n_collapsed": n_coll,
            "median_collapse_epoch": float(np.median(ep_arr)) if len(ep_arr) else None,
            "mean_final_cos": float(np.mean(finals)),
            "epochs": eps,
            "finals": finals,
        }
        print(f"  -> beta_sg={beta}: rate={n_coll}/{N_SEEDS}  "
              f"median ep={summary[beta]['median_collapse_epoch']}  "
              f"mean final cos={summary[beta]['mean_final_cos']:.3f}\n")

    out_json = os.path.join(RESULTS_DIR, "beta_sg_collapse_sweep.json")
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"Saved {out_json}")

    # Figure: rate vs beta_sg + median epoch vs beta_sg.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rates = [summary[b]["collapse_rate"] for b in BETAS_SG]
    axes[0].plot(BETAS_SG, rates, "o-", color="crimson", linewidth=2)
    axes[0].set_xlabel(r"$\beta_{SG}$")
    axes[0].set_ylabel("Collapse rate (out of 10)")
    axes[0].set_title("Collapse rate vs surrogate steepness")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    med = [summary[b]["median_collapse_epoch"] for b in BETAS_SG]
    med_x = [b for b, m in zip(BETAS_SG, med) if m is not None]
    med_y = [m for m in med if m is not None]
    axes[1].plot(med_x, med_y, "o-", color="navy", linewidth=2)
    axes[1].set_xlabel(r"$\beta_{SG}$")
    axes[1].set_ylabel("Median collapse epoch")
    axes[1].set_title("Collapse timing vs surrogate steepness")
    axes[1].grid(True, alpha=0.3)

    finals = [summary[b]["mean_final_cos"] for b in BETAS_SG]
    axes[2].plot(BETAS_SG, finals, "o-", color="forestgreen", linewidth=2)
    axes[2].set_xlabel(r"$\beta_{SG}$")
    axes[2].set_ylabel("Mean final cosine similarity")
    axes[2].set_title("Final alignment vs surrogate steepness")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "beta_sg_collapse_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
