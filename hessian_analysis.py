"""GPU Hessian analysis (non-deterministic baseline).

Mirrors ``hessian_analysis_cpu.py`` so the two can be compared directly.
Uses an isolated ``torch.Generator`` for probe vectors so the RNG stream
that drives model init/SGD is byte-identical to a clean (no-probe) run.
Any difference between probe-on and probe-off therefore reflects only
GPU non-determinism (atomic reductions, kernel scheduling, etc.), not
RNG-stream contamination.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_ind

from collapse_detection import detect_collapse
from hessian_utils import hutchinson_trace, lanczos_top_eigs
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
RESULTS_DIR = "results"
HESS_EVERY = 5
HUTCH_SAMPLES = 5
LANCZOS_ITERS = 30
LANCZOS_K = 5
RUN_TAG = "gpu"


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


def run_with_curvature(config: Dict, device: torch.device) -> Dict[str, List]:
    """Train with both Hutchinson trace and Lanczos top-k logged every
    ``HESS_EVERY`` epochs. Probe RNG is isolated from the model RNG."""
    torch.manual_seed(config["seed"])

    X, y = get_dataset(config["dataset"])
    X, y = X.to(device), y.to(device)

    true_net = MLPTrue(2, config["hidden_dim"], config["num_layers"],
                       config["beta_f"]).to(device)
    surr_net = MLPSurrogate(2, config["hidden_dim"], config["num_layers"],
                            config["beta_f"], config["beta_sg"]).to(device)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_surr = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])

    probe_gen = torch.Generator(device=device)
    probe_gen.manual_seed(10_000 + config["seed"])

    history: Dict[str, List] = {
        "epoch": [], "cosine_sim": [], "sign_agreement": [],
        "relative_magnitude": [], "true_loss": [], "grad_norm_true": [],
        "hess_epoch": [], "hutch_trace": [], "lambda_max": [], "lambda_top5": [],
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

        history["epoch"].append(epoch)
        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["sign_agreement"].append(sign_agreement(g_true, g_surr))
        history["relative_magnitude"].append(relative_magnitude(g_true, g_surr))
        history["true_loss"].append(float(loss_true.item()))
        history["grad_norm_true"].append(float(torch.norm(g_true).item()))

        if epoch % HESS_EVERY == 0:
            tr = hutchinson_trace(true_net, loss_fn, X, y,
                                  HUTCH_SAMPLES, generator=probe_gen)
            eigs = lanczos_top_eigs(true_net, loss_fn, X, y,
                                    n_iter=LANCZOS_ITERS, k=LANCZOS_K,
                                    generator=probe_gen)
            history["hess_epoch"].append(epoch)
            history["hutch_trace"].append(tr)
            history["lambda_max"].append(eigs[0])
            history["lambda_top5"].append(eigs)

        opt_true.step()

    return history


def _hess_window(history: Dict, key: str, lo: int, hi: int) -> np.ndarray:
    he = np.asarray(history["hess_epoch"])
    val = np.asarray(history[key])
    mask = (he >= lo) & (he <= hi)
    return val[mask]


def _plot_collapse_panel(histories, collapses, seeds, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for h, c, s, color in zip(histories, collapses, seeds, colors):
        axes[0].plot(h["epoch"], h["cosine_sim"], color=color,
                     label=f"seed {s}", linewidth=1.2)
        axes[1].plot(h["hess_epoch"], h["hutch_trace"], color=color,
                     marker="o", markersize=3, linewidth=1.2)
        axes[2].plot(h["hess_epoch"], h["lambda_max"], color=color,
                     marker="o", markersize=3, linewidth=1.2)
        axes[3].plot(h["epoch"], h["grad_norm_true"], color=color,
                     linewidth=1.2)
        if c is not None:
            for ax in axes:
                ax.axvline(c["epoch"], color=color, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("cos sim"); axes[0].set_title("Cosine similarity (SG vs true)")
    axes[0].legend(loc="lower left", fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("tr(H)"); axes[1].set_title("Hutchinson trace estimate")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel(r"$\lambda_{\max}(H)$")
    axes[2].set_title("Lanczos top eigenvalue")
    axes[2].grid(True, alpha=0.3)
    axes[3].set_ylabel("||g_true||"); axes[3].set_xlabel("Epoch")
    axes[3].set_title("True gradient L2 norm"); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_collapsed_vs_not(coll_hist, ncoll_hist, key, ylabel, title, out_path):
    eps = np.asarray(coll_hist[0]["hess_epoch"])
    coll = np.stack([h[key] for h in coll_hist])
    ncoll = np.stack([h[key] for h in ncoll_hist])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eps, coll.mean(0), color="crimson", linewidth=2,
            label=f"collapsed (n={len(coll_hist)})")
    if len(coll_hist) > 1:
        std = coll.std(0, ddof=1)
        ax.fill_between(eps, coll.mean(0) - std, coll.mean(0) + std,
                        color="crimson", alpha=0.2)
    ax.plot(eps, ncoll.mean(0), color="navy", linewidth=2,
            label=f"non-collapsed (n={len(ncoll_hist)})")
    if len(ncoll_hist) > 1:
        std = ncoll.std(0, ddof=1)
        ax.fill_between(eps, ncoll.mean(0) - std, ncoll.mean(0) + std,
                        color="navy", alpha=0.2)
    ax.axvspan(230, 260, color="gray", alpha=0.15, label="observed collapse window")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_main(device: torch.device, tag: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {device}  (tag={tag})")

    all_seeds = list(range(10))
    histories: Dict[int, Dict] = {}
    collapses: Dict[int, Optional[Dict]] = {}
    print(f"Running {len(all_seeds)} seeds  "
          f"(Hutchinson m={HUTCH_SAMPLES}, Lanczos n={LANCZOS_ITERS} k={LANCZOS_K}, "
          f"every {HESS_EVERY} epochs)...")
    for s in all_seeds:
        print(f"  seed {s} ... ", end="", flush=True)
        h = run_with_curvature(_base_4layer_cfg(s), device)
        c = detect_collapse(h["cosine_sim"])
        histories[s] = h
        collapses[s] = c
        print(f"final cos={np.mean(h['cosine_sim'][-50:]):.3f}  "
              f"collapse={'ep ' + str(c['epoch']) if c else 'none'}")

    collapsed = [s for s in all_seeds if collapses[s] is not None]
    non_collapsed = [s for s in all_seeds if collapses[s] is None]
    print(f"\nCollapsed: {collapsed}   Non-collapsed: {non_collapsed}")

    if collapsed:
        _plot_collapse_panel([histories[s] for s in collapsed],
                             [collapses[s] for s in collapsed], collapsed,
                             os.path.join(FIG_DIR, f"hessian_trace_collapse_{tag}.png"))
    if collapsed and non_collapsed:
        _plot_collapsed_vs_not(
            [histories[s] for s in collapsed],
            [histories[s] for s in non_collapsed],
            "hutch_trace", "tr(H)",
            "Hutchinson trace: collapsed vs non-collapsed",
            os.path.join(FIG_DIR, f"hessian_collapsed_vs_not_trace_{tag}.png"),
        )
        _plot_collapsed_vs_not(
            [histories[s] for s in collapsed],
            [histories[s] for s in non_collapsed],
            "lambda_max", r"$\lambda_{\max}(H)$",
            "Top eigenvalue: collapsed vs non-collapsed",
            os.path.join(FIG_DIR, f"hessian_collapsed_vs_not_lambdamax_{tag}.png"),
        )

    print(f"\n=== Per-seed curvature summary ({tag}) ===")
    print(f"{'seed':<5}{'collapse_ep':<13}"
          f"{'tr(H)[200-240]':<17}{'tr(H)[241-280]':<17}"
          f"{'λmax[200-240]':<17}{'λmax[241-280]':<17}"
          f"{'corr(tr,cos)':<14}{'corr(λmax,cos)':<16}")
    print("-" * 116)
    for s in all_seeds:
        h, c = histories[s], collapses[s]
        tr_pre = float(_hess_window(h, "hutch_trace", 200, 240).mean())
        tr_post = float(_hess_window(h, "hutch_trace", 241, 280).mean())
        lm_pre = float(_hess_window(h, "lambda_max", 200, 240).mean())
        lm_post = float(_hess_window(h, "lambda_max", 241, 280).mean())
        cos_at = np.asarray(h["cosine_sim"])[np.asarray(h["hess_epoch"])]
        corr_tr = float(np.corrcoef(h["hutch_trace"], cos_at)[0, 1])
        corr_lm = float(np.corrcoef(h["lambda_max"], cos_at)[0, 1])
        ep_str = str(c["epoch"]) if c else "none"
        print(f"{s:<5}{ep_str:<13}"
              f"{tr_pre:<17.3f}{tr_post:<17.3f}"
              f"{lm_pre:<17.3f}{lm_post:<17.3f}"
              f"{corr_tr:<14.3f}{corr_lm:<16.3f}")

    if collapsed:
        print(f"\n=== Welch t-test: post-collapse vs preceding 50 epochs ({tag}) ===")
        print(f"{'seed':<5}{'metric':<12}{'pre_n':<7}{'post_n':<7}"
              f"{'pre_mean':<12}{'post_mean':<12}{'t':<10}{'p':<10}")
        print("-" * 75)
        for s in collapsed:
            h, c = histories[s], collapses[s]
            ep = c["epoch"]
            for key in ("hutch_trace", "lambda_max"):
                pre = _hess_window(h, key, ep - 50, ep - 1)
                post = _hess_window(h, key, ep, ep + 25)
                if len(pre) < 2 or len(post) < 2:
                    continue
                t, p = ttest_ind(post, pre, equal_var=False)
                print(f"{s:<5}{key:<12}{len(pre):<7d}{len(post):<7d}"
                      f"{pre.mean():<12.4f}{post.mean():<12.4f}"
                      f"{t:<10.3f}{p:<10.4f}")

    out = {
        str(s): {
            "collapse_epoch": (collapses[s]["epoch"] if collapses[s] else None),
            "hess_epoch": list(map(int, histories[s]["hess_epoch"])),
            "hutch_trace": list(map(float, histories[s]["hutch_trace"])),
            "lambda_max": list(map(float, histories[s]["lambda_max"])),
            "cosine_sim_at_hess": [
                float(histories[s]["cosine_sim"][e])
                for e in histories[s]["hess_epoch"]
            ],
        }
        for s in all_seeds
    }
    json_path = os.path.join(RESULTS_DIR, f"hessian_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved per-seed scalars to {json_path}")
    return histories, collapses


def main():
    run_main(DEVICE, RUN_TAG)


if __name__ == "__main__":
    main()
