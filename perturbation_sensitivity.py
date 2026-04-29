"""Perturbation sensitivity of the SG collapse phenomenon.

For ``N`` seeds, train the depth-4 moons network twice with the *same*
``torch.manual_seed`` initialisation:

  Arm A — control:  no curvature probe.
  Arm B — probed:   Hutchinson trace + Lanczos top-k every HESS_EVERY epochs,
                    using an isolated probe RNG generator so the model RNG
                    stream is byte-identical to Arm A.

If the only difference between A and B is the floating-point drift induced
by the additional autograd graph operations (and, on GPU, by atomic-op
non-determinism), the McNemar test on the paired binary outcome
(collapse / no-collapse) tells us whether instrumentation reliably
suppresses (or induces) the collapse.

Reports:
  - paired 2x2 contingency table
  - exact McNemar p-value (binomial test on discordant pairs)
  - Fisher exact two-sided p-value on the unpaired contingency
  - Cohen's h effect size between the two collapse rates
  - Wilson-score 95% CIs on each rate and the rate difference
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest, fisher_exact, norm

from collapse_detection import detect_collapse
from hessian_utils import hutchinson_trace, lanczos_top_eigs
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
RESULTS_DIR = "results"
N_SEEDS = 30
HESS_EVERY = 5
HUTCH_SAMPLES = 5
LANCZOS_ITERS = 30
LANCZOS_K = 5


def _cfg(seed: int) -> Dict:
    return {
        "hidden_dim": 64, "num_layers": 4, "beta_f": 50, "beta_sg": 5,
        "dataset": "moons", "n_epochs": 300, "lr": 0.01, "seed": seed,
    }


def _train_one(config: Dict, device: torch.device,
               probe: bool) -> Dict[str, List]:
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
        "epoch": [], "cosine_sim": [], "true_loss": [], "grad_norm_true": [],
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
        history["true_loss"].append(float(loss_true.item()))
        history["grad_norm_true"].append(float(torch.norm(g_true).item()))

        if probe and epoch % HESS_EVERY == 0:
            hutchinson_trace(true_net, loss_fn, X, y,
                             HUTCH_SAMPLES, generator=probe_gen)
            lanczos_top_eigs(true_net, loss_fn, X, y,
                             n_iter=LANCZOS_ITERS, k=LANCZOS_K,
                             generator=probe_gen)

        opt_true.step()

    return history


# ------------------------- statistics -------------------------


def _wilson_ci(k: int, n: int, conf: float = 0.95):
    """Wilson-score CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    def phi(p):
        return 2 * np.arcsin(np.sqrt(np.clip(p, 0, 1)))
    return float(phi(p1) - phi(p2))


def _mcnemar_exact(b: int, c: int):
    """Exact McNemar test on discordant pairs (b, c) using a binomial test
    with p=0.5. Returns (statistic = min(b,c), p_two_sided)."""
    n = b + c
    if n == 0:
        return 0, 1.0
    res = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided")
    return min(b, c), float(res.pvalue)


def _diff_ci(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95):
    """Newcombe (Wilson) interval for difference of two proportions."""
    z = norm.ppf(1 - (1 - conf) / 2)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson_ci(k1, n1, conf)
    l2, u2 = _wilson_ci(k2, n2, conf)
    delta = p1 - p2
    lower = delta - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = delta + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return delta, (lower, upper)


# ------------------------- main -------------------------


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"N_SEEDS={N_SEEDS}  Hutchinson m={HUTCH_SAMPLES}  "
          f"Lanczos n={LANCZOS_ITERS} k={LANCZOS_K}  every {HESS_EVERY} epochs\n")

    rows = []  # list of dicts per seed
    for s in range(N_SEEDS):
        ha = _train_one(_cfg(s), DEVICE, probe=False)
        ca = detect_collapse(ha["cosine_sim"])
        hb = _train_one(_cfg(s), DEVICE, probe=True)
        cb = detect_collapse(hb["cosine_sim"])
        rows.append({
            "seed": s,
            "control_collapse": ca is not None,
            "control_epoch": ca["epoch"] if ca else None,
            "probed_collapse": cb is not None,
            "probed_epoch": cb["epoch"] if cb else None,
        })
        print(f"  seed {s:>2d}: control={'C' if ca else '-'}"
              f"({ca['epoch'] if ca else '-':>3}) "
              f"probed={'C' if cb else '-'}"
              f"({cb['epoch'] if cb else '-':>3})")

    n = len(rows)
    cc = sum(1 for r in rows if r["control_collapse"] and r["probed_collapse"])
    cn = sum(1 for r in rows if r["control_collapse"] and not r["probed_collapse"])
    nc = sum(1 for r in rows if not r["control_collapse"] and r["probed_collapse"])
    nn = sum(1 for r in rows if not r["control_collapse"] and not r["probed_collapse"])

    p_ctrl = (cc + cn) / n
    p_prob = (cc + nc) / n

    print("\n=== Paired 2x2 contingency table ===")
    print(f"{'':<22}{'probed: C':<12}{'probed: -':<12}{'row total':<10}")
    print(f"{'control: C':<22}{cc:<12d}{cn:<12d}{cc + cn:<10d}")
    print(f"{'control: -':<22}{nc:<12d}{nn:<12d}{nc + nn:<10d}")
    print(f"{'col total':<22}{cc + nc:<12d}{cn + nn:<12d}{n:<10d}")

    print("\n=== Marginal collapse rates ===")
    cl, cu = _wilson_ci(cc + cn, n)
    pl, pu = _wilson_ci(cc + nc, n)
    print(f"  control rate:  {p_ctrl:.3f}   "
          f"95% Wilson CI [{cl:.3f}, {cu:.3f}]   "
          f"({cc + cn}/{n})")
    print(f"  probed  rate:  {p_prob:.3f}   "
          f"95% Wilson CI [{pl:.3f}, {pu:.3f}]   "
          f"({cc + nc}/{n})")

    delta, (dl, du) = _diff_ci(cc + cn, n, cc + nc, n)
    print(f"  rate diff (control - probed): {delta:+.3f}   "
          f"95% Newcombe CI [{dl:+.3f}, {du:+.3f}]")

    h = _cohens_h(p_ctrl, p_prob)
    print(f"  Cohen's h:     {h:+.3f}   "
          f"(|h|<0.2 small, 0.2-0.5 medium, 0.5-0.8 large)")

    print("\n=== Statistical tests ===")
    print(f"  Discordant pairs:  control-only = {cn}, probed-only = {nc}")
    stat, p_mcnemar = _mcnemar_exact(cn, nc)
    print(f"  McNemar (exact, paired):       p = {p_mcnemar:.4f}   "
          f"(binomial test on {cn + nc} discordant pairs)")
    odds, p_fisher = fisher_exact(
        [[cc + cn, nc + nn - 0], [nc, nn]],  # placeholder; correct table below
        alternative="two-sided",
    )
    # The right unpaired table is collapsed-or-not vs arm membership:
    # arm: control                 collapsed = cc+cn,  not = nc+nn? wrong.
    # Each seed contributes one row to each arm in unpaired view:
    #   control arm:  collapsed = cc+cn,  not = nc+nn
    #   probed  arm:  collapsed = cc+nc,  not = cn+nn
    table = [[cc + cn, nc + nn], [cc + nc, cn + nn]]
    odds, p_fisher = fisher_exact(table, alternative="two-sided")
    print(f"  Fisher exact (unpaired 2x2):   p = {p_fisher:.4f}   "
          f"odds ratio = {odds:.3f}")

    # ---------- figure: contingency heatmap + per-seed strip ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    M = np.array([[cc, cn], [nc, nn]])
    im = axes[0].imshow(M, cmap="Blues", aspect="auto")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["probed: collapse", "probed: none"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["control: collapse", "control: none"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(M[i, j]),
                         ha="center", va="center", fontsize=18,
                         color=("white" if M[i, j] > M.max() / 2 else "black"))
    axes[0].set_title(
        f"Paired contingency (N={n})\n"
        f"control rate = {p_ctrl:.2f}   probed rate = {p_prob:.2f}\n"
        f"McNemar p = {p_mcnemar:.4f}   Cohen's h = {h:+.3f}"
    )
    plt.colorbar(im, ax=axes[0])

    # Per-seed strip showing concordance / discordance.
    seeds = [r["seed"] for r in rows]
    ctrl_y = [1 if r["control_collapse"] else 0 for r in rows]
    prob_y = [1 if r["probed_collapse"] else 0 for r in rows]
    axes[1].scatter(seeds, ctrl_y, color="navy", s=60, label="control",
                    marker="o")
    axes[1].scatter(seeds, [v + 0.05 for v in prob_y], color="crimson",
                    s=60, label="probed", marker="x", linewidths=2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["no collapse", "collapse"])
    axes[1].set_xlabel("Seed")
    axes[1].set_title("Per-seed outcome (offset for visibility)")
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].legend(loc="center right")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "perturbation_sensitivity.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    out_json = os.path.join(RESULTS_DIR, "perturbation_sensitivity.json")
    with open(out_json, "w") as f:
        json.dump({
            "n": n,
            "table": {"cc": cc, "cn": cn, "nc": nc, "nn": nn},
            "p_control": p_ctrl,
            "p_probed": p_prob,
            "wilson_ci_control": list(_wilson_ci(cc + cn, n)),
            "wilson_ci_probed": list(_wilson_ci(cc + nc, n)),
            "diff": delta,
            "newcombe_ci": list((dl, du)),
            "cohens_h": h,
            "mcnemar_p": p_mcnemar,
            "fisher_p": p_fisher,
            "fisher_odds_ratio": odds,
            "rows": rows,
        }, f, indent=2)
    print(f"Saved per-seed table to {out_json}")


if __name__ == "__main__":
    main()
