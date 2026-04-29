"""Improvement #6: bump the perturbation-sensitivity test to N=60.

Same pairing/test machinery as ``perturbation_sensitivity.py`` but
N_SEEDS=60 to address the under-powered McNemar (p=0.0625 at N=30).
Saves to results/perturbation_sensitivity_n60.json and
figures/perturbation_sensitivity_n60.png so the N=30 result is preserved.
"""
from __future__ import annotations

import os

import perturbation_sensitivity as base

base.N_SEEDS = 60
# redirect outputs
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, fisher_exact, norm

from collapse_detection import detect_collapse


def main():
    os.makedirs(base.FIG_DIR, exist_ok=True)
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    print(f"Using device: {base.DEVICE}")
    print(f"N_SEEDS={base.N_SEEDS}  Hutchinson m={base.HUTCH_SAMPLES}  "
          f"Lanczos n={base.LANCZOS_ITERS} k={base.LANCZOS_K}  "
          f"every {base.HESS_EVERY} epochs\n")

    rows = []
    for s in range(base.N_SEEDS):
        ha = base._train_one(base._cfg(s), base.DEVICE, probe=False)
        ca = detect_collapse(ha["cosine_sim"])
        hb = base._train_one(base._cfg(s), base.DEVICE, probe=True)
        cb = detect_collapse(hb["cosine_sim"])
        rows.append({
            "seed": s,
            "control_collapse": ca is not None,
            "control_epoch": ca["epoch"] if ca else None,
            "probed_collapse": cb is not None,
            "probed_epoch": cb["epoch"] if cb else None,
        })
        print(f"  seed {s:>2d}: "
              f"control={'C' if ca else '-'}({ca['epoch'] if ca else '-':>3}) "
              f"probed={'C' if cb else '-'}({cb['epoch'] if cb else '-':>3})")

    n = len(rows)
    cc = sum(1 for r in rows if r["control_collapse"] and r["probed_collapse"])
    cn = sum(1 for r in rows if r["control_collapse"] and not r["probed_collapse"])
    nc = sum(1 for r in rows if not r["control_collapse"] and r["probed_collapse"])
    nn_ = sum(1 for r in rows if not r["control_collapse"] and not r["probed_collapse"])

    p_ctrl = (cc + cn) / n
    p_prob = (cc + nc) / n

    print("\n=== Paired 2x2 contingency table (N=60) ===")
    print(f"{'':<22}{'probed: C':<12}{'probed: -':<12}{'row total':<10}")
    print(f"{'control: C':<22}{cc:<12d}{cn:<12d}{cc + cn:<10d}")
    print(f"{'control: -':<22}{nc:<12d}{nn_:<12d}{nc + nn_:<10d}")
    print(f"{'col total':<22}{cc + nc:<12d}{cn + nn_:<12d}{n:<10d}")

    cl, cu = base._wilson_ci(cc + cn, n)
    pl, pu = base._wilson_ci(cc + nc, n)
    print(f"\n  control rate:  {p_ctrl:.3f}  Wilson 95% CI [{cl:.3f}, {cu:.3f}]   "
          f"({cc + cn}/{n})")
    print(f"  probed  rate:  {p_prob:.3f}  Wilson 95% CI [{pl:.3f}, {pu:.3f}]   "
          f"({cc + nc}/{n})")
    delta, (dl, du) = base._diff_ci(cc + cn, n, cc + nc, n)
    print(f"  rate diff (control - probed): {delta:+.3f}   "
          f"Newcombe 95% CI [{dl:+.3f}, {du:+.3f}]")
    h = base._cohens_h(p_ctrl, p_prob)
    print(f"  Cohen's h:     {h:+.3f}")

    print(f"\n  Discordant: control-only={cn}, probed-only={nc}")
    _, p_mc = base._mcnemar_exact(cn, nc)
    print(f"  McNemar exact p = {p_mc:.4f}")
    table = [[cc + cn, nc + nn_], [cc + nc, cn + nn_]]
    odds, p_f = fisher_exact(table, alternative="two-sided")
    print(f"  Fisher exact p = {p_f:.4f}   OR={odds:.3f}")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    M = np.array([[cc, cn], [nc, nn_]])
    im = axes[0].imshow(M, cmap="Blues", aspect="auto")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["probed: collapse", "probed: none"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["control: collapse", "control: none"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(M[i, j]), ha="center", va="center",
                         fontsize=18,
                         color=("white" if M[i, j] > M.max() / 2 else "black"))
    axes[0].set_title(
        f"Paired contingency (N={n})\n"
        f"control={p_ctrl:.2f}  probed={p_prob:.2f}\n"
        f"McNemar p={p_mc:.4f}  Cohen's h={h:+.3f}"
    )
    plt.colorbar(im, ax=axes[0])

    seeds = [r["seed"] for r in rows]
    cy = [1 if r["control_collapse"] else 0 for r in rows]
    py = [1 if r["probed_collapse"] else 0 for r in rows]
    axes[1].scatter(seeds, cy, color="navy", s=40, label="control", marker="o")
    axes[1].scatter(seeds, [v + 0.05 for v in py], color="crimson", s=40,
                    label="probed", marker="x", linewidths=2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["no collapse", "collapse"])
    axes[1].set_xlabel("Seed")
    axes[1].set_title("Per-seed outcome (offset)")
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].legend(loc="center right")
    plt.tight_layout()
    out = os.path.join(base.FIG_DIR, "perturbation_sensitivity_n60.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    out_json = os.path.join(base.RESULTS_DIR, "perturbation_sensitivity_n60.json")
    with open(out_json, "w") as f:
        json.dump({
            "n": n,
            "table": {"cc": cc, "cn": cn, "nc": nc, "nn": nn_},
            "p_control": p_ctrl, "p_probed": p_prob,
            "wilson_ci_control": list(base._wilson_ci(cc + cn, n)),
            "wilson_ci_probed": list(base._wilson_ci(cc + nc, n)),
            "diff": delta, "newcombe_ci": list((dl, du)),
            "cohens_h": h, "mcnemar_p": p_mc, "fisher_p": p_f,
            "fisher_odds_ratio": odds,
            "rows": rows,
        }, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
