"""Regenerate selected research figures with proper error bars / uncertainty
bands.

Per-seed per-epoch traces for figs 1, 2, 6 come from
``results/per_seed_traces.json`` (produced by ``collect_per_seed_traces.py``).
Beta-sweep summaries for fig 8 come from
``results/beta_sg_collapse_sweep.json``.

Run with: ``python regenerate_figures_with_errorbars.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def _load_json(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _band(ax, epochs, arr, color, label, *, kind="std", alpha=0.15, lw=2.0):
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    err = sd if kind == "std" else sd / np.sqrt(arr.shape[0])
    ax.plot(epochs, mean, color=color, linewidth=lw, label=label)
    ax.fill_between(epochs, mean - err, mean + err, color=color, alpha=alpha,
                    linewidth=0)


# --------------------------------------------------------------------------
# FIGURE 1: depth_controlled_vs_uncontrolled_comparison.png
# --------------------------------------------------------------------------
def figure_1():
    traces = _load_json("per_seed_traces.json")
    if traces is None or "depth_uncontrolled" not in traces:
        print("Figure 1 skipped: results/per_seed_traces.json missing. "
              "Run `python collect_per_seed_traces.py` first.")
        return

    n_epochs = traces["meta"]["n_epochs"]
    epochs = np.arange(n_epochs)
    unc = traces["depth_uncontrolled"]
    ctrl = traces["depth_controlled"]
    target_params = ctrl["target_params"]
    widths = ctrl["widths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = plt.cm.viridis(np.linspace(0.1, 0.85, 4))

    for color, d in zip(palette, [1, 2, 3, 4]):
        _band(axes[0], epochs, unc[str(d)], color=color,
              label=f"{d} layer{'s' if d != 1 else ''}", kind="std",
              alpha=0.15)
    axes[0].set_title("Uncontrolled Depth (fixed width=64)\n"
                      "(shaded: mean \u00b1 1 std across seeds)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cosine similarity (SG vs true gradient)")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for color, d in zip(palette, [1, 2, 3, 4]):
        w = widths[str(d)]
        _band(axes[1], epochs, ctrl["traces"][str(d)], color=color,
              label=f"depth={d}, width={w}", kind="std", alpha=0.15)
    axes[1].set_title(f"Controlled Depth (fixed param count \u2248 {target_params})\n"
                      "(shaded: mean \u00b1 1 std across seeds)")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "depth_controlled_vs_uncontrolled_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 1 regenerated: {out.name}")


# --------------------------------------------------------------------------
# Collapse-detection helpers (mirror collapse_detection.py)
# --------------------------------------------------------------------------
def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if window <= 1 or n == 0:
        return x.copy()
    out = np.empty(n, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    for t in range(n):
        lo = max(0, t - window + 1)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def _detect_collapse(cos, window=20, threshold=0.15, min_epoch=50):
    arr = np.asarray(cos, dtype=float)
    smoothed = _rolling_mean(arr, window)
    n = len(arr)
    for t in range(max(min_epoch, window), n - window):
        prev_mean = float(smoothed[t - 1])
        post_mean = float(arr[t:t + window].mean())
        if prev_mean - post_mean > threshold:
            return t
    return None


# --------------------------------------------------------------------------
# FIGURE 2: collapse_detection_4layer.png
# --------------------------------------------------------------------------
def figure_2():
    traces = _load_json("per_seed_traces.json")
    if traces is None or "layerwise_4layer" not in traces:
        print("Figure 2 skipped: results/per_seed_traces.json missing.")
        return
    L = traces["layerwise_4layer"]
    cos = np.asarray(L["cosine_sim"])
    gnorm = np.asarray(L["grad_norm_true"])
    loss = np.asarray(L["true_loss"])
    n_seeds, n_epochs = cos.shape
    epochs = np.arange(n_epochs)

    detected_eps = [_detect_collapse(c) for c in cos.tolist()]
    coll_mask = np.array([e is not None for e in detected_eps])
    n_coll = int(coll_mask.sum())
    n_not = n_seeds - n_coll
    detected = [e for e in detected_eps if e is not None]
    if detected:
        mean_ep = float(np.mean(detected))
        std_ep = float(np.std(detected, ddof=1)) if len(detected) > 1 else 0.0
    else:
        mean_ep, std_ep = float("nan"), float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    panels = [
        (cos, axes[0], "Cosine similarity (SG vs true)", "cos sim"),
        (gnorm, axes[1], "True gradient L2 norm", "||g_true||"),
        (loss, axes[2], "Training loss (BCE)", "loss"),
    ]
    color_coll = "crimson"
    color_not = "steelblue"
    for arr, ax, title, ylabel in panels:
        if n_coll > 0:
            sub = arr[coll_mask]
            mean = sub.mean(axis=0)
            sem = (sub.std(axis=0, ddof=1) / np.sqrt(n_coll)
                   if n_coll > 1 else np.zeros_like(mean))
            ax.fill_between(epochs, mean - sem, mean + sem,
                            color=color_coll, alpha=0.25, linewidth=0)
            ax.plot(epochs, mean, color=color_coll, linewidth=2.2,
                    label=f"collapsed seeds (n={n_coll})")
        if n_not > 0:
            sub = arr[~coll_mask]
            mean = sub.mean(axis=0)
            sem = (sub.std(axis=0, ddof=1) / np.sqrt(n_not)
                   if n_not > 1 else np.zeros_like(mean))
            ax.fill_between(epochs, mean - sem, mean + sem,
                            color=color_not, alpha=0.25, linewidth=0)
            ax.plot(epochs, mean, color=color_not, linewidth=2.2,
                    label=f"non-collapsed seeds (n={n_not})")
        if detected:
            ax.axvline(mean_ep, color="black", linestyle="--", alpha=0.6,
                       linewidth=1.0,
                       label=f"mean collapse epoch ≈ {mean_ep:.0f}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.text(
        0.5, 0.005,
        f"{n_coll}/{n_seeds} runs collapse  |  mean detected epoch "
        f"{mean_ep:.0f} ± {std_ep:.0f} (sd)  |  bands = mean ± 1 SEM "
        f"within group",
        ha="center", va="bottom", fontsize=9, color="dimgray",
    )
    axes[2].set_xlabel("Epoch")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIG_DIR / "collapse_detection_4layer.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 2 regenerated: {out.name}")


# --------------------------------------------------------------------------
# FIGURES 3, 4, 5, 7, 9: explicitly no error bars per spec.
# --------------------------------------------------------------------------
def figure_3():
    print("Figure 3: no error bars needed per spec; "
          "collapse_threshold_sensitivity.png left unchanged.")


def figure_4():
    print("Figure 4: no error bars needed per spec; "
          "saturation_vs_cosine.png left unchanged.")


def figure_5():
    print("Figure 5: no error bars meaningful per spec; "
          "landscape_hessplane.png left unchanged.")


def figure_7():
    print("Figure 7: no error bars meaningful per spec; "
          "collapse_propagation_heatmap.png left unchanged.")


def figure_9():
    print("Figure 9: no error bars meaningful per spec; "
          "perturbation_sensitivity_n60.png left unchanged.")


# --------------------------------------------------------------------------
# FIGURE 6: layerwise_cosine_all_seeds.png
# --------------------------------------------------------------------------
def figure_6():
    traces = _load_json("per_seed_traces.json")
    if traces is None or "layerwise_4layer" not in traces:
        print("Figure 6 skipped: results/per_seed_traces.json missing.")
        return
    L = traces["layerwise_4layer"]
    arr = np.asarray(L["cos_per_layer"])  # (n_seeds, n_epochs, n_layers)
    n_seeds, n_epochs, n_layers = arr.shape
    n_hidden = n_layers - 1  # exclude readout (matches layerwise_alignment.py)
    epochs = np.arange(n_epochs)

    cos_total = np.asarray(L["cosine_sim"])
    detected = [_detect_collapse(c) for c in cos_total.tolist()]
    detected = [e for e in detected if e is not None]
    mean_collapse = float(np.mean(detected)) if detected else None

    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_hidden))
    fig, axes = plt.subplots(n_hidden, 1, figsize=(9, 2.4 * n_hidden),
                             sharex=True)
    if n_hidden == 1:
        axes = [axes]

    for li in range(n_hidden):
        ax = axes[li]
        per_seed = arr[:, :, li]
        for run in per_seed:
            ax.plot(epochs, run, color="lightgray", linewidth=0.7, alpha=0.7)
        mean = np.nanmean(per_seed, axis=0)
        sd = np.nanstd(per_seed, axis=0, ddof=1) if n_seeds > 1 \
            else np.zeros_like(mean)
        ax.fill_between(epochs, mean - sd, mean + sd, color=palette[li],
                        alpha=0.2, linewidth=0, label="mean ± 1 std")
        ax.plot(epochs, mean, color=palette[li], linewidth=2,
                label=f"mean (layer {li})")
        if mean_collapse is not None:
            ax.axvline(mean_collapse, color="red", linestyle="--",
                       label=f"mean collapse ≈ {mean_collapse:.0f}")
        ax.set_ylabel(f"layer {li} cos")
        ax.set_ylim(-0.2, 1.05)
        ax.set_title(f"Per-layer cosine similarity, hidden layer {li} "
                     f"(0=input-side)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out = FIG_DIR / "layerwise_cosine_all_seeds.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 6 regenerated: {out.name}")


# --------------------------------------------------------------------------
# FIGURE 8: beta_sg_collapse_sweep.png
# --------------------------------------------------------------------------
def _clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _bootstrap_median_ci(x, n_boot=10000, alpha=0.05, rng=None):
    """Percentile bootstrap 95% CI for the median."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2))
    hi = float(np.quantile(meds, 1 - alpha / 2))
    return lo, hi


def figure_8():
    data = _load_json("beta_sg_collapse_sweep.json")
    if data is None:
        print("Figure 8 skipped: results/beta_sg_collapse_sweep.json not found.")
        return

    Z = 1.959963984540054  # 97.5% normal quantile -> 95% CI
    rng = np.random.default_rng(0)

    betas = sorted(int(k) for k in data.keys())
    rates, rate_lo, rate_hi = [], [], []
    med_eps, med_lo, med_hi = [], [], []
    finals_mean, finals_lo, finals_hi = [], [], []
    n_seeds_total = 0

    for b in betas:
        s = data[str(b)]
        n_seeds = len(s["finals"])
        n_seeds_total = n_seeds
        n_coll = int(s["n_collapsed"])

        # Left: proportion + Clopper-Pearson exact 95% CI.
        rate = n_coll / n_seeds
        lo, hi = _clopper_pearson(n_coll, n_seeds)
        rates.append(rate)
        rate_lo.append(rate - lo)
        rate_hi.append(hi - rate)

        # Centre: median + percentile bootstrap 95% CI of the median.
        eps = [e for e in s["epochs"] if e is not None]
        if eps:
            m = float(np.median(eps))
            blo, bhi = _bootstrap_median_ci(eps, rng=rng)
            med_eps.append(m)
            med_lo.append(m - blo)
            med_hi.append(bhi - m)
        else:
            med_eps.append(None)
            med_lo.append(0.0)
            med_hi.append(0.0)

        # Right: mean + normal 95% CI = mean +/- 1.96 * SEM.
        finals = np.asarray(s["finals"], dtype=float)
        mu = float(finals.mean())
        sem = (float(finals.std(ddof=1) / np.sqrt(len(finals)))
               if len(finals) > 1 else 0.0)
        finals_mean.append(mu)
        finals_lo.append(Z * sem)
        finals_hi.append(Z * sem)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: collapse rate.
    yerr = np.array([rate_lo, rate_hi])
    axes[0].errorbar(betas, rates, yerr=yerr, fmt="o-", color="crimson",
                     linewidth=2, capsize=5, ecolor="crimson", elinewidth=1.2)
    axes[0].set_xlabel(r"$\beta_{SG}$")
    axes[0].set_ylabel(f"Collapse rate (out of {n_seeds_total})")
    axes[0].set_title("Collapse rate vs surrogate steepness\n(95% CI)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Centre: median collapse epoch.
    med_x = [b for b, m in zip(betas, med_eps) if m is not None]
    med_y = [m for m in med_eps if m is not None]
    med_lo_f = [lo for lo, m in zip(med_lo, med_eps) if m is not None]
    med_hi_f = [hi for hi, m in zip(med_hi, med_eps) if m is not None]
    axes[1].errorbar(med_x, med_y, yerr=np.array([med_lo_f, med_hi_f]),
                     fmt="o-", color="navy", linewidth=2, capsize=5,
                     ecolor="navy", elinewidth=1.2)
    axes[1].set_xlabel(r"$\beta_{SG}$")
    axes[1].set_ylabel("Median collapse epoch")
    axes[1].set_title("Collapse timing vs surrogate steepness\n(95% CI)")
    axes[1].grid(True, alpha=0.3)

    # Right: mean final cosine similarity.
    axes[2].errorbar(betas, finals_mean,
                     yerr=np.array([finals_lo, finals_hi]),
                     fmt="o-", color="forestgreen", linewidth=2, capsize=5,
                     ecolor="forestgreen", elinewidth=1.2)
    axes[2].set_xlabel(r"$\beta_{SG}$")
    axes[2].set_ylabel("Mean final cosine similarity")
    axes[2].set_title("Final alignment vs surrogate steepness\n(95% CI)")
    axes[2].grid(True, alpha=0.3)

    fig.text(
        0.5, -0.02,
        "95% CIs: left = Clopper\u2013Pearson exact (proportion); "
        "centre = percentile bootstrap of the median (10\u202f000 resamples); "
        "right = normal CI of the mean (mean \u00b1 1.96\u00b7SEM).",
        ha="center", va="top", fontsize=8, color="dimgray",
    )

    plt.tight_layout()
    out = FIG_DIR / "beta_sg_collapse_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 8 regenerated: {out.name}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()


if __name__ == "__main__":
    main()
