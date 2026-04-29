"""Improvement #7: bootstrap confidence intervals for the per-seed scalars
recorded by the rigor experiments.

Reads the result JSON files produced by ``hessian_analysis.py``,
``hessian_analysis_cpu.py``, ``perturbation_sensitivity.py``, the N=60
re-run, and the ``beta_sg`` sweep, and reports nonparametric bootstrap
95% CIs on:

  - mean / median collapse epoch among collapsed seeds
  - collapse rate (proportion)
  - mean lambda_max over the [200, 280] epoch window
  - per-condition (control vs probed) collapse rates
  - rate difference (paired bootstrap on the seed-level outcomes)

This replaces the parametric Welch t-tests for scalars where the seed
is the unit of replication and N is small (10-60).
"""
from __future__ import annotations

import json
import os
from typing import Callable, List, Sequence, Tuple

import numpy as np

RESULTS_DIR = "results"
N_BOOT = 10000
RNG_SEED = 20260429


def _bootstrap_ci(data: Sequence[float], stat: Callable[[np.ndarray], float],
                  n_boot: int = N_BOOT, conf: float = 0.95,
                  rng: np.random.Generator = None) -> Tuple[float, float, float]:
    """Return (point_estimate, lower, upper) percentile bootstrap CI."""
    arr = np.asarray(data, dtype=float)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(arr))
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        boots[i] = stat(sample)
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (point, float(lo), float(hi))


def _paired_diff_ci(a: Sequence[bool], b: Sequence[bool],
                    n_boot: int = N_BOOT, conf: float = 0.95
                    ) -> Tuple[float, float, float]:
    """Paired bootstrap CI for rate(a) - rate(b)."""
    rng = np.random.default_rng(RNG_SEED)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    lo, hi = np.quantile(diffs, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (point, float(lo), float(hi))


def _section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _do_hessian(path, tag):
    if not os.path.exists(path):
        print(f"[skip] {path}")
        return
    with open(path) as f:
        d = json.load(f)
    seeds = sorted(int(s) for s in d.keys())
    eps = [d[str(s)]["collapse_epoch"] for s in seeds]
    coll_eps = [e for e in eps if e is not None]
    rate = sum(1 for e in eps if e is not None) / len(eps)

    _section(f"Hessian results [{tag}]   (n_seeds={len(seeds)})")
    print(f"collapse rate: {rate:.3f}  ({sum(1 for e in eps if e is not None)}"
          f"/{len(eps)})")
    if coll_eps:
        p, lo, hi = _bootstrap_ci(coll_eps, np.mean)
        print(f"mean collapse epoch (collapsed only):   "
              f"{p:.1f}  95% CI [{lo:.1f}, {hi:.1f}]")
        p, lo, hi = _bootstrap_ci(coll_eps, np.median)
        print(f"median collapse epoch (collapsed only): "
              f"{p:.1f}  95% CI [{lo:.1f}, {hi:.1f}]")

    # Mean lambda_max over the 200-280 window per seed.
    means = []
    for s in seeds:
        ep = np.asarray(d[str(s)]["hess_epoch"])
        lm = np.asarray(d[str(s)]["lambda_max"])
        mask = (ep >= 200) & (ep <= 280)
        if mask.any():
            means.append(float(lm[mask].mean()))
    if means:
        p, lo, hi = _bootstrap_ci(means, np.mean)
        print(f"mean lambda_max over epochs 200-280: "
              f"{p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")


def _do_perturbation(path, tag):
    if not os.path.exists(path):
        print(f"[skip] {path}")
        return
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    a = [r["control_collapse"] for r in rows]
    b = [r["probed_collapse"] for r in rows]
    _section(f"Perturbation [{tag}]   (n_seeds={len(rows)})")
    pa, la, ha = _bootstrap_ci([float(x) for x in a], np.mean)
    pb, lb, hb = _bootstrap_ci([float(x) for x in b], np.mean)
    print(f"control rate: {pa:.3f}  95% CI [{la:.3f}, {ha:.3f}]")
    print(f"probed  rate: {pb:.3f}  95% CI [{lb:.3f}, {hb:.3f}]")
    pd, ld, hd = _paired_diff_ci(a, b)
    print(f"paired diff (control - probed): "
          f"{pd:+.3f}  95% CI [{ld:+.3f}, {hd:+.3f}]")
    print(f"reported McNemar p (from JSON): {d['mcnemar_p']:.4f}")
    print(f"reported Cohen's h (from JSON): {d['cohens_h']:+.3f}")


def _do_beta_sweep(path):
    if not os.path.exists(path):
        print(f"[skip] {path}")
        return
    with open(path) as f:
        d = json.load(f)
    _section("Beta_sg sweep   (per-beta bootstrap CIs)")
    print(f"{'beta_sg':<10}{'rate (CI)':<32}"
          f"{'median ep (CI)':<28}{'mean final cos (CI)':<28}")
    print("-" * 98)
    for k in sorted(d.keys(), key=float):
        v = d[k]
        eps = v["epochs"]
        finals = v["finals"]
        rate_data = [1.0 if e is not None else 0.0 for e in eps]
        coll_eps = [e for e in eps if e is not None]
        pr, lr, hr = _bootstrap_ci(rate_data, np.mean)
        if coll_eps:
            pm, lm, hm = _bootstrap_ci(coll_eps, np.median)
            ep_str = f"{pm:.1f} [{lm:.1f}, {hm:.1f}]"
        else:
            ep_str = "n/a"
        pf, lf, hf = _bootstrap_ci(finals, np.mean)
        rate_str = f"{pr:.2f} [{lr:.2f}, {hr:.2f}]"
        f_str = f"{pf:.3f} [{lf:.3f}, {hf:.3f}]"
        print(f"{k:<10}{rate_str:<32}{ep_str:<28}{f_str:<28}")


def main():
    print("=== IMPROVEMENT #7: bootstrap CIs (n_boot={}) ===".format(N_BOOT))
    _do_hessian(os.path.join(RESULTS_DIR, "hessian_gpu.json"), "GPU")
    _do_hessian(os.path.join(RESULTS_DIR, "hessian_cpu.json"), "CPU")
    _do_perturbation(os.path.join(RESULTS_DIR,
                                  "perturbation_sensitivity.json"), "N=30")
    _do_perturbation(os.path.join(RESULTS_DIR,
                                  "perturbation_sensitivity_n60.json"), "N=60")
    _do_beta_sweep(os.path.join(RESULTS_DIR, "beta_sg_collapse_sweep.json"))


if __name__ == "__main__":
    main()
