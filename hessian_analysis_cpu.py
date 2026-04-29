"""Deterministic CPU rerun of ``hessian_analysis.py``.

Same code path (we reuse ``run_with_curvature``), but on CPU with
``torch.use_deterministic_algorithms(True)`` so that the trajectory is
bitwise-reproducible. With probe RNG already isolated in
``hessian_utils``, this run gives a clean baseline for which seeds
collapse and is what the GPU baseline should be compared against.

Also produces a side-by-side comparison plot: for each seed (0..9), the
collapse outcome under CPU-deterministic vs GPU.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import hessian_analysis  # for run_main

FIG_DIR = "figures"
RESULTS_DIR = "results"


def _enable_determinism():
    # cuBLAS workspace must be set BEFORE any CUDA op, but we won't be using
    # CUDA here. Setting it anyway keeps the environment hermetic if someone
    # imports this module after other code has touched CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)


def _make_comparison_plot(cpu_path, gpu_path, out_path):
    if not (os.path.exists(cpu_path) and os.path.exists(gpu_path)):
        print(f"[skip] need both {cpu_path} and {gpu_path}")
        return
    cpu = json.load(open(cpu_path))
    gpu = json.load(open(gpu_path))

    seeds = sorted({int(s) for s in cpu.keys()} | {int(s) for s in gpu.keys()})

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: collapse-epoch dot plot per seed.
    for ax_idx, (results, color, label) in enumerate(
        [(cpu, "navy", "CPU deterministic"),
         (gpu, "crimson", "GPU non-deterministic")]
    ):
        ys, xs = [], []
        for s in seeds:
            ep = results.get(str(s), {}).get("collapse_epoch")
            if ep is not None:
                xs.append(s)
                ys.append(ep)
        axes[0].scatter(xs, ys, color=color, s=80, label=label,
                        marker="o" if ax_idx == 0 else "x", linewidths=2)
    axes[0].set_ylabel("Detected collapse epoch")
    axes[0].set_title("Collapse outcome per seed: CPU-deterministic vs GPU")
    axes[0].set_xticks(seeds)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Bottom: λ_max trajectory mean across seeds for each tag.
    for results, color, label in [
        (cpu, "navy", "CPU deterministic"),
        (gpu, "crimson", "GPU non-deterministic"),
    ]:
        all_eps, all_lm = None, []
        for s in seeds:
            d = results.get(str(s))
            if d is None:
                continue
            if all_eps is None:
                all_eps = np.asarray(d["hess_epoch"])
            all_lm.append(d["lambda_max"])
        if all_lm:
            arr = np.asarray(all_lm)
            mean = arr.mean(0)
            std = arr.std(0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            axes[1].plot(all_eps, mean, color=color, linewidth=2, label=label)
            axes[1].fill_between(all_eps, mean - std, mean + std,
                                 color=color, alpha=0.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\lambda_{\max}(H)$  (mean ± std across seeds)")
    axes[1].set_title("Top Hessian eigenvalue trajectory: CPU vs GPU")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure to {out_path}")


def main():
    _enable_determinism()
    cpu_device = torch.device("cpu")
    hessian_analysis.run_main(cpu_device, tag="cpu")

    _make_comparison_plot(
        os.path.join(RESULTS_DIR, "hessian_cpu.json"),
        os.path.join(RESULTS_DIR, "hessian_gpu.json"),
        os.path.join(FIG_DIR, "hessian_cpu_vs_gpu.png"),
    )


if __name__ == "__main__":
    main()
