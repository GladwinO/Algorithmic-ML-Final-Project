"""Run all four SG-vs-true-gradient experiments and save plots + results."""
from __future__ import annotations

import json
import os

from sg_experiment.device import DEVICE
from sg_experiment.experiment import run_experiment
from sg_experiment.plots import plot_all_metrics, plot_metrics_over_training

FIG_DIR = "figures"
RESULTS_DIR = "results"


def _base(**overrides):
    cfg = {
        "hidden_dim": 64,
        "num_layers": 3,
        "beta_f": 50,
        "beta_sg": 5,
        "dataset": "moons",
        "n_epochs": 300,
        "lr": 0.01,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


# --- Experiment 1: depth ---
depth_configs = [
    _base(num_layers=1),
    _base(num_layers=2),
    _base(num_layers=3),
    _base(num_layers=4),
]
depth_labels = ["1 layer", "2 layers", "3 layers", "4 layers"]

# --- Experiment 2: beta_sg ---
beta_configs = [
    _base(beta_sg=1),
    _base(beta_sg=5),
    _base(beta_sg=20),
    _base(beta_sg=49),
]
beta_labels = ["betaSG=1", "betaSG=5", "betaSG=20", "betaSG=49"]

# --- Experiment 3: width ---
width_configs = [
    _base(hidden_dim=16),
    _base(hidden_dim=64),
    _base(hidden_dim=256),
]
width_labels = ["width=16", "width=64", "width=256"]

# --- Experiment 4: datasets ---
dataset_configs = [
    _base(dataset="moons"),
    _base(dataset="circles"),
]
dataset_labels = ["moons", "circles"]


def _run(configs, labels, title_prefix):
    print(f"\n=== {title_prefix} ===")
    histories = []
    for cfg, label in zip(configs, labels):
        print(f"  running {label} ... ", end="", flush=True)
        h = run_experiment(cfg)
        histories.append(h)
        print(
            f"final cos={h['cosine_sim'][-1]:.3f}  "
            f"sign={h['sign_agreement'][-1]:.3f}  "
            f"loss={h['true_loss'][-1]:.4f}"
        )
    return histories


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    all_results = {}

    # ---- Experiment 1 ----
    depth_histories = _run(depth_configs, depth_labels, "Depth experiment")
    for metric, t in [
        ("cosine_sim", "Effect of Network Depth on SG Alignment"),
        ("sign_agreement", "Effect of Network Depth on Sign Agreement"),
        ("relative_magnitude", "Effect of Network Depth on Relative Magnitude"),
    ]:
        plot_metrics_over_training(depth_histories, depth_labels, metric, t, FIG_DIR)
    for h, label in zip(depth_histories, depth_labels):
        plot_all_metrics(h, f"depth {label}", FIG_DIR)
    all_results["depth"] = {
        "configs": depth_configs,
        "labels": depth_labels,
        "histories": depth_histories,
    }

    # ---- Experiment 2 ----
    beta_histories = _run(beta_configs, beta_labels, "Beta_sg experiment")
    for metric, t in [
        ("cosine_sim", "Effect of Surrogate Steepness on SG Alignment"),
        ("sign_agreement", "Effect of Surrogate Steepness on Sign Agreement"),
        ("relative_magnitude", "Effect of Surrogate Steepness on Relative Magnitude"),
    ]:
        plot_metrics_over_training(beta_histories, beta_labels, metric, t, FIG_DIR)
    all_results["beta"] = {
        "configs": beta_configs,
        "labels": beta_labels,
        "histories": beta_histories,
    }

    # ---- Experiment 3 ----
    width_histories = _run(width_configs, width_labels, "Width experiment")
    for metric, t in [
        ("cosine_sim", "Effect of Network Width on SG Alignment"),
        ("sign_agreement", "Effect of Network Width on Sign Agreement"),
    ]:
        plot_metrics_over_training(width_histories, width_labels, metric, t, FIG_DIR)
    all_results["width"] = {
        "configs": width_configs,
        "labels": width_labels,
        "histories": width_histories,
    }

    # ---- Experiment 4 ----
    dataset_histories = _run(dataset_configs, dataset_labels, "Dataset experiment")
    for metric, t in [
        ("cosine_sim", "Effect of Dataset Geometry on SG Alignment"),
        ("sign_agreement", "Effect of Dataset Geometry on Sign Agreement"),
    ]:
        plot_metrics_over_training(
            dataset_histories, dataset_labels, metric, t, FIG_DIR
        )
    all_results["dataset"] = {
        "configs": dataset_configs,
        "labels": dataset_labels,
        "histories": dataset_histories,
    }

    out_path = os.path.join(RESULTS_DIR, "all_histories.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved histories to {out_path}")
    print(f"Saved figures to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
