"""Plotting helpers."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_")


def plot_metrics_over_training(
    histories: List[Dict],
    labels: List[str],
    metric: str = "cosine_sim",
    title: str = "",
    out_dir: str = "figures",
):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for h, label in zip(histories, labels):
        ax.plot(h["epoch"], h[metric], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    if metric == "cosine_sim":
        ax.axhline(0, color="red", linestyle="--", alpha=0.4)
    elif metric == "sign_agreement":
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{metric}_{_slug(title)}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all_metrics(history: Dict, config_label: str, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["cosine_sim"])
    axes[0].set_title("Cosine Similarity (SG vs True)")
    axes[0].set_xlabel("Epoch")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5, label="orthogonal")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["sign_agreement"])
    axes[1].set_title("Sign Agreement Fraction")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["true_loss"])
    axes[2].set_title("Training Loss (True Gradient)")
    axes[2].set_xlabel("Epoch")

    fig.suptitle(config_label)
    plt.tight_layout()
    path = os.path.join(out_dir, f"all_metrics_{_slug(config_label)}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path
