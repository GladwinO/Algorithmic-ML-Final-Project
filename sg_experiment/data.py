"""Datasets used in the experiments."""
from __future__ import annotations

import torch
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler

from .device import DEVICE


def get_dataset(name: str = "moons", n_samples: int = 1000, noise: float = 0.2):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    return X, y
