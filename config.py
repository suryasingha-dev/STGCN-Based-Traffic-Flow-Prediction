"""
config.py
==========
Centralized configuration. Nothing in this file is a secret -- the
TomTom API key is read from the environment (see .env.example) and is
never written here or anywhere else in the repo.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # --- Data ---
    data_path: str = "data/segments.csv"
    t_seq: int = 10                 # timesteps per sequence
    train_split: float = 0.8

    # --- Graph construction ---
    # Edges connect segments whose pairwise distance falls in the
    # closest `adjacency_percentile` of all pairwise distances.
    adjacency_percentile: float = 10.0

    # --- Model ---
    hidden_dim: int = 64
    dropout: float = 0.3

    # --- Optimization ---
    learning_rate: float = 0.003
    num_epochs: int = 200
    early_stop_patience: int = 20
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    # --- Reproducibility ---
    seed: int = 42

    # --- Output ---
    checkpoint_path: str = "best_model.pth"
    figures_dir: str = "assets"


@dataclass(frozen=True)
class BenchmarkConfig:
    n_samples: int = 500
    n_features: int = 4
    test_size: float = 0.2
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stop_patience: int = 5
    seed: int = 0
    figures_dir: str = "assets"


def get_tomtom_api_key() -> str | None:
    """Read the TomTom API key from the environment.

    Returns None if unset -- callers (data_loader.fetch_live_traffic)
    treat that as "skip the live API, use synthetic fallback" rather
    than an error, since the project is designed to run end-to-end
    without any credentials configured.
    """
    return os.environ.get("TOMTOM_API_KEY")
