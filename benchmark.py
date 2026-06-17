#!/usr/bin/env python3
"""
benchmark.py
=============
Compares five lightweight architectures (ST-GCN, DCRNN, Bi-GRCN,
SA-DGWN, EG-NODE -- see models/architectures.py) on a controlled
synthetic binary-classification task, to sanity-check relative
architecture behavior independent of TimeAwareSTGCN's real-data
training run in train.py.

Benchmark Methodology (read this before trusting any number below)
---------------------------------------------------------------------
This benchmark intentionally uses synthetic, uniformly-random
features (np.random.rand) with a threshold-derived binary label. It
is NOT a traffic dataset and is NOT connected to train.py's model or
data. Its purpose is narrow: confirm that all five architectures
train stably end-to-end, produce comparable evaluation plots, and
behave sensibly on a known, simple decision boundary -- a "do these
all actually work" smoke test, not a claim about traffic-forecasting
performance. See README "Benchmark Methodology" for the full
rationale and how to point this script at real, held-out traffic data
instead.

Usage
-----
    python benchmark.py
    python benchmark.py --samples 1000 --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from config import BenchmarkConfig
from models import build_benchmark_models

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute both regression-style and classification-style metrics.

    Why both: y_pred comes from a sigmoid output in [0, 1], and the
    task is genuinely binary classification (BCEWithLogitsLoss is the
    training loss). MAE/RMSE/R2 on a [0,1]-vs-{0,1} target are
    reported here for continuity with train.py's metric set so the
    two scripts' output is visually comparable -- but Accuracy is the
    metric that actually answers "did the model get the classification
    right." Don't read R2 here as a meaningful regression score; this
    is a classification task wearing regression-metric clothing on
    purpose, for side-by-side plotting.
    """
    y_pred_class = np.round(y_pred)
    y_true_class = np.round(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) /
                           np.clip(y_true, 1e-2, None))) * 100
    r2 = r2_score(y_true, y_pred)
    acc = accuracy_score(y_true_class, y_pred_class)
    return {
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
                     "Accuracy": acc},
        "y_true_class": y_true_class,
        "y_pred_class": y_pred_class,
    }


def train_and_evaluate(model: nn.Module, X_train: torch.Tensor,
                        y_train: torch.Tensor, X_test: torch.Tensor,
                        y_test: torch.Tensor, cfg: BenchmarkConfig) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()

    best_loss = float("inf")
    wait = 0
    losses: list[float] = []

    for _ in range(cfg.num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test)).numpy()

    result = evaluate(y_test.numpy(), preds)
    result["losses"] = losses
    return result


def plot_metric_subplots(metric_values: dict, labels: list[str],
                          out_path: str) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    axs = axs.flatten()

    for i, metric in enumerate(metric_values):
        axs[i].bar(labels, metric_values[metric], color="teal")
        axs[i].set_title(f"Model Comparison - {metric}")
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Models")
        axs[i].grid(True)

    for j in range(len(metric_values), len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle("Evaluation Metrics by Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_normalized_metrics(metric_values: dict, labels: list[str],
                             out_path: str) -> None:
    """Min-max normalize each metric across models so they share a
    0-1 axis for direct visual comparison. MAE/RMSE/MAPE are inverted
    (1 - normalized) so that "higher bar = better" holds consistently
    across every metric in the chart, including the error ones.
    """
    normalized = {}
    for key, raw_values in metric_values.items():
        values = np.array(raw_values, dtype=float)
        span = values.max() - values.min() + 1e-9
        if key in ("MAE", "RMSE", "MAPE"):
            norm = 1 - ((values - values.min()) / span)
        else:
            norm = (values - values.min()) / span
        normalized[key] = np.maximum(norm, 0.02)  # keep zero-bars visible

    x = np.arange(len(labels))
    width = 0.15

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(normalized):
        plt.bar(x + i * width, normalized[metric], width, label=metric)

    plt.xticks(x + width * 2, labels, rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Normalized Score (higher = better, all metrics)")
    plt.title("Normalized Model Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    overrides = {}
    if args.samples:
        overrides["n_samples"] = args.samples
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        cfg = BenchmarkConfig(**{**cfg.__dict__, **overrides})

    os.makedirs(cfg.figures_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("NOTE: this benchmark runs on synthetic data and is not "
          "connected to train.py. See module docstring / README "
          "'Benchmark Methodology' before interpreting results.\n")

    data = np.random.rand(cfg.n_samples, cfg.n_features)
    labels_bin = (data[:, 2] > 0.5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_bin, test_size=cfg.test_size, random_state=cfg.seed)

    X_train_t = torch.tensor(X_train).float().reshape(-1, cfg.n_features, 1)
    X_test_t = torch.tensor(X_test).float().reshape(-1, cfg.n_features, 1)
    y_train_t = torch.tensor(y_train).float().reshape(-1, 1)
    y_test_t = torch.tensor(y_test).float().reshape(-1, 1)

    models = build_benchmark_models(n_features=cfg.n_features)

    results = {}
    metric_values = {"MAE": [], "RMSE": [], "MAPE": [], "R2": [],
                      "Accuracy": []}
    labels: list[str] = []
    all_y_true: list[float] = []
    all_y_pred: list[float] = []

    for name, model in models.items():
        logger.info("Training %s...", name)
        result = train_and_evaluate(model, X_train_t, y_train_t, X_test_t,
                                      y_test_t, cfg)
        labels.append(name)
        for k in metric_values:
            metric_values[k].append(result["metrics"][k])
        results[name] = result["metrics"]
        all_y_true.extend(result["y_true_class"])
        all_y_pred.extend(result["y_pred_class"])

    plot_metric_subplots(metric_values, labels,
                          os.path.join(cfg.figures_dir,
                                       "benchmark_metric_subplots.png"))
    plot_normalized_metrics(metric_values, labels,
                             os.path.join(cfg.figures_dir,
                                          "benchmark_normalized.png"))

    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Combined Confusion Matrix - All Models")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_dir, "benchmark_confusion_matrix.png"),
                dpi=150)
    plt.close()

    print("\nEvaluation Results:")
    for model_name, metric in results.items():
        formatted = ", ".join(f"{k}={v:.4f}" for k, v in metric.items())
        print(f"  {model_name}: {formatted}")


if __name__ == "__main__":
    main()
