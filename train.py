#!/usr/bin/env python3
"""
train.py
=========
Trains TimeAwareSTGCN on real road-segment coordinates, builds an
adjacency graph from pairwise distance, and evaluates speed-forecasting
accuracy.

Data flow: real CSV coordinates -> live TomTom API (if configured) ->
synthetic fallback (see data_loader.py). The dataset's `.source` field
is logged and printed so you always know which path produced a given
result.

Usage
-----
    python train.py --data-path data/segments.csv
    python train.py --data-path data/segments.csv --epochs 100 --no-map

Run `python train.py --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import TrainConfig, get_tomtom_api_key
from data_loader import load_traffic_dataset
from models import TimeAwareSTGCN

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_features(speed_series: np.ndarray, flow_series: np.ndarray,
                    t_seq: int, rng: np.random.Generator) -> np.ndarray:
    """Assemble the per-(segment, timestep) feature matrix.

    Features: [speed, speed_lag1, speed_lag2, flow, time_of_day,
    weekday, is_weekend, is_peak_period].

    The lag features (speed_lag1/2) are built with np.roll along the
    time axis, which wraps the first 1-2 timesteps around from the
    end of the sequence. We explicitly overwrite those wrapped-around
    entries with the timestep-0 value instead, so the model doesn't
    learn from an artificial wrap-around discontinuity at the start of
    every sequence.
    """
    n = speed_series.shape[0]

    lag1 = np.roll(speed_series, 1, axis=1)
    lag2 = np.roll(speed_series, 2, axis=1)
    lag1[:, 0] = speed_series[:, 0]
    lag2[:, :2] = speed_series[:, :1]

    time_of_day = np.linspace(0, 1, t_seq).reshape(1, t_seq).repeat(n, axis=0)
    weekday_raw = rng.integers(0, 7, (n, 1))
    weekday = (weekday_raw / 6.0).repeat(t_seq, axis=1)
    is_weekend = ((weekday_raw > 4).astype(float)).repeat(t_seq, axis=1)
    is_peak = (((weekday_raw * 6) % 7 > 1).astype(float)).repeat(t_seq, axis=1)

    return np.stack(
        [speed_series, lag1, lag2, flow_series, time_of_day, weekday,
         is_weekend, is_peak],
        axis=2,
    )


def build_adjacency(coords: np.ndarray,
                     percentile: float) -> torch.Tensor:
    """Build a PyG-style edge_index from pairwise coordinate distance.

    Two segments are connected if their distance falls within the
    given percentile of all pairwise distances (i.e. they're among the
    closer pairs in the dataset). This is a simple proxy for "nearby
    segments influence each other" -- it does not account for actual
    road topology (one-way streets, highway vs. side-street
    adjacency). See README "Limitations & Roadmap" for the proper-fix
    suggestion (use real road-network edges, e.g. via OSMnx).
    """
    dist_matrix = cdist(coords, coords)
    threshold = np.percentile(dist_matrix, percentile)
    adjacency = (dist_matrix < threshold).astype(int)
    edge_index = torch.tensor(np.array(adjacency.nonzero()), dtype=torch.long)
    logger.info("Built adjacency graph: %d edges among %d segments "
                "(percentile=%.1f)", edge_index.shape[1], len(coords),
                percentile)
    return edge_index


def train_model(model: nn.Module, x_tensor: torch.Tensor,
                 edge_index: torch.Tensor, y_targets: torch.Tensor,
                 train_idx: torch.Tensor, test_idx: torch.Tensor,
                 cfg: TrainConfig) -> list[float]:
    """Train with early stopping on held-out test loss.

    Note on evaluation protocol: train_idx/test_idx is a *segment*
    split (different road segments in train vs. test), not a time
    split. That means this measures how well the model generalizes to
    unseen *locations* given the graph structure, not how well it
    forecasts *future* timesteps for segments it has already seen.
    Both are valid questions; this script answers the first one. See
    README for how to adapt this to a temporal holdout if that's what
    your use case needs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_scheduler_patience,
        factor=cfg.lr_scheduler_factor,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    losses: list[float] = []

    for epoch in range(cfg.num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, edge_index).squeeze()
        loss = loss_fn(out[train_idx], y_targets[train_idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_out = model(x_tensor, edge_index).squeeze()
            test_loss = loss_fn(test_out[test_idx], y_targets[test_idx])

        scheduler.step(test_loss)
        logger.info("Epoch %03d - Train Loss: %.4f, Test Loss: %.4f",
                     epoch + 1, loss.item(), test_loss.item())

        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            torch.save(model.state_dict(), cfg.checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stop_patience:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break

    model.load_state_dict(torch.load(cfg.checkpoint_path))
    return losses


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) /
                           np.maximum(np.abs(actual), 1))) * 100
    accuracy = max(0.0, 100 - (mae / max(np.mean(actual), 1)) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Accuracy": accuracy,
            "MAPE": mape}


def save_diagnostic_plots(losses: list[float], train_metrics: dict,
                           test_metrics: dict, actual_test: np.ndarray,
                           pred_test: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    plt.close()

    for name, metrics, color in (("test", test_metrics, "orange"),
                                  ("train", train_metrics, "skyblue")):
        plt.figure(figsize=(10, 4))
        plt.bar(list(metrics.keys()), list(metrics.values()), color=color)
        plt.title(f"{name.capitalize()} Evaluation Metrics")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_metrics.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(actual_test, label="Actual (Test)", linewidth=2)
    plt.plot(pred_test, label="Predicted (Test)", linewidth=2,
              linestyle="--")
    plt.title("Predicted vs Actual Speed - Test Set")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predicted_vs_actual.png"), dpi=150)
    plt.close()

    logger.info("Saved diagnostic plots to %s/", out_dir)


def render_map_snapshot(coords: np.ndarray, predicted: np.ndarray,
                         actual: np.ndarray, out_path: str) -> None:
    """Render a single static Folium map of predictions, saved to HTML.

    The original notebook ran an infinite `while True` loop that
    redrew a live map every few seconds until manually interrupted --
    that pattern only makes sense inside an interactive Colab cell and
    will hang a non-interactive script forever. This renders one
    snapshot to an HTML file instead, which you can open in a browser
    or embed in the README. If you want the live-refresh behavior
    back for an interactive session, see the "Optional: live map"
    section in the README.
    """
    import folium

    m = folium.Map(location=[float(np.mean(coords[:, 0])),
                              float(np.mean(coords[:, 1]))],
                    zoom_start=13)

    for i, (lat, lon) in enumerate(coords):
        pred = round(float(predicted[i % len(predicted)]), 2)
        true = round(float(actual[i % len(actual)]), 2)
        if pred < 15:
            color = "darkred"
        elif pred < 30:
            color = "orange"
        elif pred < 50:
            color = "lightblue"
        else:
            color = "green"
        folium.Marker(
            location=[lat, lon],
            popup=f"Actual: {true} km/h | Predicted: {pred} km/h",
            icon=folium.Icon(color=color),
        ).add_to(m)

    m.save(out_path)
    logger.info("Saved map snapshot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None,
                         help="Path to coordinate CSV/TSV "
                              "(default: config.TrainConfig.data_path)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-map", action="store_true",
                         help="Skip rendering the Folium map snapshot.")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.data_path:
        cfg = TrainConfig(**{**cfg.__dict__, "data_path": args.data_path})
    if args.epochs:
        cfg = TrainConfig(**{**cfg.__dict__, "num_epochs": args.epochs})
    if args.seed is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "seed": args.seed})

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    dataset = load_traffic_dataset(cfg.data_path, t_seq=cfg.t_seq,
                                    api_key=get_tomtom_api_key(),
                                    seed=cfg.seed)
    logger.info("Dataset source: %s", dataset.source)
    print(f"\nData source for this run: {dataset.source}")
    if dataset.source == "synthetic_fallback":
        print("  (No live TomTom data used -- set TOMTOM_API_KEY in your "
              "environment to attempt live ingestion. See .env.example.)\n")

    n_segments = len(dataset.coords)
    features = build_features(dataset.speed_series, dataset.flow_series,
                               cfg.t_seq, rng)
    x_all = features.reshape(n_segments * cfg.t_seq, -1)
    y_all = dataset.speed_series.reshape(-1, 1)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_tensor = torch.tensor(x_scaler.fit_transform(x_all), dtype=torch.float)

    edge_index = build_adjacency(dataset.coords, cfg.adjacency_percentile)

    split = int(cfg.train_split * n_segments)
    train_idx = torch.arange(0, split)
    test_idx = torch.arange(split, n_segments)

    y_targets_raw = dataset.speed_series[:, -1]
    y_targets_scaled = torch.tensor(
        y_scaler.fit_transform(y_targets_raw.reshape(-1, 1)).squeeze(),
        dtype=torch.float,
    )

    model = TimeAwareSTGCN(input_dim=features.shape[2],
                            hidden_dim=cfg.hidden_dim, output_dim=1,
                            t_seq=cfg.t_seq)

    print("Training model...")
    losses = train_model(model, x_tensor, edge_index, y_targets_scaled,
                          train_idx, test_idx, cfg)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(x_tensor, edge_index).squeeze().numpy()
    preds_real = y_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)).squeeze()
    actual_real = y_scaler.inverse_transform(
        y_targets_scaled.numpy().reshape(-1, 1)).squeeze()

    train_metrics = compute_metrics(actual_real[train_idx.numpy()],
                                     preds_real[train_idx.numpy()])
    test_metrics = compute_metrics(actual_real[test_idx.numpy()],
                                    preds_real[test_idx.numpy()])

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}")
    print("\nFinal Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.2f}")

    save_diagnostic_plots(losses, train_metrics, test_metrics,
                           actual_real[test_idx.numpy()],
                           preds_real[test_idx.numpy()], cfg.figures_dir)

    if not args.no_map:
        render_map_snapshot(dataset.coords[test_idx.numpy()],
                             preds_real[test_idx.numpy()],
                             actual_real[test_idx.numpy()],
                             os.path.join(cfg.figures_dir, "map_snapshot.html"))


if __name__ == "__main__":
    main()
