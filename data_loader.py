"""
data_loader.py
================
Data ingestion layer for the traffic forecasting pipeline.

Responsibilities
-----------------
1. Load road-segment coordinates (latitude/longitude) from a real,
   user-supplied CSV.
2. Attempt to pull live speed/flow observations for those coordinates
   from the TomTom Traffic Flow API.
3. If the API call is unavailable (no key configured, network error,
   or rate limit), fall back to a synthetic-but-structured generator
   so the rest of the pipeline can still be exercised end-to-end.

Known limitation (tracked in README "Limitations & Roadmap")
---------------------------------------------------------------
The TomTom ingestion path is implemented and called first, but has
not yet been validated against a live key in this repository's
history. Treat `fetch_live_traffic()` as a documented integration
point rather than a verified data source until you've confirmed it
against your own TomTom account. Every record returned by this module
carries a `source` field ("tomtom_api" or "synthetic_fallback") so
downstream code -- and anyone reviewing results -- can always tell
which path produced a given row.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/"
    "absolute/10/json"
)


@dataclass
class TrafficDataset:
    """Container for the arrays the modeling code consumes.

    Attributes
    ----------
    coords : np.ndarray, shape (n_segments, 2)
        Latitude/longitude for each road segment.
    speed_series : np.ndarray, shape (n_segments, t_seq)
        Speed (km/h) per segment per timestep.
    flow_series : np.ndarray, shape (n_segments, t_seq)
        Flow/volume proxy per segment per timestep.
    source : str
        Either "tomtom_api" or "synthetic_fallback". Always check this
        before reporting results -- it tells you whether the numbers
        downstream came from observed or generated data.
    """

    coords: np.ndarray
    speed_series: np.ndarray
    flow_series: np.ndarray
    source: str


def load_coordinates(csv_path: str, lat_col: str = "Latitude",
                      lon_col: str = "Longitude",
                      delimiter: str = "\t") -> np.ndarray:
    """Load road-segment coordinates from a real CSV/TSV file.

    Parameters
    ----------
    csv_path : str
        Path to the dataset. Tab-delimited by default to match the
        original data export this project was built against; pass
        delimiter="," if your file is comma-separated.
    lat_col, lon_col : str
        Column names for latitude/longitude.

    Returns
    -------
    np.ndarray of shape (n, 2)

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist. We raise rather than silently
        falling back here, because a missing coordinate file is a
        setup error the user should fix, not paper over.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Coordinate dataset not found at '{csv_path}'. "
            "Place your CSV there or pass --data-path to point at it."
        )

    df = pd.read_csv(csv_path, delimiter=delimiter)
    missing = [c for c in (lat_col, lon_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected column(s) {missing} not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    coords = df[[lat_col, lon_col]].dropna().values
    logger.info("Loaded %d coordinate rows from %s", len(coords), csv_path)
    return coords


def fetch_live_traffic(coords: np.ndarray, api_key: str | None,
                        t_seq: int = 10,
                        request_timeout: float = 5.0,
                        retry_pause: float = 0.2) -> TrafficDataset | None:
    """Attempt to pull live speed data from the TomTom Flow API.

    For each coordinate, queries TomTom's flowSegmentData endpoint and
    builds a (n_segments, t_seq) array by repeating the single live
    reading across the time axis (TomTom's free-tier flow endpoint
    returns a current snapshot, not a time series -- see README for
    how to extend this to true historical series via a paid tier or
    your own polling cache).

    Returns None (rather than raising) on any failure, so callers can
    cleanly fall back to synthetic data. This is intentional: a
    missing API key or a transient network blip should degrade the
    pipeline, not crash it.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    api_key : str or None
        Read from the TOMTOM_API_KEY environment variable by callers.
        Never hardcode this -- see .env.example.
    t_seq : int
        Number of timesteps the rest of the pipeline expects.

    Returns
    -------
    TrafficDataset or None
    """
    if not api_key:
        logger.warning(
            "No TomTom API key configured (set TOMTOM_API_KEY). "
            "Skipping live fetch, will use synthetic fallback."
        )
        return None

    speeds = np.full((len(coords), t_seq), np.nan)
    flows = np.full((len(coords), t_seq), np.nan)

    try:
        for i, (lat, lon) in enumerate(coords):
            params = {"key": api_key, "point": f"{lat},{lon}"}
            resp = requests.get(TOMTOM_FLOW_URL, params=params,
                                 timeout=request_timeout)
            resp.raise_for_status()
            payload = resp.json()
            segment = payload.get("flowSegmentData", {})
            speed = segment.get("currentSpeed")
            free_flow = segment.get("freeFlowSpeed")

            if speed is None:
                logger.debug("No currentSpeed for point %s,%s", lat, lon)
                return None

            speeds[i, :] = speed
            flows[i, :] = free_flow if free_flow is not None else speed
            time.sleep(retry_pause)  # stay well under rate limits

    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("TomTom API fetch failed (%s). Falling back to "
                        "synthetic data.", exc)
        return None

    if np.isnan(speeds).any():
        logger.warning("Incomplete TomTom response. Falling back to "
                        "synthetic data.")
        return None

    logger.info("Fetched live TomTom data for %d segments.", len(coords))
    return TrafficDataset(coords=coords, speed_series=speeds,
                           flow_series=flows, source="tomtom_api")


def generate_synthetic_traffic(coords: np.ndarray, t_seq: int = 10,
                                seed: int | None = None) -> TrafficDataset:
    """Generate structured synthetic speed/flow series as a fallback.

    Encodes a simple rush-hour pattern (sinusoidal dip in speed around
    peak hours) plus noise, rather than pure random values, so the
    downstream model has *some* learnable temporal signal to validate
    the architecture against. This is explicitly a stand-in for real
    sensor data -- see TrafficDataset.source == "synthetic_fallback".

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    t_seq : int
    seed : int, optional
        Set for reproducible synthetic runs (e.g. in CI / tests).
    """
    rng = np.random.default_rng(seed)
    n = len(coords)

    base = np.linspace(6, 22, t_seq) / 24.0
    rush = np.sin(base * np.pi * 2).reshape(1, t_seq)
    noise = rng.normal(0, 3, (n, t_seq))
    pattern = rush + noise / 15

    speed = 30 + 20 * pattern + rng.normal(0, 5, (n, t_seq))
    speed = np.clip(speed, 5, 80)
    flow = 100 + speed * 2 + rng.normal(0, 10, (n, t_seq))

    logger.info("Generated synthetic traffic series for %d segments "
                "(seed=%s). source=synthetic_fallback", n, seed)
    return TrafficDataset(coords=coords, speed_series=speed,
                           flow_series=flow, source="synthetic_fallback")


def load_traffic_dataset(csv_path: str, t_seq: int = 10,
                          api_key: str | None = None,
                          seed: int | None = None) -> TrafficDataset:
    """Top-level entry point: real coords -> live API -> synthetic fallback.

    This is the function train.py calls. It always returns a populated
    TrafficDataset; check `.source` to know which path produced it.
    """
    coords = load_coordinates(csv_path)

    api_key = api_key or os.environ.get("TOMTOM_API_KEY")
    live = fetch_live_traffic(coords, api_key=api_key, t_seq=t_seq)
    if live is not None:
        return live

    return generate_synthetic_traffic(coords, t_seq=t_seq, seed=seed)
