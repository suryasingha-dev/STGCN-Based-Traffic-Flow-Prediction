"""
models package
================
- stgcn_gru.TimeAwareSTGCN : the primary forecasting model (train.py)
- architectures.{STGCN, DCRNN, BiGRCN, SADGWN, EGNODE} : comparison
  baselines used by benchmark.py
"""

from .architectures import (
    BiGRCN,
    DCRNN,
    EGNODE,
    SADGWN,
    STGCN,
    build_benchmark_models,
)
from .stgcn_gru import TimeAwareSTGCN

__all__ = [
    "TimeAwareSTGCN",
    "STGCN",
    "DCRNN",
    "BiGRCN",
    "SADGWN",
    "EGNODE",
    "build_benchmark_models",
]
