"""
models/architectures.py
=========================
Five lightweight reference architectures used as a comparison
benchmark against TimeAwareSTGCN (see stgcn_gru.py).

These names echo well-known traffic-forecasting model families from
the literature, but the implementations here are compact, single-file
reference versions for benchmarking purposes -- not the full
published architectures. Specifically:

- ST-GCN   : Conv1d + BatchNorm temporal feature extractor (no graph
             convolution in this reference version -- compare against
             stgcn_gru.TimeAwareSTGCN for the graph-aware version).
- DCRNN    : single-layer GRU, standing in for the full diffusion
             convolutional recurrent network.
- Bi-GRCN  : bidirectional GRU.
- SA-DGWN  : single-head self-attention over the time axis, standing
             in for a dynamic graph wavelet network.
- EG-NODE  : plain MLP, standing in for an explicit-graph neural ODE.

The point of this module is a controlled comparison of "how much does
architectural complexity buy you on this task," not a reproduction of
each paper's full method. See README "Benchmark Methodology" for the
honest version of this caveat.
"""

import torch
import torch.nn as nn


class STGCN(nn.Module):
    """Temporal-conv reference block (no graph convolution)."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels,
                                kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2,
                                kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class DCRNN(nn.Module):
    """Single-layer GRU reference block."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])


class BiGRCN(nn.Module):
    """Bidirectional GRU reference block."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, batch_first=True,
                           bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])


class SADGWN(nn.Module):
    """Single-head self-attention reference block."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1,
                                           batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.fc(attn_out[:, -1, :])


class EGNODE(nn.Module):
    """Plain MLP reference block (flattens the time axis)."""

    def __init__(self, n_features: int = 4, hidden_size: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


def build_benchmark_models(n_features: int = 4) -> dict[str, nn.Module]:
    """Factory returning a fresh instance of every benchmark model."""
    return {
        "ST-GCN": STGCN(),
        "DCRNN": DCRNN(),
        "Bi-GRCN": BiGRCN(),
        "SA-DGWN": SADGWN(),
        "EG-NODE": EGNODE(n_features=n_features),
    }
