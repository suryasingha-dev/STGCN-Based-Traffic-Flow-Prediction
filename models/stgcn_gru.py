"""
models/stgcn_gru.py
=====================
TimeAwareSTGCN: a temporal-convolution + graph-convolution + GRU
hybrid for road-segment speed forecasting.

Architecture
------------
    Input (B, T, F)
        -> Conv1d over the time axis (local temporal smoothing)
        -> 2x GCNConv over the road-segment adjacency graph
           (spatial message passing between nearby segments)
        -> GRU over the time axis (sequence summarization)
        -> Dropout -> MLP head -> scalar speed prediction

This mirrors the family of Spatio-Temporal Graph Convolutional
Networks (STGCN) used in traffic forecasting literature (e.g. Yu et
al., 2018, "Spatio-Temporal Graph Convolutional Networks: A Deep
Learning Framework for Traffic Forecasting"), with a GRU substituted
for the second temporal block for a lighter parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TimeAwareSTGCN(nn.Module):
    """Spatio-temporal graph network for single-step speed forecasting.

    Parameters
    ----------
    input_dim : int
        Number of input features per (segment, timestep) pair, e.g.
        [speed, speed_lag1, speed_lag2, flow, time_of_day, weekday,
        weekend, day_type] -> 8.
    hidden_dim : int
        Width of the temporal conv, GCN layers, and GRU.
    output_dim : int
        Prediction dimensionality (1 for scalar speed).
    t_seq : int
        Number of timesteps per input sequence. Needed at construction
        time because the forward pass reshapes a flattened
        (segments * t_seq, features) batch back into
        (segments, t_seq, features) -- t_seq must match how the caller
        flattened the data in the first place.

    Notes
    -----
    GCNConv expects a single homogeneous node set per forward call. We
    flatten the (segment, timestep) batch to apply the graph
    convolution at every timestep using the *same* spatial adjacency
    (segments don't move), then reshape back for the GRU. This is a
    deliberate simplification: it assumes the road graph is static
    across the sequence window, which holds for fixed road-segment
    sensors but would need revisiting for a moving-sensor setup.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 t_seq: int):
        super().__init__()
        self.t_seq = t_seq
        self.temporal_conv = nn.Conv1d(input_dim, hidden_dim,
                                        kernel_size=3, padding=1)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (n_segments * t_seq, input_dim)
            Flattened (segment, timestep) feature matrix.
        edge_index : torch.Tensor, shape (2, n_edges)
            Road-segment adjacency in PyG edge_index format.

        Returns
        -------
        torch.Tensor, shape (n_segments, output_dim)
        """
        n_segments = x.shape[0] // self.t_seq
        x = x.view(n_segments, self.t_seq, -1)
        x = x.transpose(1, 2)             # (B, F, T) for Conv1d
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)             # (B, T, hidden)
        x = x.reshape(-1, x.shape[-1])    # (B*T, hidden) for GCNConv

        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        x = x.view(n_segments, self.t_seq, -1)
        _, h_n = self.gru(x)
        h_n = self.dropout(h_n[-1])
        return self.fc(h_n)
