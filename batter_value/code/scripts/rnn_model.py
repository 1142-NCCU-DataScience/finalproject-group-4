"""PyTorch LSTM regressor for 3-season batter trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn


class BatterLSTM(nn.Module):
    """
    LSTM reads 3 timesteps (S-3, S-2, S-1) of process + wRC+ features,
    then concatenates age and position one-hot for the final prediction.
    """

    def __init__(
        self,
        seq_features: int = 6,
        hidden_size: int = 32,
        num_layers: int = 1,
        pos_dim: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1 + pos_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_age: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]
        feats = torch.cat([last, x_age, x_pos], dim=1)
        return self.head(feats).squeeze(-1)
