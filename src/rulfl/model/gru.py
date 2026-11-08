"""The RUL model: GRU -> last time step -> small MLP head -> one number (DECISIONS M1, M4).

    window (batch, 30 flights, n_features)
      -> self.gru   reads the 30 flights in order, keeps a running memory
      -> memory after the LAST flight (batch, hidden)
      -> self.head  Linear -> ReLU -> Linear
      -> one number per window = predicted RUL / 125

`gru` and `head` are separate submodules so the head-refit probe can freeze one and retrain the other.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RULModel(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,                              # input is (batch, flights, features), like our X
            dropout=dropout if layers > 1 else 0.0,        # PyTorch only applies dropout between GRU layers
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                 # out: (batch, flights, hidden) = memory after every flight
        last = out[:, -1]                    # memory after the last flight: (batch, hidden)
        return self.head(last).squeeze(-1)   # (batch,)  predicted RUL / 125


def predict_rul(model: nn.Module, X: np.ndarray, cap: int, batch: int = 1024) -> np.ndarray:
    """Predicted RUL in cycles, clamped to [0, cap] (M4). Used for every score, for every arm.
    Runs in eval mode without gradients, in batches to keep memory small; restores the model's mode."""
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(torch.from_numpy(X[i:i + batch]))
            preds.append((out * cap).clamp(0, cap).numpy())
    model.train(was_training)
    return np.concatenate(preds)
