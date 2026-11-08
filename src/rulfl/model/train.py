"""Centralized training: one model on all training windows (DECISIONS M3, M5).

Each epoch:
  1. shuffle the training windows (order fixed by the seed)
  2. for each batch of 256: MSE between model output and y/125, backprop, clip gradient size, Adam step
  3. validation MAE in cycles, with the same predict_rul every arm uses (M4)
  4. if it's the best so far, remember a copy of the weights
Stop when validation MAE hasn't improved for `patience` epochs (or at max_epochs);
return the model with the BEST epoch's weights, not the last.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from rulfl.data.pipeline import Dataset
from rulfl.model.gru import RULModel, predict_rul


def train_centralized(cfg: dict, ds: Dataset, seed: int) -> tuple[RULModel, dict]:
    """Returns (model at its best epoch, history). history = per-epoch 'train_loss' and 'val_mae'
    lists, plus 'best_epoch' = index into those lists of the epoch whose weights were returned.
    Reads cfg['model'] (hidden, layers, lr, dropout: chosen by tuning, M2) and cfg['train'] (M3)."""
    m, t, cap = cfg["model"], cfg["train"], cfg["data"]["rul_cap"]
    torch.use_deterministic_algorithms(True)        # M5: error out rather than silently be non-repeatable
    torch.manual_seed(seed)                         # same seed -> same starting weights
    rng = np.random.default_rng(seed)               # same seed -> same batch order every epoch

    model = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
    X = torch.from_numpy(ds.X_train)
    y = torch.from_numpy(ds.y_train) / cap          # train on RUL / 125 (M3)

    hist = {"train_loss": [], "val_mae": [], "best_epoch": 0}
    best_mae, best_state = float("inf"), None
    for epoch in range(t["max_epochs"]):
        model.train()
        order = torch.from_numpy(rng.permutation(len(X)))
        total = 0.0
        for i in range(0, len(X), t["batch"]):
            idx = order[i:i + t["batch"]]
            loss = torch.nn.functional.mse_loss(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])   # cap the update size (M3)
            opt.step()
            total += loss.item() * len(idx)
        hist["train_loss"].append(total / len(X))  # mean loss per window over the epoch

        val_mae = float(np.abs(predict_rul(model, ds.X_val, cap) - ds.y_val).mean())
        hist["val_mae"].append(val_mae)
        if val_mae < best_mae:                      # strictly better = improvement
            best_mae, best_state, hist["best_epoch"] = val_mae, copy.deepcopy(model.state_dict()), epoch
        elif epoch - hist["best_epoch"] >= t["patience"]:
            break                                   # `patience` epochs in a row without improvement

    model.load_state_dict(best_state)
    model.eval()
    return model, hist
