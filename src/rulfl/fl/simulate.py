"""Plain-PyTorch federated simulator: the reference engine that Flower must reproduce (DECISIONS F1-F8, Gate 2).

One round:  server sends global weights  ->  every client runs client_update  ->  server averages with fedavg
"""
from __future__ import annotations

import numpy as np
import torch

from rulfl.data.pipeline import Dataset
from rulfl.fl.client import client_seed, client_update
from rulfl.fl.fedla import fedla_weights, label_histogram
from rulfl.model.gru import RULModel, predict_rul


def fedavg(updates: list[tuple[dict, int]]) -> dict:
    """Average client weights, each weighted by its number of training windows (F3).
    updates = [(weights, n_windows), ...].  Example: 1000 windows x 2.0 and 3000 x 4.0 -> 3.5."""
    total = sum(n for _, n in updates)
    first = updates[0][0]
    return {k: sum(w[k] * (n / total) for w, n in updates) for k in first}


def client_rows(engine: np.ndarray, assignment: dict, K: int) -> list[np.ndarray]:
    """Which rows (windows) belong to each client. assignment = engine id -> client, keys int or str
    (json.load turns int keys into strings).  Example: engine [3, 3, 8], {3: 0, 8: 1}, K=2 -> [[0, 1], [2]]."""
    client_of = {int(e): int(c) for e, c in assignment.items()}
    owner = np.array([client_of[int(e)] for e in engine])
    return [np.flatnonzero(owner == k) for k in range(K)]


def mae(model: RULModel, X: np.ndarray, y: np.ndarray, cap: int) -> float:
    """Mean absolute error in cycles; nan if there are no windows (a client with no validation engines)."""
    return float(np.abs(predict_rul(model, X, cap) - y).mean()) if len(X) else float("nan")


def run_federated(cfg: dict, ds: Dataset, manifest: dict, algo: str, seed: int) -> dict:
    """Federated training: R rounds, K clients, all clients every round (F1), all R rounds always run (F6).
    manifest = {'train': {engine: client}, 'val': {engine: client}} (PT5, PT7).  algo = 'fedavg' | 'fedprox' | 'fedla'.
    Returns per-round lists 'val_mae' (all validation windows), 'client_val_mae' (K values: each client's routed
    validation engines), 'bytes' (F8), plus 'best_round' (counted from 1) and 'best_state' (its global weights)."""
    m, f, cap = cfg["model"], cfg["fl"], cfg["data"]["rul_cap"]
    K, R = f["K"], f["R"]
    if algo in ("fedavg", "fedla"):
        mu = 0.0                                              # FedLA = FedAvg + per-window weights (LA5)
    elif algo == "fedprox":
        mu = f["mu"]                                          # chosen once at H = 1 on validation (F5)
    else:
        raise ValueError(f"unknown algo {algo!r}; expected 'fedavg', 'fedprox' or 'fedla'")

    torch.use_deterministic_algorithms(True)                  # as train_centralized (M5)
    torch.manual_seed(seed)                                   # same start weights as train_centralized(seed)
    model = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    global_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n_params = sum(v.numel() for v in global_w.values())

    train_rows = client_rows(ds.engine_train, manifest["train"], K)
    val_rows = client_rows(ds.engine_val, manifest["val"], K)

    weights, hist_bytes = [None] * K, 0                       # FedAvg / FedProx: no per-window weights
    if algo == "fedla":                                       # once, before round 1 (LA4)
        edges, clip = cfg["fedla"]["bin_edges"], cfg["fedla"]["clip"]
        local = [label_histogram(ds.y_train[r], edges, cap) for r in train_rows]   # each client counts (up)
        glob = sum(local)                                                           # server adds them up
        weights = [fedla_weights(ds.y_train[r], h, glob, edges, cap, clip)         # global sent back (down)
                   for r, h in zip(train_rows, local)]
        hist_bytes = 2 * K * len(edges) * 4                   # K x 6 counts up + K x 6 down, 4 bytes each (LA6)

    log = {"val_mae": [], "client_val_mae": [], "bytes": [], "best_round": 0, "best_state": None}
    for rnd in range(1, R + 1):                               # rounds count from 1, like Flower's server_round
        updates = [client_update(model, global_w, ds.X_train[r], ds.y_train[r], cfg,
                                 mu=mu, seed=client_seed(seed, rnd, k), sample_weight=weights[k])
                   for k, r in enumerate(train_rows)]
        global_w = fedavg(updates)

        model.load_state_dict(global_w)                       # score the new global model on validation only
        log["val_mae"].append(mae(model, ds.X_val, ds.y_val, cap))
        log["client_val_mae"].append([mae(model, ds.X_val[r], ds.y_val[r], cap) for r in val_rows])
        log["bytes"].append(2 * K * n_params * 4 + (hist_bytes if rnd == 1 else 0))   # down + up, float32 (F8)
        if log["val_mae"][-1] < min(log["val_mae"][:-1], default=float("inf")):   # strictly better
            log["best_round"], log["best_state"] = rnd, {k: v.clone() for k, v in global_w.items()}
    return log
