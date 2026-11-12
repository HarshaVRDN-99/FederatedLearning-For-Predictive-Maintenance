"""One client, one round of federated training (DECISIONS F2, F4, LA5).

    receive global weights g  ->  train E epochs on this client's own windows  ->  send back new weights + count

Same code for all three arms:
    FedAvg   mu = 0,  no sample_weight
    FedProx  mu > 0:  loss += (mu/2) * sum over weights of (p - g)^2   (pulls p back toward g)
    FedLA    mu = 0,  sample_weight = per-window weights from fedla_weights
"""
from __future__ import annotations

import numpy as np
import torch

from rulfl.model.gru import RULModel


def client_seed(run_seed: int, rnd: int, client: int) -> int:
    """A seed for one client in one round, derived only from (run seed, round, client id), so it is the same in
    the plain simulator and in Flower, where clients run in separate workers."""
    return int(np.random.SeedSequence([run_seed, rnd, client]).generate_state(1)[0])


def client_update(model: RULModel, global_weights: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
                  mu: float, seed: int, sample_weight: np.ndarray | None = None) -> tuple[dict, int]:
    """Train `model` from the global weights on this client's windows (X, y in cycles) for cfg['fl']['E'] epochs.
    Returns (new weights as independent copies, number of training windows). `global_weights` is never changed.
    Fresh Adam every call: no optimiser state is carried between rounds (F2)."""
    f, cap = cfg["fl"], cfg["data"]["rul_cap"]
    g = {k: v.detach().clone() for k, v in global_weights.items()}   # snapshot BEFORE training (F4)
    model.load_state_dict(g)                                          # start from the global model
    model.train()
    torch.use_deterministic_algorithms(True)                          # M5; Flower workers are fresh processes
    torch.manual_seed(seed)                                         # e.g. dropout masks (2-layer GRU)
    rng = np.random.default_rng(seed)                                 # batch order
    opt = torch.optim.Adam(model.parameters(), lr=f["lr"])

    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y) / cap        # train on RUL / 125 (M3)
    w_t = None if sample_weight is None else torch.from_numpy(sample_weight).float()   # float32 like FedAvg's loss
    for _ in range(f["E"]):
        order = torch.from_numpy(rng.permutation(len(X)))
        for i in range(0, len(X), f["batch"]):
            idx = order[i:i + f["batch"]]
            err2 = (model(X_t[idx]) - y_t[idx]) ** 2                  # squared error per window
            loss = err2.mean() if w_t is None else (err2 * w_t[idx]).mean()
            if mu > 0:                                                # FedProx term (F4); skipped for FedAvg/FedLA
                loss = loss + (mu / 2) * sum(((p - g[name]) ** 2).sum() for name, p in model.named_parameters())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), f["clip"])
            opt.step()

    return {k: v.detach().clone() for k, v in model.state_dict().items()}, len(X)
