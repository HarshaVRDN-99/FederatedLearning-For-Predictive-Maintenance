"""Toy client for testing the Flower wiring before client_update exists (step A only).

A numpy linear regression with unequal client sizes. Each client regenerates its own data from its id, so the
client is stateless (F2) and Ray workers need nothing shipped to them. Not used in any registered run.
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 3
BASE_N = 20          # client k holds BASE_N * (k + 1) samples, so client sizes differ (tests F3 weighting)


def toy_data(client_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + client_id)
    X = rng.normal(size=(BASE_N * (client_id + 1), N_FEATURES))
    y = X @ np.arange(1.0, N_FEATURES + 1) + 0.5 * client_id + rng.normal(scale=0.1, size=len(X))
    return X, y


def toy_initial() -> list[np.ndarray]:
    return [np.zeros(N_FEATURES), np.zeros(1)]


def toy_train(arrays: list[np.ndarray], client_id: int, config: dict) -> tuple[list[np.ndarray], int]:
    """A few full-batch gradient steps on MSE from the received global weights."""
    w, b = (np.array(a, dtype=np.float64) for a in arrays)
    X, y = toy_data(client_id)
    for _ in range(int(config["steps"])):
        err = X @ w + b[0] - y
        w = w - config["lr"] * 2 * X.T @ err / len(y)
        b = b - config["lr"] * 2 * err.mean()
    return [w, b], len(y)


def toy_eval(arrays: list[np.ndarray], K: int) -> dict[str, float]:
    """Global MSE over every client's data (stands in for global validation MAE, F6)."""
    w, b = arrays
    errs = [X @ w + b[0] - y for X, y in (toy_data(k) for k in range(K))]
    return {"mse": float(np.mean(np.concatenate(errs) ** 2))}
