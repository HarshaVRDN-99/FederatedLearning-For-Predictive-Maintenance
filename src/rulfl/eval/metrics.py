
"""M6: Evaluation metrics for RUL prediction."""

from __future__ import annotations

import numpy as np


def _validate_inputs(pred, true):
    """Validate prediction and ground-truth arrays."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError("pred and true must be 1-D arrays")

    if pred.size == 0 or true.size == 0:
        raise ValueError("pred and true must not be empty")

    if len(pred) != len(true):
        raise ValueError("pred and true must have equal lengths")

    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
        raise ValueError("pred and true must contain only finite values")

    return pred, true


def mae(pred, true) -> float:
    pred, true = _validate_inputs(pred, true)
    return float(np.mean(np.abs(pred - true)))


def rmse(pred, true) -> float:
    pred, true = _validate_inputs(pred, true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r2(pred, true) -> float:
    pred, true = _validate_inputs(pred, true)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)

    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate the asymmetric NASA PHM08 score (sum over engines)."""
    pred, true = _validate_inputs(pred, true)
    d = pred - true

    costs = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0
    )

    return float(np.sum(costs))


def metrics(pred, true) -> dict:
    """Return MAE, RMSE and NASA score as Python floats."""
    pred, true = _validate_inputs(pred, true)
    d = pred - true

    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "nasa": nasa_score(pred, true),
    }


def all_metrics(pred, true) -> dict:
    """Preserve the existing project metrics interface."""
    return {
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "r2": r2(pred, true),
        "nasa": nasa_score(pred, true),
    }


def life_stage_mae(pred, true, edges=(50, 125)) -> dict:
    """Calculate MAE for near-failure, mid-life and healthy engines."""
    pred, true = _validate_inputs(pred, true)

    if len(edges) != 2 or not edges[0] < edges[1]:
        raise ValueError("edges must contain two strictly increasing values")

    near_failure = true < edges[0]
    mid = (true >= edges[0]) & (true < edges[1])
    healthy = true >= edges[1]

    def calculate_mae(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(pred[mask] - true[mask])))

    return {
        "near_failure": calculate_mae(near_failure),
        "mid": calculate_mae(mid),
        "healthy": calculate_mae(healthy),
        "n_near_failure": int(np.sum(near_failure)),
        "n_mid": int(np.sum(mid)),
        "n_healthy": int(np.sum(healthy)),
    }


def rounds_to_threshold(val_curve, threshold_value: float):
    """Return the first 1-based round reaching the threshold.

    Returns (None, False) if the threshold is never reached.
    """
    for i, v in enumerate(val_curve, start=1):
        if v is not None and v <= threshold_value:
            return i, True

    return None, False
