"""FedLA: label-aware per-window loss weights (DECISIONS LA1-LA7).

Each client reweights its training windows so its label distribution matches the global one:
weight of a window = p_global(bin) / p_local(bin), clipped, then rescaled to mean 1 on the client.
Bins: [e0,e1) ... [e4,e5) plus the capped bin {cap}; with the config edges that is 6 bins (LA1).
Weights are computed once before round 1 (LA4) and passed to client_update as sample_weight (LA5).
"""
from __future__ import annotations

import numpy as np


def _bin_index(y: np.ndarray, edges: list[int], cap: int) -> np.ndarray:
    """Bin of every label: 0..len(edges)-2 for the half-open intervals, len(edges)-1 for the capped bin."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges[0] != 0 or edges[-1] != cap or np.any(np.diff(edges) <= 0):
        raise ValueError(f"edges must rise strictly from 0 to cap={cap}, got {edges.tolist()}")
    y = np.asarray(y, dtype=np.float64)
    if y.size and (y.min() < 0 or y.max() > cap):
        raise ValueError(f"labels must lie in [0, {cap}], got [{y.min()}, {y.max()}]")
    # side='right' puts a label equal to an edge into the bin starting at that edge;
    # a label equal to cap lands one past the last interval, which is the capped bin.
    return np.searchsorted(edges, y, side="right") - 1


def label_histogram(y: np.ndarray, edges: list[int], cap: int) -> np.ndarray:
    """Window counts per bin (LA3): [0,25) [25,50) [50,75) [75,100) [100,125) {125} with the config edges."""
    idx = _bin_index(y, edges, cap)
    return np.bincount(idx, minlength=len(edges)).astype(np.int64)


def fedla_weights(y: np.ndarray, local_counts, global_counts, edges: list[int], cap: int, clip) -> np.ndarray:
    """One weight per window of this client, mean 1 (LA2).

    local_counts must be label_histogram(y, edges, cap); global_counts is the sum of every client's histogram.
    """
    local = np.asarray(local_counts, dtype=np.float64)
    glob = np.asarray(global_counts, dtype=np.float64)
    n_bins = len(edges)
    if local.shape != (n_bins,) or glob.shape != (n_bins,):
        raise ValueError(f"histograms must have {n_bins} bins, got {local.shape} and {glob.shape}")
    if not np.array_equal(local, label_histogram(y, edges, cap)):
        raise ValueError("local_counts does not match the histogram of y")
    if np.any(glob < local):
        raise ValueError("global_counts must include this client's counts (global >= local in every bin)")
    if len(y) == 0:
        return np.zeros(0, dtype=np.float64)

    p_local = local / local.sum()
    p_global = glob / glob.sum()
    # Bins the client has no windows in get no weight; they are never indexed below.
    w_bin = np.divide(p_global, p_local, out=np.zeros(n_bins), where=local > 0)
    lo, hi = clip
    w_bin = np.where(local > 0, np.clip(w_bin, lo, hi), 0.0)

    w = w_bin[_bin_index(y, edges, cap)]
    return w / w.mean()
