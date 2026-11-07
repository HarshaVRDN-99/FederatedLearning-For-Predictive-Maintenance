"""Data pipeline: NASA C-MAPSS text files -> model-ready windows. Decisions D1-D8 in docs/DECISIONS.md.

Built step by step:
  1. load_cmapss       read a NASA file into a table with named columns
  2. add_train_labels  RUL label for every training row: min(cycles left, cap)
  3. split_engines     fixed engine-level train/validation split (D1)
  4. add_condition     which of the operating conditions each flight was in (D2)
  5. fit_scaler        learn, from TRAINING rows only, which sensors to keep and each sensor's
                       mean/std per condition (D3, D4)
  6. apply_scaler      turn any rows (train, val or test) into model features: normalised kept
                       sensors + condition one-hot (D5)
  7. make_windows      every 30-flight window of an engine, labelled with its last flight's RUL (D6)
  8. last_windows      the last 30 flights of each test engine, short ones padded (D7)
  9. build_dataset     all of the above, in order: NASA files in, model-ready arrays out
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["altitude", "mach", "throttle"]          # the 3 operational settings
SENSORS = [f"s{i}" for i in range(1, 22)]            # sensor 1..21, NASA's own numbering
COLUMNS = ["engine", "cycle"] + SETTINGS + SENSORS   # the 26 columns of every NASA file


def load_cmapss(path: str | Path) -> pd.DataFrame:
    """Read one NASA file (train_FD00x.txt or test_FD00x.txt). One row = one engine on one flight (cycle)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)   # values are separated by spaces
    return df.astype({"engine": int, "cycle": int})


def add_train_labels(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add column 'rul' = min(T - t, cap), where T is the engine's last cycle (it failed after T)
    and t is the current cycle. Only valid for training files, where every engine runs to failure."""
    last_cycle = df.groupby("engine")["cycle"].transform("max")      # T, repeated on every row of that engine
    return df.assign(rul=np.minimum(last_cycle - df["cycle"], cap))


def split_engines(engine_ids, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split ENGINES (not rows) into train and validation (D1). Done before any windowing or scaling,
    so no engine can ever appear on both sides. Same ids + same seed -> same split, whatever the input order."""
    ids = sorted(int(e) for e in set(engine_ids))
    n_val = round(val_frac * len(ids))
    rng = np.random.default_rng(seed)
    val = sorted(rng.choice(ids, size=n_val, replace=False).tolist())
    train = [e for e in ids if e not in set(val)]
    return train, val


def add_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'cond': the flight's operating condition, from its 3 settings rounded to whole numbers
    (altitude, Mach x 100, throttle), e.g. '42_84_100' (D2). Real jitter is <= 0.008, so rounding is exact.
    Whole numbers avoid a '-0' vs '0' split when tiny jitter around zero rounds negative."""
    def whole(x: pd.Series) -> pd.Series:
        return x.round().astype(int).astype(str)
    return df.assign(cond=whole(df["altitude"]) + "_" + whole(df["mach"] * 100) + "_" + whole(df["throttle"]))


@dataclass
class Scaler:
    """Everything learned from training rows that is needed to turn any rows into model features."""
    conditions: list[str]                    # operating conditions seen in training, sorted
    kept: list[str]                          # sensors used as features
    dropped: list[str]                       # sensors constant in every condition (D4)
    mean: dict[str, dict[str, float]]        # mean[condition][sensor]
    std: dict[str, dict[str, float]]         # std[condition][sensor]; 0.0 = constant in that condition

    @property
    def feature_names(self) -> list[str]:
        return self.kept + [f"cond_{c}" for c in self.conditions]


def fit_scaler(train: pd.DataFrame, tol: float) -> Scaler:
    """Learn the scaler from TRAINING rows only (never validation or test: no leakage).
    A sensor is dropped if its std is <= tol in every condition (D4)."""
    grouped = train.groupby("cond")[SENSORS]
    means, stds = grouped.mean(), grouped.std(ddof=0)
    stds = stds.where(stds > tol, 0.0)                        # treat numerically-constant as exactly constant
    dropped = [s for s in SENSORS if (stds[s] == 0).all()]
    kept = [s for s in SENSORS if s not in dropped]
    conditions = sorted(means.index)
    return Scaler(
        conditions=conditions,
        kept=kept,
        dropped=dropped,
        mean={c: {s: float(means.at[c, s]) for s in kept} for c in conditions},
        std={c: {s: float(stds.at[c, s]) for s in kept} for c in conditions},
    )


def apply_scaler(df: pd.DataFrame, scaler: Scaler) -> pd.DataFrame:
    """Features for any rows: each kept sensor z-scored with ITS condition's training mean/std (D3),
    plus a one-hot column per condition (D5). Where a sensor was constant in a condition during
    training, its value there is set to 0 ("as usual"), which also avoids dividing by zero."""
    unseen = set(df["cond"]) - set(scaler.conditions)
    if unseen:
        raise ValueError(f"operating condition(s) not seen in training: {sorted(unseen)}")
    mean = pd.DataFrame(scaler.mean).T.loc[df["cond"], scaler.kept].to_numpy()   # each row's own condition stats
    std = pd.DataFrame(scaler.std).T.loc[df["cond"], scaler.kept].to_numpy()
    raw = df[scaler.kept].to_numpy(dtype=float)
    z = np.divide(raw - mean, std, out=np.zeros_like(raw), where=std > 0)
    out = pd.DataFrame(z, columns=scaler.kept, index=df.index)
    for c in scaler.conditions:
        out[f"cond_{c}"] = (df["cond"] == c).astype(float).to_numpy()
    return out


def make_windows(features: np.ndarray, engine: np.ndarray, rul: np.ndarray, window: int):
    """Cut every engine into overlapping windows of `window` consecutive flights (stride 1).
    Rows must be grouped by engine and in cycle order. Label of a window = RUL at its last flight (D6).
    A window never spans two engines; an engine shorter than `window` gives no windows.
    Returns X (n, window, n_features), y (n,), engine id per window (n,)."""
    Xs, ys, es = [], [], []
    for e in np.unique(engine):
        rows = engine == e
        f, r = features[rows], rul[rows]
        if len(f) < window:
            continue
        w = np.lib.stride_tricks.sliding_window_view(f, (window, f.shape[1]))[:, 0]   # (n_windows, window, n_features)
        Xs.append(w)
        ys.append(r[window - 1:])
        es.append(np.full(len(w), e))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(es)


def last_windows(features: np.ndarray, engine: np.ndarray, window: int):
    """One window per engine: its last `window` flights. An engine with fewer flights is padded at the
    front by repeating its first flight (D7). Returns X (n_engines, window, n_features), engine ids (sorted)."""
    Xs, ids = [], np.unique(engine)
    for e in ids:
        f = features[engine == e]
        if len(f) < window:
            f = np.vstack([np.repeat(f[:1], window - len(f), axis=0), f])
        Xs.append(f[-window:])
    return np.stack(Xs), ids


@dataclass
class Dataset:
    """Model-ready data. One row of X = one window of 30 flights x n_features."""
    X_train: np.ndarray            # (n, 30, f) float32
    y_train: np.ndarray            # (n,) RUL label, capped
    engine_train: np.ndarray       # (n,) which engine each window came from
    X_val: np.ndarray
    y_val: np.ndarray
    engine_val: np.ndarray
    X_test: np.ndarray             # (n_test_engines, 30, f): last window per test engine
    y_test: np.ndarray             # true RUL, capped at rul_cap (headline, D8)
    y_test_uncapped: np.ndarray    # true RUL as NASA gives it
    feature_names: list[str]
    train_engine_lifetimes: dict[int, int]   # engine id -> total flights (for the partitioner)
    val_engine_lifetimes: dict[int, int]     # same for validation engines (routing, PT5)
    scaler: Scaler                           # how features were made (reused by every later step)


def build_dataset(cfg: dict) -> Dataset:
    """NASA files in, model-ready arrays out. Order matters for leakage:
    split engines FIRST, learn the scaler from training engines ONLY, then apply it to everything."""
    d = cfg["data"]
    raw, fd, cap, window = Path(d["raw_dir"]), d["fd"], d["rul_cap"], d["window"]

    train_all = add_condition(add_train_labels(load_cmapss(raw / f"train_{fd}.txt"), cap))
    train_all = train_all.sort_values(["engine", "cycle"], ignore_index=True)
    test = add_condition(load_cmapss(raw / f"test_{fd}.txt")).sort_values(["engine", "cycle"], ignore_index=True)
    rul_true = np.loadtxt(raw / f"RUL_{fd}.txt", ndmin=1)        # line i = true RUL of test engine i

    train_ids, val_ids = split_engines(train_all["engine"].unique(), d["val_frac"], d["split_seed"])   # D1
    tr = train_all[train_all["engine"].isin(train_ids)]
    va = train_all[train_all["engine"].isin(val_ids)]

    scaler = fit_scaler(tr, d["const_std_tol"])                   # D3, D4: training engines only

    def windows_of(part: pd.DataFrame):
        feats = apply_scaler(part, scaler).to_numpy(dtype=np.float32)
        X, y, e = make_windows(feats, part["engine"].to_numpy(), part["rul"].to_numpy(), window)
        return X, y.astype(np.float32), e

    X_train, y_train, e_train = windows_of(tr)
    X_val, y_val, e_val = windows_of(va)
    X_test, e_test = last_windows(apply_scaler(test, scaler).to_numpy(dtype=np.float32), test["engine"].to_numpy(), window)

    y_test_uncapped = rul_true[e_test - 1].astype(np.float32)
    y_test = np.minimum(y_test_uncapped, cap) if d["cap_test_labels"] else y_test_uncapped

    return Dataset(
        X_train=X_train, y_train=y_train, engine_train=e_train,
        X_val=X_val, y_val=y_val, engine_val=e_val,
        X_test=X_test, y_test=y_test, y_test_uncapped=y_test_uncapped,
        feature_names=scaler.feature_names,
        train_engine_lifetimes={int(e): int(T) for e, T in tr.groupby("engine")["cycle"].max().items()},
        val_engine_lifetimes={int(e): int(T) for e, T in va.groupby("engine")["cycle"].max().items()},
        scaler=scaler,
    )
