"""The committed real-data slice (tests/data/) is what every component test runs on.
These tests check that it is intact and has the properties later tests rely on."""
import hashlib
import json

import numpy as np
import pytest

from conftest import RAW_DIR, SLICE_DIR
from make_slice import make_slice

N_COLS = 26                     # engine id, cycle, 3 operating settings, 21 sensors
WINDOW = 30
RUL_CAP = 125
# FD004's six (altitude, Mach, throttle) operating conditions; the first is FD001's only one.
CONDITIONS = np.array([[0, 0.00, 100], [10, 0.25, 100], [20, 0.70, 100],
                       [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]])
JITTER = np.array([0.02, 0.005, 0.0])   # real rows sit within this distance of a condition centre


def load(fd):
    train = np.loadtxt(SLICE_DIR / f"train_{fd}.txt")
    test = np.loadtxt(SLICE_DIR / f"test_{fd}.txt")
    rul = np.loadtxt(SLICE_DIR / f"RUL_{fd}.txt", ndmin=1)
    return train, test, rul


def lengths(arr):
    """{engine id: number of rows}"""
    ids, counts = np.unique(arr[:, 0].astype(int), return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def test_slice_files_are_unchanged():
    meta = json.loads((SLICE_DIR / "SLICE.json").read_text())
    for name, expected in meta["files"].items():
        actual = hashlib.sha256((SLICE_DIR / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} was modified; regenerate with tests/make_slice.py, never hand-edit"


@pytest.mark.parametrize("fd", ["FD004", "FD001"])
def test_real_column_layout(fd):
    train, test, rul = load(fd)
    assert train.shape[1] == N_COLS and test.shape[1] == N_COLS
    assert rul.ndim == 1


@pytest.mark.parametrize("fd", ["FD004", "FD001"])
def test_engine_ids_and_cycles_are_contiguous(fd):
    for arr in load(fd)[:2]:
        ids = arr[:, 0].astype(int)
        assert sorted(set(ids)) == list(range(1, ids.max() + 1))       # engines 1..N, renumbered
        for e in set(ids):
            cycles = arr[ids == e, 1].astype(int)
            assert cycles.tolist() == list(range(1, len(cycles) + 1))   # 1, 2, 3, ... no gaps


@pytest.mark.parametrize("fd", ["FD004", "FD001"])
def test_one_rul_row_per_test_engine(fd):
    _, test, rul = load(fd)
    assert len(rul) == len(lengths(test))
    assert (rul >= 1).all()


@pytest.mark.parametrize("fd", ["FD004", "FD001"])
def test_every_train_engine_lives_past_the_cap(fd):
    train, _, _ = load(fd)
    assert min(lengths(train).values()) > RUL_CAP      # so every engine has a capped "healthy" stretch


def test_fd004_has_test_engines_shorter_than_one_window():
    _, test, _ = load("FD004")
    assert sum(n < WINDOW for n in lengths(test).values()) >= 3     # needed to test left-padding


@pytest.mark.parametrize("fd, n_conditions", [("FD004", 6), ("FD001", 1)])
def test_operating_conditions(fd, n_conditions):
    train, _, _ = load(fd)
    settings = train[:, 2:5]
    nearest = np.abs(settings[:, None, :] - CONDITIONS[None]).sum(axis=2).argmin(axis=1)
    assert (np.abs(settings - CONDITIONS[nearest]) <= JITTER + 1e-9).all()   # every row near a centre
    assert len(set(nearest)) == n_conditions


@pytest.mark.skipif(not (RAW_DIR / "train_FD004.txt").exists(), reason="full NASA files not in data/raw (local only)")
def test_slice_regenerates_identically_from_raw(tmp_path):
    make_slice(RAW_DIR, tmp_path)
    meta = json.loads((SLICE_DIR / "SLICE.json").read_text())
    for name in meta["files"]:
        assert (tmp_path / name).read_bytes() == (SLICE_DIR / name).read_bytes(), name
