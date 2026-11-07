"""What the fake C-MAPSS data must look like. If these pass, every later test can trust the fakes."""
import numpy as np
import pytest

from fakes import CONDITIONS, CONSTANT_SENSORS, TRENDING_SENSORS, make_fake_cmapss

N_COLS = 26          # engine id, cycle, 3 operating settings, 21 sensors
WINDOW = 30


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    info = make_fake_cmapss(out, n_train=20, n_test=20, min_life=60, max_life=150, n_short_test=2, seed=0)
    train = np.loadtxt(out / "train_FD004.txt")
    test = np.loadtxt(out / "test_FD004.txt")
    rul = np.loadtxt(out / "RUL_FD004.txt", ndmin=1)
    return {"out": out, "info": info, "train": train, "test": test, "rul": rul}


def _lengths(arr):
    """{engine id: number of rows} for a loaded file."""
    ids, counts = np.unique(arr[:, 0].astype(int), return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def test_files_have_real_column_layout(fake):
    assert fake["train"].shape[1] == N_COLS
    assert fake["test"].shape[1] == N_COLS
    assert fake["rul"].ndim == 1


def test_engine_ids_and_cycles_are_contiguous(fake):
    for arr in (fake["train"], fake["test"]):
        ids = arr[:, 0].astype(int)
        assert sorted(set(ids)) == list(range(1, ids.max() + 1))      # engines 1..N, none missing
        for e in set(ids):
            cycles = arr[ids == e, 1].astype(int)
            assert cycles.tolist() == list(range(1, len(cycles) + 1))  # cycles 1, 2, 3, ... no gaps


def test_train_lifetimes_match_returned_truth(fake):
    info = fake["info"]
    assert _lengths(fake["train"]) == info.train_life
    assert all(60 <= life <= 150 for life in info.train_life.values())


def test_test_engines_are_cut_before_failure_and_rul_is_correct(fake):
    info = fake["info"]
    assert _lengths(fake["test"]) == info.test_len
    assert len(fake["rul"]) == len(info.test_len)                       # one RUL row per test engine
    for e, n_seen in info.test_len.items():
        assert n_seen < info.test_life[e]                               # cut before failure
        assert fake["rul"][e - 1] == info.test_life[e] - n_seen == info.test_rul[e]
        assert info.test_rul[e] >= 1


def test_some_test_engines_are_shorter_than_one_window(fake):
    short = [e for e, n in fake["info"].test_len.items() if n < WINDOW]
    assert len(short) == 2                                              # needed to test left-padding later


def test_six_real_operating_conditions(fake):
    settings = np.round(fake["train"][:, 2:5], 2)
    found = {tuple(row) for row in settings}
    assert found == {tuple(c) for c in CONDITIONS}


def test_sensors_trend_with_wear_and_constant_ones_do_not(fake):
    train = fake["train"]
    ids, cycles = train[:, 0].astype(int), train[:, 1]
    life = np.array([fake["info"].train_life[e] for e in ids])
    wear = cycles / life                                                # 0 = new, 1 = failed
    cond = np.array([CONDITIONS.index(tuple(r)) for r in np.round(train[:, 2:5], 2)])
    for s in TRENDING_SENSORS:
        x = train[:, 5 + s].copy()
        for c in range(len(CONDITIONS)):                                # remove each condition's own level
            x[cond == c] -= x[cond == c].mean()
        assert abs(np.corrcoef(x, wear)[0, 1]) > 0.5, f"sensor {s} does not trend with wear"
    for s in CONSTANT_SENSORS:
        x = train[:, 5 + s]
        for c in range(len(CONDITIONS)):
            assert np.ptp(x[cond == c]) == 0, f"sensor {s} varies within condition {c}"


def test_same_seed_same_files_different_seed_different_files(tmp_path):
    names = ["train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"]
    for d, seed in (("a", 7), ("b", 7), ("c", 8)):
        make_fake_cmapss(tmp_path / d, n_train=5, n_test=5, seed=seed)
    read = lambda d: [(tmp_path / d / n).read_bytes() for n in names]
    assert read("a") == read("b")
    assert read("a") != read("c")


def test_single_condition_mode_for_fd001(tmp_path):
    make_fake_cmapss(tmp_path, fd="FD001", n_conditions=1, n_train=5, n_test=5, seed=0)
    train = np.loadtxt(tmp_path / "train_FD001.txt")
    assert {tuple(r) for r in np.round(train[:, 2:5], 2)} == {tuple(CONDITIONS[0])}
