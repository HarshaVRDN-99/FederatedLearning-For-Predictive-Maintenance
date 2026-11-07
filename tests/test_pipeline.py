"""Data pipeline tests, on the committed real-data slice (tests/data/)."""
import numpy as np
import pytest

from rulfl.data.pipeline import (COLUMNS, add_condition, add_train_labels, apply_scaler, fit_scaler,
                                 load_cmapss, split_engines)


# ---------- load ----------

def test_load_reads_real_layout(slice_dir):
    df = load_cmapss(slice_dir / "train_FD004.txt")
    assert list(df.columns) == COLUMNS                      # 26 named columns
    assert len(df) == 6660 and df["engine"].nunique() == 25
    assert df["engine"].dtype.kind == "i" and df["cycle"].dtype.kind == "i"


# ---------- labels ----------

def test_labels_count_down_to_zero_and_are_capped(slice_dir):
    df = add_train_labels(load_cmapss(slice_dir / "train_FD004.txt"), cap=125)
    e1 = df[df["engine"] == 1].set_index("cycle")["rul"]   # engine 1 lived 321 cycles
    assert e1[321] == 0                                     # last flight: fails after it
    assert e1[300] == 21                                    # 321 - 300 cycles left
    assert e1[196] == 125 and e1[197] == 124                # 321 - 196 = 125: cap starts here
    assert e1[1] == 125                                     # early life: capped "healthy"
    assert df["rul"].between(0, 125).all()


# ---------- split (first required test: no engine in two splits) ----------

def test_split_puts_every_engine_in_exactly_one_set():
    ids = list(range(1, 250))                               # FD004 has 249 training engines
    train, val = split_engines(ids, val_frac=0.2, seed=0)
    assert set(train) & set(val) == set()                   # NO engine in both
    assert sorted(train + val) == ids                       # and none lost
    assert len(val) == 50                                   # 20% of 249, rounded


def test_split_is_fixed_by_seed():
    ids = list(range(1, 250))
    assert split_engines(ids, 0.2, seed=0) == split_engines(ids, 0.2, seed=0)
    assert split_engines(ids, 0.2, seed=0) != split_engines(ids, 0.2, seed=1)
    # order of the input must not matter, only which engines exist
    assert split_engines(ids[::-1], 0.2, seed=0) == split_engines(ids, 0.2, seed=0)


# ---------- operating conditions (D2) ----------

@pytest.mark.parametrize("fd, n_conditions", [("FD004", 6), ("FD001", 1)])
def test_condition_count(slice_dir, fd, n_conditions):
    df = add_condition(load_cmapss(slice_dir / f"train_{fd}.txt"))
    assert df["cond"].nunique() == n_conditions          # FD001: jitter around 0 must not split into "0" and "-0"


# ---------- scaler: sensor selection (D4) + per-condition z-score (D3) + one-hot (D5) ----------

@pytest.fixture(scope="module")
def fd004(slice_dir):
    df = add_condition(load_cmapss(slice_dir / "train_FD004.txt"))
    return df, fit_scaler(df, tol=1e-8)


def test_drops_only_sensors_constant_in_every_condition(fd004):
    _, scaler = fd004
    assert scaler.dropped == ["s1", "s5", "s18", "s19"]
    assert len(scaler.kept) == 17


def test_normalised_sensors_have_mean_0_std_1_per_condition(fd004):
    df, scaler = fd004
    X = apply_scaler(df, scaler)
    for c in scaler.conditions:
        rows = X[(df["cond"] == c).to_numpy()]
        for s in scaler.kept:
            col = rows[s]
            if scaler.std[c][s] > 0:                       # sensor varies in this condition
                assert abs(col.mean()) < 1e-6 and abs(col.std(ddof=0) - 1) < 1e-6
            else:                                          # constant here (e.g. s16 in some conditions): becomes 0
                assert (col == 0).all()


def test_one_hot_marks_exactly_the_rows_condition(fd004):
    df, scaler = fd004
    X = apply_scaler(df, scaler)
    onehot = X[[f"cond_{c}" for c in scaler.conditions]].to_numpy()
    assert (onehot.sum(axis=1) == 1).all()                 # exactly one condition switched on per row
    assert (np.array(scaler.conditions)[onehot.argmax(axis=1)] == df["cond"].to_numpy()).all()
    assert list(X.columns) == scaler.feature_names == scaler.kept + [f"cond_{c}" for c in scaler.conditions]


def test_scaler_is_learned_only_from_the_rows_it_is_given(fd004):
    """Leakage guard: rows NOT passed to fit_scaler (e.g. validation engines) can't change it."""
    df, _ = fd004
    given = df[df["engine"] <= 20]                         # pretend these are the training engines
    scaler = fit_scaler(given, tol=1e-8)
    for c in scaler.conditions:                            # stats must equal hand-computed stats of `given` only
        rows = given[given["cond"] == c]
        for s in scaler.kept:
            assert scaler.mean[c][s] == pytest.approx(rows[s].mean())
            assert scaler.std[c][s] == pytest.approx(rows[s].std(ddof=0) if rows[s].std(ddof=0) > 1e-8 else 0.0)
    before = (dict(scaler.mean), dict(scaler.std))
    apply_scaler(df[df["engine"] > 20], scaler)            # applying it to other engines must not change it
    assert (scaler.mean, scaler.std) == before


def test_unseen_condition_is_an_error(fd004):
    df, scaler = fd004
    odd = df.head(3).assign(cond="99_0_100")
    with pytest.raises(ValueError, match="condition"):
        apply_scaler(odd, scaler)


# ---------- windows (D6) and test padding (D7): tiny hand-made arrays ----------

from rulfl.data.pipeline import build_dataset, last_windows, make_windows   # noqa: E402


def test_make_windows_slides_one_step_and_labels_the_last_row():
    feats = np.arange(33 * 2, dtype=float).reshape(33, 2)        # one engine, 33 flights, 2 features
    rul = np.arange(32, -1, -1)                                   # 32, 31, ..., 0
    X, y, eng = make_windows(feats, np.full(33, 7), rul, window=30)
    assert X.shape == (4, 30, 2)                                  # 33 - 30 + 1 windows
    assert (X[0] == feats[0:30]).all() and (X[3] == feats[3:33]).all()
    assert y.tolist() == [3, 2, 1, 0]                             # label of each window's last flight
    assert eng.tolist() == [7, 7, 7, 7]


def test_make_windows_keeps_engines_separate():
    feats = np.arange(64, dtype=float).reshape(32, 2)             # engine 1: rows 0-30 (31), engine 2: row 31 (1)
    engine = np.array([1] * 31 + [2])
    X, y, eng = make_windows(feats, engine, np.zeros(32), window=30)
    assert eng.tolist() == [1, 1]                                 # engine 2 is too short: no window; none mixes engines
    assert (X[1] == feats[1:31]).all()


def test_last_window_pads_short_engines_by_repeating_first_flight():
    feats = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0]])
    engine = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2])             # engine 1: 3 flights, engine 2: 7 flights
    X, eng = last_windows(feats, engine, window=5)
    assert eng.tolist() == [1, 2]
    assert X[0, :, 0].tolist() == [1, 1, 1, 2, 3]                 # padded at the front with flight 1
    assert X[1, :, 0].tolist() == [12, 13, 14, 15, 16]            # long enough: just the last 5


# ---------- the whole pipeline, end to end, on the real slice ----------

@pytest.fixture()
def ds(slice_cfg):
    return build_dataset(slice_cfg)


def test_end_to_end_no_engine_in_two_splits(ds, slice_dir):
    train_e, val_e = set(ds.engine_train), set(ds.engine_val)
    assert train_e & val_e == set()                               # the key leakage check
    assert train_e | val_e == set(range(1, 26))                   # all 25 slice engines used
    assert len(val_e) == 5                                        # 20% of 25
    assert set(ds.train_engine_lifetimes) == train_e and set(ds.val_engine_lifetimes) == val_e


def test_end_to_end_shapes_and_values(ds):
    n_feat = len(ds.feature_names)
    assert n_feat == 23                                           # 17 sensors + 6 conditions
    assert ds.X_train.shape[1:] == (30, n_feat) and ds.X_val.shape[1:] == (30, n_feat)
    assert ds.X_test.shape == (25, 30, n_feat)                    # one window per test engine
    n_windows = sum(T - 30 + 1 for T in ds.train_engine_lifetimes.values())
    assert len(ds.X_train) == len(ds.y_train) == len(ds.engine_train) == n_windows
    for X in (ds.X_train, ds.X_val, ds.X_test):
        assert np.isfinite(X).all()                               # no NaN / inf anywhere
    assert ds.y_train.min() == 0 and ds.y_train.max() == 125


def test_end_to_end_test_labels(ds, slice_dir):
    true = np.loadtxt(slice_dir / "RUL_FD004.txt")
    assert ds.y_test_uncapped.tolist() == true.tolist()
    assert ds.y_test.tolist() == np.minimum(true, 125).tolist()   # D8: capped for the headline


def test_end_to_end_scaler_learned_from_training_engines_only(ds, slice_cfg):
    df = add_condition(load_cmapss(slice_dir_path(slice_cfg) / "train_FD004.txt"))
    expected = fit_scaler(df[df["engine"].isin(ds.train_engine_lifetimes)], tol=1e-8)
    assert ds.scaler.mean == expected.mean and ds.scaler.std == expected.std


def test_end_to_end_is_deterministic(slice_cfg):
    a, b = build_dataset(slice_cfg), build_dataset(slice_cfg)
    assert (a.X_train == b.X_train).all() and (a.y_train == b.y_train).all() and (a.X_test == b.X_test).all()


def slice_dir_path(cfg):
    from pathlib import Path
    return Path(cfg["data"]["raw_dir"])
