"""FedLA label-aware weights (DECISIONS LA1-LA7). Pure numpy; numbers come from configs/base.yaml."""
import numpy as np
import pytest

from rulfl.fl.fedla import fedla_weights, label_histogram


@pytest.fixture()
def la(slice_cfg):
    """edges, cap, clip exactly as the config says (rule 5: no numbers typed into code)."""
    return dict(edges=slice_cfg["fedla"]["bin_edges"], cap=slice_cfg["data"]["rul_cap"],
                clip=tuple(slice_cfg["fedla"]["clip"]))


def test_bin_boundaries(la):
    # [0,25) [25,50) [50,75) [75,100) [100,125) and the capped bin {125}
    y = np.array([0, 24, 25, 49, 50, 74, 75, 99, 100, 124, 125], dtype=np.float32)
    h = label_histogram(y, la["edges"], la["cap"])
    assert h.tolist() == [2, 2, 2, 2, 2, 1]
    assert label_histogram(np.array([124.0]), la["edges"], la["cap"]).tolist() == [0, 0, 0, 0, 1, 0]
    assert label_histogram(np.array([125.0]), la["edges"], la["cap"]).tolist() == [0, 0, 0, 0, 0, 1]


def test_out_of_range_labels_raise(la):
    for bad in (-1.0, 126.0):
        with pytest.raises(ValueError):
            label_histogram(np.array([bad]), la["edges"], la["cap"])


def test_weights_are_one_when_local_matches_global(la):
    rng = np.random.default_rng(0)
    y = rng.integers(0, la["cap"] + 1, size=500).astype(np.float32)
    local = label_histogram(y, la["edges"], la["cap"])
    for glob in (local, 7 * local):            # same distribution, any total
        w = fedla_weights(y, local, glob, la["edges"], la["cap"], la["clip"])
        np.testing.assert_allclose(w, 1.0, rtol=1e-12)


def test_mean_is_one_after_renormalising(la):
    rng = np.random.default_rng(1)
    y = rng.integers(0, la["cap"] + 1, size=300).astype(np.float32)
    local = label_histogram(y, la["edges"], la["cap"])
    glob = local + rng.integers(0, 400, size=local.size)    # other clients add arbitrary counts
    w = fedla_weights(y, local, glob, la["edges"], la["cap"], la["clip"])
    assert w.shape == y.shape
    assert np.all(w > 0)
    assert w.mean() == pytest.approx(1.0, rel=1e-12)


def test_upper_clip_applied(la):
    # Local: 99 windows at RUL 0, 1 at 125 -> p_local = [0.99, 0, 0, 0, 0, 0.01]
    # Global: half and half -> raw weight of the capped bin = 0.5 / 0.01 = 50, above the clip of 10.
    y = np.array([0.0] * 99 + [125.0], dtype=np.float32)
    local = label_histogram(y, la["edges"], la["cap"])
    glob = np.array([500, 0, 0, 0, 0, 500])
    w = fedla_weights(y, local, glob, la["edges"], la["cap"], la["clip"])
    lo, hi = la["clip"]
    w0 = min(max(0.5 / 0.99, lo), hi)          # bin 0 raw weight, inside the clip
    # Renormalising rescales every weight by the same factor, so the ratio shows whether the clip bit
    assert w[-1] / w[0] == pytest.approx(hi / w0, rel=1e-9)        # 19.8, not the unclipped 99
    assert w.mean() == pytest.approx(1.0, rel=1e-12)


def test_lower_clip_applied(la):
    # Local over-represents the capped bin 300x relative to global -> raw weight 1/300 < 0.1
    y = np.array([0.0] * 50 + [125.0] * 50, dtype=np.float32)
    local = label_histogram(y, la["edges"], la["cap"])
    glob = np.array([2990, 0, 0, 0, 0, 10]) + local
    w = fedla_weights(y, local, glob, la["edges"], la["cap"], la["clip"])
    lo, hi = la["clip"]
    p_g = glob / glob.sum()
    raw0 = p_g[0] / 0.5
    raw5 = p_g[5] / 0.5
    assert raw5 < lo                                   # the case really needs the lower clip
    assert w[-1] / w[0] == pytest.approx(lo / min(raw0, hi), rel=1e-9)


def test_local_counts_must_match_labels(la):
    y = np.array([0.0, 125.0])
    with pytest.raises(ValueError):
        fedla_weights(y, np.array([2, 0, 0, 0, 0, 0]), np.array([5, 0, 0, 0, 0, 5]),
                      la["edges"], la["cap"], la["clip"])


def test_on_real_slice_split_into_clients(slice_ds, la):
    """Real labels: split the slice's training engines into 2 groups by lifetime, as an extreme skew would."""
    eng = slice_ds.engine_train
    order = sorted(slice_ds.train_engine_lifetimes, key=lambda e: (slice_ds.train_engine_lifetimes[e], e))
    short = set(order[: len(order) // 2])
    masks = [np.isin(eng, list(short)), ~np.isin(eng, list(short))]
    hists = [label_histogram(slice_ds.y_train[m], la["edges"], la["cap"]) for m in masks]
    glob = hists[0] + hists[1]
    assert glob.sum() == len(slice_ds.y_train)
    for m, h in zip(masks, hists):
        w = fedla_weights(slice_ds.y_train[m], h, glob, la["edges"], la["cap"], la["clip"])
        assert np.all(np.isfinite(w))
        assert w.mean() == pytest.approx(1.0, rel=1e-9)
