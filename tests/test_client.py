"""FL client tests: one client, one round (DECISIONS F2, F4, LA5)."""
import numpy as np
import pytest
import torch

from rulfl.fl.client import client_seed, client_update
from rulfl.model.gru import RULModel


@pytest.fixture()
def cfg(slice_cfg):
    """Small, fast settings. E and lr are chosen by the pilot (F7); these are test values."""
    slice_cfg["fl"]["E"] = 1
    slice_cfg["fl"]["lr"] = 0.003
    return slice_cfg


@pytest.fixture()
def data(slice_ds):
    return slice_ds.X_train[:1000], slice_ds.y_train[:1000]    # one client's windows (any subset will do)


def new_model(seed=0):
    torch.manual_seed(seed)
    return RULModel(n_features=23, hidden=16, layers=1, dropout=0.0)


def global_weights():
    return {k: v.clone() for k, v in new_model(seed=0).state_dict().items()}


def distance(a, b):
    """How far apart two sets of weights are: sqrt of the sum of squared differences."""
    return sum(((a[k] - b[k]) ** 2).sum() for k in a).sqrt().item()


def test_returns_new_weights_and_number_of_windows(cfg, data):
    X, y = data
    g = global_weights()
    new, n = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1)
    assert n == len(X)                                       # the server weights this client by n (F3)
    assert set(new) == set(g)                                # same parameter names
    assert distance(new, g) > 0                              # it actually trained


def test_starts_from_the_global_weights_not_the_models_own(cfg, data):
    """Whatever weights the model object had before must not matter: every client starts from the global model."""
    X, y = data
    g = global_weights()
    a, _ = client_update(new_model(seed=5), g, X, y, cfg, mu=0.0, seed=1)
    b, _ = client_update(new_model(seed=9), g, X, y, cfg, mu=0.0, seed=1)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_same_result_different_seed_different(cfg, data):
    X, y = data
    g = global_weights()
    a, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1)
    b, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1)
    c, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=2)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)      # seed sets the batch order


def test_global_snapshot_is_a_real_copy(cfg, data):
    """Trap: model.state_dict() holds references to the live weights, not copies. Without the client's own
    snapshot, the 'global' g would move with p during training, (p - g) would stay 0, and FedProx would silently
    become FedAvg. So: passing live references must give exactly the same result as passing a clean copy."""
    X, y = data
    model = new_model()
    clean = {k: v.clone() for k, v in model.state_dict().items()}
    a, _ = client_update(new_model(), clean, X, y, cfg, mu=1.0, seed=1)
    b, _ = client_update(model, model.state_dict(), X, y, cfg, mu=1.0, seed=1)   # aliased: g IS model's weights
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_global_and_returned_weights_are_independent(cfg, data):
    X, y = data
    model, g = new_model(), global_weights()
    before = {k: v.clone() for k, v in g.items()}
    new, _ = client_update(model, g, X, y, cfg, mu=0.1, seed=1)
    assert all(torch.equal(g[k], before[k]) for k in g)      # the caller's global weights are untouched
    new["head.0.weight"] += 1.0                              # editing the returned copy must not touch the model
    assert not torch.equal(model.state_dict()["head.0.weight"], new["head.0.weight"])


def test_fedprox_keeps_weights_closer_to_global(cfg, data):
    """F4: the (mu/2)*sum (p-g)^2 penalty pulls local weights back toward the global ones; bigger mu, closer."""
    X, y = data
    g = global_weights()
    dist = {mu: distance(client_update(new_model(), g, X, y, cfg, mu=mu, seed=1)[0], g) for mu in (0.0, 1.0, 10.0)}
    assert dist[0.0] > dist[1.0] > dist[10.0], dist


def test_sample_weight_of_ones_is_plain_fedavg(cfg, data):
    """LA5: FedLA is the FedAvg code path with per-window weights. All weights 1 must change nothing."""
    X, y = data
    g = global_weights()
    a, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1)
    b, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1, sample_weight=np.ones(len(X), np.float32))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_sample_weight_changes_training(cfg, data):
    X, y = data
    g = global_weights()
    w = np.where(y == 125, 0.2, 1.5).astype(np.float32)     # e.g. down-weight capped windows
    a, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1)
    b, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1, sample_weight=w)
    assert not all(torch.equal(a[k], b[k]) for k in a)


def test_client_seed_is_fixed_and_distinct():
    """Each (run seed, round, client) gets its own seed, the same every time and in every process (Flower too)."""
    assert client_seed(1, 3, 2) == client_seed(1, 3, 2)
    seeds = {client_seed(s, r, k) for s in (1, 2) for r in range(1, 61) for k in range(5)}
    assert len(seeds) == 2 * 60 * 5                          # no two clients/rounds/runs share a seed


def test_sample_weight_precision_does_not_matter(cfg, data):
    """fedla_weights returns float64; the loss must still be computed in float32 like FedAvg's, so the arms
    differ only in the weights, not in arithmetic precision (LA5)."""
    X, y = data
    g = global_weights()
    w64 = np.where(y == 125, 0.7, 1.3)                       # float64, as fedla_weights returns
    a, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1, sample_weight=w64)
    b, _ = client_update(new_model(), g, X, y, cfg, mu=0.0, seed=1, sample_weight=w64.astype(np.float32))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_client_turns_on_deterministic_mode_itself(cfg, data):
    """M5. In Flower each client runs in its own worker process, where nobody else has set the flag."""
    X, y = data
    torch.use_deterministic_algorithms(False)                # a fresh worker process starts like this
    client_update(new_model(), global_weights(), X, y, cfg, mu=0.0, seed=1)
    assert torch.are_deterministic_algorithms_enabled()
