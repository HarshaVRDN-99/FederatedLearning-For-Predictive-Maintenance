"""Flower app, step A: a toy K-client simulation through the real ClientApp/ServerApp wiring.

The toy model is a numpy linear regression, so nothing here depends on client_update yet. The plain-numpy
FedAvg loop in `_reference_fedavg` is a miniature of what Gate 2 does with the real simulator.
"""
import numpy as np
import pytest

from flwr.serverapp.strategy import FedAvg

from rulfl.flower_app.app import make_strategy, run_flower
from rulfl.flower_app.toy import toy_eval, toy_initial, toy_train

ROUNDS = 3
TRAIN_CONFIG = {"lr": 0.05, "steps": 5}


def _reference_fedavg(K, rounds):
    """FedAvg by hand: every client trains from the same global weights, then a window-count-weighted mean (F3)."""
    w = toy_initial()
    for _ in range(rounds):
        outs = [toy_train(w, k, TRAIN_CONFIG) for k in range(K)]
        n = np.array([o[1] for o in outs], dtype=np.float64)
        w = [sum(o[0][i] * nk for o, nk in zip(outs, n)) / n.sum() for i in range(len(w))]
    return w


@pytest.fixture(scope="module")
def toy_run():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    K = cfg["fl"]["K"]
    res = run_flower(train_fn=toy_train, initial=toy_initial(), K=K, num_rounds=ROUNDS,
                     train_config=TRAIN_CONFIG, evaluate_fn=lambda r, w: toy_eval(w, K), best_key="mse")
    return K, res


def test_toy_simulation_runs_to_completion(toy_run):
    """TEAM_TASKS first test: a K-client simulation finishes every round."""
    K, res = toy_run
    assert [h["round"] for h in res.history] == list(range(ROUNDS + 1))   # round 0 = initial weights
    assert all(np.all(np.isfinite(a)) for a in res.final)


def test_matches_hand_computed_fedavg(toy_run):
    """Unequal client sizes, all K clients every round (F1), weighted by window count (F3)."""
    K, res = toy_run
    ref = _reference_fedavg(K, ROUNDS)
    for got, want in zip(res.final, ref):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_dropping_a_client_would_be_caught(toy_run):
    """Guards the test above: the K-1 answer is clearly different, so a missing client cannot pass silently."""
    K, res = toy_run
    wrong = _reference_fedavg(K - 1, ROUNDS)
    assert not all(np.allclose(g, w, rtol=1e-6) for g, w in zip(res.final, wrong))


def test_server_metrics_every_round_and_best_round(toy_run):
    """F6: global evaluation after every round; the reported model is the best-evaluation round."""
    K, res = toy_run
    mse = [h["metrics"]["mse"] for h in res.history]
    assert mse[-1] < mse[0]                                  # it learned
    assert res.best_round == int(np.argmin(mse))
    best = toy_eval(res.best_arrays, K)["mse"]
    assert best == pytest.approx(min(mse), rel=1e-12)


def test_strategy_is_plain_fedavg_full_participation():
    """F4: the proximal term lives in the client loss, so the server must never be Flower's FedProx."""
    s = make_strategy(K=5)
    assert type(s) is FedAvg                                 # FedProx subclasses FedAvg, so isinstance is not enough
    assert s.fraction_train == 1.0
    assert s.min_train_nodes == s.min_available_nodes == 5
    assert s.fraction_evaluate == 0.0                        # evaluation is server-side, on all val engines (F6)
    assert s.weighted_by_key == "num-examples"


def test_bad_k_raises():
    with pytest.raises(ValueError):
        make_strategy(K=1)
