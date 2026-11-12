"""FL simulator tests: server aggregation and the full federated loop (DECISIONS F1-F8)."""
import numpy as np
import pytest
import torch

from rulfl.fl.client import client_seed
from rulfl.fl.simulate import fedavg, run_federated
from rulfl.model.gru import RULModel, predict_rul


@pytest.fixture()
def cfg(slice_cfg):
    """Small, fast settings. `model:` exists only after tuning (M2); E, R, lr, mu after the pilot (F5, F7)."""
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    slice_cfg["fl"].update(E=2, R=1, lr=0.003, mu=0.01)
    return slice_cfg


def manifest_for(ds, K):
    """Engines dealt round-robin to K clients. Stand-in until the partitioner writes real manifests.
    Keys are strings because that's what json.load gives back."""
    deal = lambda engines: {str(e): i % K for i, e in enumerate(sorted(set(engines.tolist())))}
    return {"train": deal(ds.engine_train), "val": deal(ds.engine_val)}


def test_fedavg_weights_by_number_of_windows():
    """1000 windows x 2.0 and 3000 windows x 4.0 -> (2000 + 12000) / 4000 = 3.5 (F3)."""
    a = {"w": torch.full((2, 2), 2.0)}
    b = {"w": torch.full((2, 2), 4.0)}
    avg = fedavg([(a, 1000), (b, 3000)])
    assert torch.allclose(avg["w"], torch.full((2, 2), 3.5))


def test_k1_fedavg_equals_centralized_training(cfg, slice_ds):
    """One client holding every training window: federated training must be plain centralized training.
    1 round of 2 epochs vs a hand-written 2-epoch loop with the same start weights and the same batch order."""
    cfg["fl"]["K"] = 1
    out = run_federated(cfg, slice_ds, manifest_for(slice_ds, 1), "fedavg", seed=7)

    m, f, cap = cfg["model"], cfg["fl"], cfg["data"]["rul_cap"]
    torch.manual_seed(7)                                          # same start as train_centralized(seed=7)
    model = RULModel(slice_ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=f["lr"])
    rng = np.random.default_rng(client_seed(7, 1, 0))             # round 1, client 0: its batch order
    X, y = torch.from_numpy(slice_ds.X_train), torch.from_numpy(slice_ds.y_train) / cap
    for _ in range(2):
        order = torch.from_numpy(rng.permutation(len(X)))
        for i in range(0, len(X), f["batch"]):
            idx = order[i:i + f["batch"]]
            loss = torch.nn.functional.mse_loss(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), f["clip"])
            opt.step()

    ref = model.state_dict()
    assert all(torch.allclose(out["best_state"][k], ref[k], atol=1e-6) for k in ref)


def load(cfg, ds, state):
    m = cfg["model"]
    model = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    model.load_state_dict(state)
    return model


def test_logs_every_round_with_exact_bytes(cfg, slice_ds):
    """F6: one entry per round. F8: bytes per round = 2 (down + up) x K x n_parameters x 4 (float32)."""
    cfg["fl"].update(K=2, R=3)
    out = run_federated(cfg, slice_ds, manifest_for(slice_ds, 2), "fedavg", seed=1)
    n_params = sum(v.numel() for v in out["best_state"].values())
    assert len(out["val_mae"]) == 3
    assert [len(c) for c in out["client_val_mae"]] == [2, 2, 2]
    assert out["bytes"] == [2 * 2 * n_params * 4] * 3


def test_returns_the_best_validation_round(cfg, slice_ds):
    """F6: all R rounds run; the reported model is the round with the lowest global validation MAE."""
    cfg["fl"].update(K=2, R=3)
    out = run_federated(cfg, slice_ds, manifest_for(slice_ds, 2), "fedavg", seed=1)
    best = int(np.argmin(out["val_mae"])) + 1                         # rounds count from 1
    assert out["best_round"] == best
    pred = predict_rul(load(cfg, slice_ds, out["best_state"]), slice_ds.X_val, 125)
    assert np.abs(pred - slice_ds.y_val).mean() == pytest.approx(out["val_mae"][best - 1])


def test_client_val_mae_uses_only_that_clients_engines(cfg, slice_ds):
    """PT5: client k is scored on the validation engines routed to it."""
    cfg["fl"].update(K=2, R=2)
    man = manifest_for(slice_ds, 2)
    out = run_federated(cfg, slice_ds, man, "fedavg", seed=1)
    model = load(cfg, slice_ds, out["best_state"])
    for k in range(2):
        mine = np.isin(slice_ds.engine_val, [int(e) for e, c in man["val"].items() if c == k])
        mae = np.abs(predict_rul(model, slice_ds.X_val[mine], 125) - slice_ds.y_val[mine]).mean()
        assert mae == pytest.approx(out["client_val_mae"][out["best_round"] - 1][k])


def test_same_seed_same_run(cfg, slice_ds):
    """Needed for Gate 2: a rerun (or Flower) must be able to reproduce the log exactly."""
    cfg["fl"].update(K=2, R=2)
    a = run_federated(cfg, slice_ds, manifest_for(slice_ds, 2), "fedavg", seed=1)
    b = run_federated(cfg, slice_ds, manifest_for(slice_ds, 2), "fedavg", seed=1)
    assert a["val_mae"] == b["val_mae"] and a["client_val_mae"] == b["client_val_mae"]


def test_fedprox_with_mu_0_is_fedavg_and_mu_above_0_is_not(cfg, slice_ds):
    """F4: FedProx is the FedAvg code path plus the proximal term; mu comes from cfg['fl']['mu']."""
    cfg["fl"].update(K=2, R=1)
    man = manifest_for(slice_ds, 2)
    avg = run_federated(cfg, slice_ds, man, "fedavg", seed=1)["best_state"]
    prox = run_federated(cfg, slice_ds, man, "fedprox", seed=1)["best_state"]
    assert not all(torch.equal(avg[k], prox[k]) for k in avg)
    cfg["fl"]["mu"] = 0.0
    prox0 = run_federated(cfg, slice_ds, man, "fedprox", seed=1)["best_state"]
    assert all(torch.equal(avg[k], prox0[k]) for k in avg)


def test_unknown_algo_raises(cfg, slice_ds):
    with pytest.raises(ValueError):
        run_federated(cfg, slice_ds, manifest_for(slice_ds, 1), "fedsgd", seed=1)


def manifest_by_lifetime(ds):
    """Extreme skew with K = 2: shorter-lived half of the training engines -> client 0, longer-lived -> client 1."""
    life = ds.train_engine_lifetimes
    order = sorted(life, key=lambda e: (life[e], e))
    train = {str(e): int(i >= len(order) // 2) for i, e in enumerate(order)}
    return {"train": train, "val": manifest_for(ds, 2)["val"]}


def test_fedla_without_skew_is_exactly_fedavg(cfg, slice_ds):
    """LA5. K = 1: the client's histogram IS the global one, so every weight is exactly 1 and nothing changes."""
    cfg["fl"]["K"] = 1
    man = manifest_for(slice_ds, 1)
    avg = run_federated(cfg, slice_ds, man, "fedavg", seed=1)["best_state"]
    la = run_federated(cfg, slice_ds, man, "fedla", seed=1)["best_state"]
    assert all(torch.equal(avg[k], la[k]) for k in avg)


def test_fedla_with_skew_differs_from_fedavg(cfg, slice_ds):
    cfg["fl"]["K"] = 2
    man = manifest_by_lifetime(slice_ds)
    avg = run_federated(cfg, slice_ds, man, "fedavg", seed=1)["best_state"]
    la = run_federated(cfg, slice_ds, man, "fedla", seed=1)["best_state"]
    assert not all(torch.equal(avg[k], la[k]) for k in avg)


def test_fedla_bytes_add_the_histograms_once(cfg, slice_ds):
    """LA6: before round 1, K histograms of 6 counts go up and the global one goes down to each of the K
    clients; 4 bytes per count, as F8 counts 4 bytes per weight. Later rounds cost the same as FedAvg."""
    cfg["fl"].update(K=2, R=2)
    man = manifest_by_lifetime(slice_ds)
    avg = run_federated(cfg, slice_ds, man, "fedavg", seed=1)["bytes"]
    la = run_federated(cfg, slice_ds, man, "fedla", seed=1)["bytes"]
    assert la == [avg[0] + 2 * 2 * 6 * 4, avg[1]]
