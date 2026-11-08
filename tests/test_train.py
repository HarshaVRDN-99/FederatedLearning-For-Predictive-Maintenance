"""Centralized training tests (DECISIONS M3, M5), on the real slice with a small model and few epochs."""
import numpy as np
import pytest
import torch

from rulfl.model.gru import predict_rul
from rulfl.model.train import train_centralized


@pytest.fixture()
def cfg(slice_cfg):
    """Small, fast settings. The real `model:` section only exists after tuning (M2)."""
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.2}
    slice_cfg["train"]["max_epochs"] = 6
    slice_cfg["train"]["patience"] = 3
    return slice_cfg


def test_same_seed_gives_identical_result(cfg, slice_ds):
    m1, h1 = train_centralized(cfg, slice_ds, seed=1)
    m2, h2 = train_centralized(cfg, slice_ds, seed=1)
    assert h1 == h2                                                    # same losses, epoch by epoch
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)                                       # bit-identical weights
    _, h3 = train_centralized(cfg, slice_ds, seed=2)
    assert h3["train_loss"] != h1["train_loss"]                        # the seed really matters


def test_returns_the_best_epoch_not_the_last(cfg, slice_ds):
    model, hist = train_centralized(cfg, slice_ds, seed=1)
    best = hist["best_epoch"]                                          # index into the per-epoch lists
    assert hist["val_mae"][best] == min(hist["val_mae"])
    mae = np.abs(predict_rul(model, slice_ds.X_val, cap=125) - slice_ds.y_val).mean()
    assert mae == pytest.approx(hist["val_mae"][best], abs=1e-4)       # returned model IS that epoch's model


def test_early_stop_after_patience_epochs_without_improvement(cfg, slice_ds):
    """lr = 0: weights never change, so validation MAE never improves after epoch 0.
    Training must stop after epoch 0 + `patience` more epochs, not run to max_epochs."""
    cfg["model"]["lr"] = 0.0
    cfg["train"]["max_epochs"] = 50
    _, hist = train_centralized(cfg, slice_ds, seed=1)
    assert hist["best_epoch"] == 0
    assert len(hist["val_mae"]) == len(hist["train_loss"]) == 1 + cfg["train"]["patience"]


def test_learns_something(cfg, slice_ds):
    """After a few epochs it must beat the dumb baseline: always predict the average training label."""
    _, hist = train_centralized(cfg, slice_ds, seed=1)
    baseline = np.abs(slice_ds.y_train.mean() - slice_ds.y_val).mean()
    assert min(hist["val_mae"]) < 0.8 * baseline, f"{min(hist['val_mae']):.1f} vs baseline {baseline:.1f}"
    assert hist["train_loss"][-1] < hist["train_loss"][0]
