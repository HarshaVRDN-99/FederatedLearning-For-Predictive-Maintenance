"""GRU model tests (DECISIONS M1, M4)."""
import numpy as np
import pytest
import torch

from rulfl.data.pipeline import build_dataset
from rulfl.model.gru import RULModel, predict_rul


def test_one_output_per_window():
    model = RULModel(n_features=23, hidden=32, layers=1, dropout=0.0)
    out = model(torch.randn(8, 30, 23))                # 8 windows of 30 flights x 23 features
    assert out.shape == (8,)                           # one number per window


def test_head_is_a_separate_part_from_the_gru():
    """The head-refit probe freezes model.gru and retrains only model.head, so they must not overlap."""
    model = RULModel(n_features=23, hidden=32, layers=2, dropout=0.2)
    gru_ids = {id(p) for p in model.gru.parameters()}
    head_ids = {id(p) for p in model.head.parameters()}
    assert gru_ids and head_ids and not (gru_ids & head_ids)
    assert gru_ids | head_ids == {id(p) for p in model.parameters()}   # nothing outside the two parts


@pytest.mark.parametrize("layers, expected", [(1, 0.0), (2, 0.2)])
def test_dropout_only_between_two_gru_layers(layers, expected):
    model = RULModel(n_features=23, hidden=32, layers=layers, dropout=0.2)
    assert model.gru.dropout == expected               # PyTorch ignores (and warns about) dropout with 1 layer


def test_predict_rul_converts_to_cycles_and_clamps():
    class Fixed(torch.nn.Module):                      # stand-in model with known outputs (RUL / 125)
        def forward(self, x):
            return torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.2])
    pred = predict_rul(Fixed(), np.zeros((5, 30, 23), dtype=np.float32), cap=125)
    assert pred.tolist() == [0.0, 0.0, 62.5, 125.0, 125.0]   # x125, then clamp to [0, 125]


def test_overfit_one_batch(slice_cfg):
    """Required first test (CLAUDE.md): on 64 real windows, training loss must fall below 1% of its start.
    A model that cannot even memorise 64 examples is wired wrong.
    Stricter than CLAUDE.md on purpose: the start loss is high (untrained guesses), so 1% alone is reached
    while predictions are still ~15 cycles off. We also require memorisation: mean error < 2 cycles."""
    ds = build_dataset(slice_cfg)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds.X_train), size=64, replace=False)      # spread across engines and life stages
    X = torch.from_numpy(ds.X_train[idx])
    y = torch.from_numpy(ds.y_train[idx]) / 125                    # train on RUL / 125 (M3)
    assert y.std() > 0.1                                           # the batch really has varied labels

    torch.manual_seed(0)
    model = RULModel(n_features=X.shape[2], hidden=32, layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = None
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(model(X), y)
        start = loss.item() if start is None else start
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.01 * start, f"loss only fell from {start:.4f} to {loss.item():.4f}"
    mae = np.abs(predict_rul(model, ds.X_train[idx], cap=125) - ds.y_train[idx]).mean()
    assert mae < 2.0, f"model did not memorise the batch: mean error {mae:.2f} cycles"
