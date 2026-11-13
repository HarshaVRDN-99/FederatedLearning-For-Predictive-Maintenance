import numpy as np
from rulfl.eval.metrics import nasa_score
from rulfl.eval.metrics import life_stage_mae
import pytest

def test_nasa_equal_reference_costs():
    pred = np.array([10, 20])
    true = np.array([0, 33])

    expected = 2 * (np.e - 1)
    assert abs(nasa_score(pred, true) - expected) < 1e-6

def test_late_prediction_costs_more_than_early():
    late = nasa_score([5], [0])
    early = nasa_score([0], [5])

    assert late > early
    assert abs(late - 0.64872) < 1e-4
    assert abs(early - 0.46905) < 1e-4


from rulfl.eval.metrics import metrics


def test_mae_rmse_hand_calculated():
    result = metrics([10, 20], [0, 33])

    assert result["mae"] == 11.5
    assert abs(result["rmse"] - 11.59741) < 1e-4

    result2 = metrics([1, 2, 3], [2, 2, 5])

    assert result2["mae"] == 1.0
    assert abs(result2["rmse"] - 1.29099) < 1e-4

def test_life_stage_boundaries():
    result = life_stage_mae(
        pred=[50, 60, 130, 140],
        true=[49, 50, 124, 125],
    )

    assert result["n_near_failure"] == 1
    assert result["n_mid"] == 2
    assert result["n_healthy"] == 1

    assert result["near_failure"] == 1.0
    assert result["mid"] == 8.0
    assert result["healthy"] == 15.0

def test_empty_life_stage_group_returns_nan():
    result = life_stage_mae(
        pred=[60, 70],
        true=[55, 80],
    )

    assert np.isnan(result["near_failure"])
    assert result["n_near_failure"] == 0

def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        metrics([1, 2], [1])

    with pytest.raises(ValueError):
        metrics([], [])

def test_non_finite_predictions_raise_value_error():
    with pytest.raises(ValueError):
        metrics([float("nan")], [10])

    with pytest.raises(ValueError):
        metrics([float("inf")], [10])

def test_life_stage_uses_config_edges():
    import yaml

    with open("configs/base.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    edges = tuple(config["eval"]["life_stage_edges"])

    result = life_stage_mae(
        pred=[10, 60, 130],
        true=[40, 75, 140],
        edges=edges,
    )

    assert result["n_near_failure"] == 1
    assert result["n_mid"] == 1
    assert result["n_healthy"] == 1