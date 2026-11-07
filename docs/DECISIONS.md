# Decisions

Choices the proposal left open, decided before the code that uses them is built.
Derived from our own design (CLAUDE.md); the reference repo's list was used only as a cross-check.
Once a section is committed it is frozen: changing a decision needs a new entry here saying what changed and why,
made before any registered run uses it. Numbers that live in code are in `configs/base.yaml`.

Status: Data ✅ · Model ⏳ · Partition ⏳ · FL (FedAvg / FedProx) ⏳ · FedLA ⏳ · Evaluation and analysis ⏳

## Project-level

| # | Decision | Reason |
|---|---|---|
| P1 | Python 3.12; every package pinned in `requirements.lock` (torch is the CPU build) | One environment for everyone; 3.12 is the version the pinned stack was tested on |
| P2 | A package is added to `requirements.txt` only when code needs it, with the reason in a comment | Every dependency can be justified (CLAUDE.md rule 7) |
| P3 | Tests run on a committed, hashed slice of the real NASA files (`tests/data/`, made by `tests/make_slice.py`), not synthetic data. Deviation from the CLAUDE.md plan (`tests/fakes.py`) | Real data has quirks a generator gets wrong (we found three: settings jitter, near-constant sensors, lifetimes > 125). CI can run real-format tests without the full data |
| P4 | Raw NASA files are never committed; their SHA-256 is in `data/raw/CHECKSUMS.txt` | Everyone checks they have byte-identical inputs |

## Data pipeline

| # | Decision | Reason |
|---|---|---|
| D1 | Validation = 20% of FD004 training engines (50 of 249), chosen once with `split_seed: 0`, fixed forever. Split by engine before windowing | ~40 train engines per client at K = 5; ~10 validation engines per client for worst-client MAE |
| D2 | Operating condition of a row = its settings rounded (altitude to integer, Mach to 2 dp, throttle to integer). The set of conditions is read from training-split rows | Real jitter is ≤ 0.008, so rounding is exact; no clustering library or random init needed |
| D3 | Each sensor is z-scored per operating condition, using mean and std from training-split rows only, pooled over all clients | Same input features at every H and every arm, so differences come only from label skew. Limitation for the report: a real federation could not pool raw statistics |
| D4 | Drop a sensor column only if its std is zero (≤ 1e-8) within every operating condition, measured on training-split rows; the same columns are then dropped from train, validation and test. Raw files are never modified | Such a sensor only encodes the condition (already in the one-hot, D5) and would divide by zero in per-condition z-scoring. On real FD004 this drops sensors 0, 4, 17, 18 (0-based) and keeps 17. A first proposal (std < 0.1% of mean) was rejected after checking real data: it would have dropped sensors 1, 7, 12, which trend with wear |
| D5 | Input per cycle = kept normalised sensors + one-hot of the operating condition | Normalisation removes the condition's level; the one-hot keeps which condition the cycle was in |
| D6 | Train and validation windows: every 30-cycle window (stride 1), label `min(T − t, 125)` at the window's last cycle. Validation MAE uses all validation windows | All-window validation MAE is stable; it drives model selection |
| D7 | Test input = last 30 cycles per test engine; engines shorter than 30 are left-padded by repeating their first cycle | A repeated early cycle looks like the healthy engine it is; zero padding would insert fake "average" readings |
| D8 | Test labels capped at 125 for the headline metric; uncapped values kept as `y_test_uncapped` | Comparable to published FD004 results; a model trained on capped labels cannot predict above 125 |

Frozen signature:

```python
build_dataset(cfg: dict) -> Dataset
# Dataset fields:
#   X_train (n, 30, f), y_train (n,), engine_train (n,)   one row per training window
#   X_val, y_val, engine_val                               same for validation engines
#   X_test (n_test_engines, 30, f), y_test, y_test_uncapped
#   feature_names: list[str]
#   train_engine_lifetimes: dict[int, int]                 engine id -> total cycles (for the partitioner)
```

## Model

⏳ To decide before building `src/rulfl/model/`.

## Partition

⏳

## FL (FedAvg / FedProx)

⏳

## FedLA

⏳

## Evaluation and analysis

⏳
