# Decisions

Choices the proposal left open, decided before the code that uses them is built.
Derived from our own design (CLAUDE.md); the reference repo's list was used only as a cross-check.
Once a section is committed it is frozen: changing a decision needs a new entry here saying what changed and why,
made before any registered run uses it. Numbers that live in code are in `configs/base.yaml`.

Status: Data ✅ · Model ✅ · Partition ✅ · FL (FedAvg / FedProx) ✅ · FedLA ✅ · Evaluation and analysis ✅

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
| D2 | Operating condition of a row = its settings rounded to whole numbers: altitude, Mach × 100, throttle (key like `42_84_100`). The set of conditions is read from training-split rows | Real jitter is ≤ 0.008, so rounding is exact; no clustering library or random init needed. Whole numbers avoid FD001's jitter around Mach 0 splitting into "0.0" and "-0.0" (found by a test) |
| D3 | Each sensor is z-scored per operating condition, using mean and std from training-split rows only, pooled over all clients | Same input features at every H and every arm, so differences come only from label skew. Limitation for the report: a real federation could not pool raw statistics |
| D4 | Drop a sensor column only if its std is zero (≤ 1e-8) within every operating condition, measured on training-split rows; the same columns are then dropped from train, validation and test. Raw files are never modified | Such a sensor only encodes the condition (already in the one-hot, D5) and would divide by zero in per-condition z-scoring. On real FD004 this drops s1, s5, s18, s19 (NASA numbering) and keeps 17. A first proposal (std < 0.1% of mean) was rejected after checking real data: it would have dropped s2, s8, s13, which trend with wear. Where a kept sensor is constant in one condition only (s16 in 4 of 6 FD004 conditions), its normalised value in that condition is set to 0 (no division by zero) |
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
#   val_engine_lifetimes: dict[int, int]                   same for validation engines (routing, PT5)   [added when built]
#   scaler: Scaler                                         conditions, kept/dropped sensors, per-condition mean/std [added when built]
```

As built (real FD004): 43,064 train windows (199 engines), 10,964 validation windows (50 engines), 248 test windows,
23 features (17 sensors + 6 condition one-hot). Builds in under a second, so it is rebuilt in memory, not cached.

## Model

| # | Decision | Reason |
|---|---|---|
| M1 | GRU (batch-first) → hidden state at the last time step → head: Linear(hidden, hidden/2) → ReLU → Linear(hidden/2, 1). The head is its own submodule, `model.head` | The head-refit probe freezes `model.gru` and retrains only `model.head`; the boundary must exist in code |
| M2 | Tuned once, centralized, on validation MAE only: hidden {32, 64, 128} × layers {1, 2} × lr {0.001, 0.003} (12 configs, seed 1). Top 2 rerun with seeds 1–3; lower mean wins. Dropout 0.2 between GRU layers when layers = 2, else 0. Then frozen for every arm | Small grid a laptop can run; covers common FD004 sizes. Never retuned on federated results (CLAUDE.md) |
| M3 | Loss = MSE on y/125 (model outputs RUL/125). Adam, batch 256, gradient-norm clip 1.0. Centralized: max 100 epochs, early stop after 10 epochs without validation-MAE improvement, keep best epoch | Targets in 0–1 train more smoothly; clipping guards against GRU gradient spikes; early stopping prevents overfitting |
| M4 | Evaluation (validation and test, every arm) uses one function: prediction = clamp(model(x) × 125, 0, 125). Training loss is unclamped | RUL cannot be negative and labels never exceed 125. Clamping during training would block gradients; a sigmoid output would saturate on the many labels at the cap |
| M5 | All registered runs on CPU, seeded (init, batch order) with `torch.use_deterministic_algorithms(True)`. Revisit after tuning: if the frozen model is too slow on CPU, ALL registered runs move to GPU together, never mixed | Measured on a 12-thread laptop CPU (random data, real shapes, 40k windows): 1.7 s/epoch (32×1) to 10.8 s/epoch (128×2), i.e. ~7–42 min per federated run. Mixing devices would add numerical noise to paired comparisons, and the expected FedAvg/FedProx differences are small. One CPU lock file for everyone. [added when built] CPU thread count also changes results (~7th digit; measured 1 vs 3 vs 12 threads), so every registered run uses the same count: 3 (from the parallel-throughput benchmark), set by the runner |

Frozen signatures:

```python
class RULModel(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float): ...
    def forward(self, x: Tensor) -> Tensor:   # x (batch, 30, n_features) -> (batch,)  RUL / 125
    # submodules: self.gru, self.head

predict_rul(model, X, cap) -> np.ndarray       # clamp(model(X) * cap, 0, cap), cap = data.rul_cap (M4)
train_centralized(cfg, ds, seed) -> tuple[RULModel, dict]   # dict: per-epoch train_loss, val_mae, best_epoch
    # [added when built] reads cfg['model'] = {hidden, layers, lr, dropout} (set by tuning, M2) and cfg['train'];
    # best_epoch = 0-based index into the per-epoch lists; the returned model has that epoch's weights
```

## Partition

Rule (agreed in the proposal): sort training engines by lifetime, cut into K blocks; engine in block b goes to
client b with probability H, otherwise to a uniformly random client (so P(client k) = H·1[b = k] + (1 − H)/K).

| # | Decision | Reason |
|---|---|---|
| PT1 | H ∈ {0, 0.25, 0.5, 0.75, 1} | Evenly spaced from IID to extreme skew; enough points to see a trend |
| PT2 | Sort by (lifetime, engine id); K contiguous blocks as equal as possible (199 engines → 40, 40, 40, 40, 39) | Tie rule makes the blocks deterministic |
| PT3 | Partition randomness comes from the run seed (main seeds 1–5). The same (H, seed) partition feeds all three arms | Several partitions per H, so no result rests on one lucky split; identical partitions keep arms paired |
| PT4 | No redraws. If any client gets fewer than 10 training engines, raise an error | Expected ~40 per client, so this should never fire; a silent redraw would hide it if it did |
| PT5 | The 50 validation engines are routed to clients by the same rule, H and seed (their own lifetime blocks). Best-round selection uses all 50; worst-client MAE uses each client's routed ones | Gives every client its own validation set without moving any engine across the fixed train/val split |
| PT6 | Skew = mean over clients of the Wasserstein-1 distance (in cycles) between that client's training RUL labels and all training labels. A test checks the seed-averaged skew strictly increases over the H grid | Required by the proposal; Wasserstein reads as "cycles of label shift", easy to explain; computed with numpy only |
| PT7 | Each (H, seed) partition is written once to `manifests/H{H:.2f}_s{seed}.json` (engine → client for train and val, plus SHA-256), committed, never hand-edited | Every arm and every rerun reads the identical partition (CLAUDE.md rule 4) |

Frozen signatures:

```python
assign_clients(lifetimes: dict[int, int], K: int, H: float, seed: int) -> dict[int, int]   # engine id -> client 0..K-1
skew(labels_by_client: list[np.ndarray], all_labels: np.ndarray) -> float               # mean Wasserstein-1, cycles
```

## FL (FedAvg / FedProx)

| # | Decision | Reason |
|---|---|---|
| F1 | All K clients train every round (full participation) | With K = 5, client sampling would add randomness unrelated to label skew |
| F2 | Each client builds a fresh Adam optimiser every round; batch 256, gradient-norm clip 1.0 (as centralized) | No hidden client state carried across rounds; matches a stateless Flower client (Gate 2) |
| F3 | Server aggregation = average of client weights, weighted by each client's number of training windows | Agreed in the proposal |
| F4 | FedProx client loss = MSE + (μ/2)·Σ(p − g)², g = global weights copied before local training. FedAvg is the same code with μ = 0 | One code path for both arms, so they stay comparable (proposal) |
| F5 | μ chosen once from {0.001, 0.01, 0.1, 1} by lowest mean validation MAE at H = 1, seeds 101–103 | H = 1 is where FedProx should help most (its best shot); selection seeds never overlap the reported seeds 1–5 |
| F6 | Every run does all R rounds (no early stopping); global validation MAE (all validation engines) after every round; the reported model is the best-validation round | Identical round counts for reference and Flower; enables rounds-to-110% |
| F7 | E, R and federated lr set in the pilot: E ∈ {1, 3} × lr ∈ {tuned lr, tuned lr / 3}, R = 60, H ∈ {0, 1}, seeds 101–102, validation only. Pick the setting where FedAvg learns best at H = 0. If FedAvg does not degrade at H = 1, switch to K = 10 (proposal fallback) before the main grid | Settings chosen before any result that matters; the fallback keeps the study meaningful |
| F8 | Bytes communicated per round = 2 × K × n_parameters × 4 (float32 down + up); FedLA adds its one-time histograms | Defines the communication metric exactly |
| F9 | The Flower server is always Flower's plain `FedAvg` (`fraction_train=1.0`, `min_train_nodes=min_available_nodes=K`, `fraction_evaluate=0.0`, `weighted_by_key="num-examples"`), for all three arms. Training goes through a pluggable `train_fn(arrays, client_id, config) -> (new_arrays, n_training_windows)`; the client is stateless and never uses `context.state` | One server for FedAvg, FedProx and FedLA, since μ and sample weights live in the client loss (F4, LA5). Flower's built-in `FedProx` would apply μ a second time, and it subclasses `FedAvg`, so the test checks `type(s) is FedAvg`, not `isinstance` |
| F10 | Evaluation is server-side only (`evaluate_fn`), on all validation engines, after every round including round 0 (the initial weights). The best round is the lowest validation MAE; ties keep the earlier round | Implements F6 the same way as the reference simulator, so Gate 2 compares like with like. Round 0 gives the untrained baseline for every run |

Frozen signatures:

```python
client_update(model, global_weights, X, y, cfg, mu: float, sample_weight=None) -> tuple[dict, int]  # new weights, n_examples
run_federated(cfg, ds, manifest, algo: str, seed: int) -> dict   # per round: global val MAE, per-client val MAE, bytes
```

`sample_weight` lets FedLA reuse `client_update` unchanged.

## FedLA

Evidence (real FD004 training engines, H = 1, K = 5): every engine lives > 125 cycles, so each passes through every
RUL value 0–124 once. Within 0–124 every client's labels are flat (client 1: ~19% per 25-cycle bin; client 5: ~8%).
The skew is in the capped bin (RUL = 125, "healthy"): 8% of client 1's windows vs 62% of client 5's.
Client sizes also differ (6,673 vs 16,243 windows), so window-weighted aggregation gives long-lived clients ~2.4× the say.

| # | Decision | Reason |
|---|---|---|
| LA1 | Bins: [0,25), [25,50), [50,75), [75,100), [100,125) and the capped bin {125}: 6 bins. Fixed, not selected on validation (deviation from CLAUDE.md "bin width fixed on validation") | Labels are flat inside 0–124, so bin width barely changes the weights; selection runs would choose between near-identical options |
| LA2 | Clip weights to [0.1, 10], then renormalise to mean 1 on the client. Fixed, not selected (same deviation). Revisit only if the K = 10 fallback is used | Largest unclipped weight at K = 5, H = 1 is ≈ 5.4 (client 1, capped bin), so the clip never binds; it is a safety net |
| LA3 | Histograms count training windows (what the loss sees), not engines | Weights must match the per-window loss |
| LA4 | Weights computed once before round 1 and fixed for all rounds | Labels never change |
| LA5 | FedLA = FedAvg code path with `sample_weight`, μ = 0, same window-count aggregation | Only the weights differ, so FedLA and FedAvg stay paired |
| LA6 | Communication: K × 6 counts up and 6 down, once; added to bytes communicated | Exact accounting |
| LA7 | Limitation stated in the report: histograms reveal each client's label distribution | No privacy guarantees are claimed (CLAUDE.md rule 8) |
| LA8 | Labels outside [0, cap], or bin edges that do not rise strictly from 0 to cap, raise ValueError | A bad label would otherwise land silently in the wrong bin |
| LA9 | `fedla_weights` raises if `local_counts` differs from `label_histogram(y)`, or if global < local in any bin | Catches a caller passing the wrong client's histogram |
| LA10 | "Mean 1 before clipping" holds only if the client has windows in every bin with global mass; true on real FD004 (every engine outlives 125 cycles). Renormalisation guarantees mean 1 regardless | States the limit of the note under LA7 |

Note: before clipping the weights already have mean 1 on each client (Σ_b p_local(b)·p_global(b)/p_local(b) = 1),
so renormalisation only matters after clipping. Required tests: weights = 1 when local = global; mean = 1 after
renormalising; clipping applied.

Frozen signatures:

```python
label_histogram(y: np.ndarray, edges: list[int], cap: int) -> np.ndarray      # 6 window counts
fedla_weights(y: np.ndarray, local_counts, global_counts, edges, cap, clip) -> np.ndarray   # per-window weights, mean 1
```

## Evaluation and analysis

| # | Decision | Reason |
|---|---|---|
| E1 | Headline: test MAE over the 248 last-window test predictions (clamped, M4; capped labels, D8). Also RMSE and NASA score: d = predicted − true, cost exp(d/10) − 1 if d ≥ 0 (late) else exp(−d/13) − 1 (early), summed over engines | Proposal metrics; late predictions are penalised harder because they are dangerous |
| E2 | Worst-client MAE = max over clients of the global model's validation MAE on that client's routed validation engines (PT5), at the best round | Shows whether skew hurts some clients far more than others |
| E3 | Rounds-to-110%: first round whose global validation MAE ≤ 1.10 × mean best validation MAE of the 5 centralized runs. Runs that never reach it are recorded as "not reached" (censored), never dropped | Proposal: censored runs counted |
| E4 | Life-stage breakdown on test engines by true RUL: near failure < 50 (79 engines), mid 50–124 (102), healthy ≥ 125 / capped (67). Computed from saved predictions | Balanced groups; answers where the damage happens. Counts come from labels only, no model output |
| E5 | Baselines: centralized (all training engines, seeds 1–5); local-only (each client alone) at H = 0 and H = 1, seeds 1–5 | Centralized = the target; local-only shows what federation adds, at the two extremes only to save compute |
| E6 | Head-refit probe (diagnostic, labelled as such): take a run's best-round global model, freeze `model.gru`, re-initialise `model.head`, train only the head on all training windows with M3 settings (early stop on validation), then score on test | If test MAE returns near centralized, the damage is in the head; if not, it is in the features |
| E7 | Comparisons are paired by (H, seed). Per H: mean difference vs FedAvg with a 95% bootstrap interval over seeds. One-sided exact Wilcoxon signed-rank per H and pooled over H ≥ 0.5 (15 pairs). Implemented with numpy only | Paired design removes partition luck. With 5 pairs the smallest one-sided p is 1/32 ≈ 0.031, so per-H tests are weak; the intervals and the pooled test carry the evidence. Hypotheses written down before the main grid |
| E8 | Every run saves its config, manifest hash, per-round log, final metrics, best checkpoint and test predictions. Every report number and figure is generated from these files | CLAUDE.md rule 6 |

Frozen signatures:

```python
nasa_score(pred: np.ndarray, true: np.ndarray) -> float
metrics(pred, true) -> dict                     # mae, rmse, nasa
life_stage_mae(pred, true, edges=(50, 125)) -> dict
```
