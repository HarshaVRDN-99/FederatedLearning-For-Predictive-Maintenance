"""Fake C-MAPSS data in the exact text layout of the NASA files, with known true answers.

Every later test (pipeline, leakage, windows, model, partitioner) runs on this instead of the
real data, so tests are fast, run in CI (the real data is never committed), and can check exact values.

Layout of each row (26 space-separated columns, like train_FD004.txt):
    engine_id  cycle  setting1 setting2 setting3  sensor1 ... sensor21

Each sensor reading = level of the current operating condition + wear trend + noise.
Wear = (cycle / lifetime) ** 2, so degradation speeds up near failure.

Idea of a condition-level + wear-trend generator borrowed from the reference repo's
src/rulfl/data/synthetic.py; the conditions here are the six actually found in train_FD004.txt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (altitude in 1000 ft, Mach number, throttle resolver angle): the six FD004 operating conditions.
# CONDITIONS[0] (sea level) is also FD001's single condition.
CONDITIONS = [
    (0.0, 0.00, 100.0),
    (10.0, 0.25, 100.0),
    (20.0, 0.70, 100.0),
    (25.0, 0.62, 60.0),
    (35.0, 0.84, 100.0),
    (42.0, 0.84, 100.0),
]
N_SENSORS = 21
# 0-based sensor indices, mirroring the real data: these 6 are constant within a condition ...
CONSTANT_SENSORS = [0, 4, 9, 15, 17, 18]
# ... these 14 degrade with wear; sensor 5 is pure noise.
TRENDING_SENSORS = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
NOISE_ONLY_SENSORS = [5]

WINDOW = 30          # only used to decide which test engines count as "short"
NOISE_SD = 0.5       # sensor noise; small next to the trend sizes (3 to 6) so wear is learnable


@dataclass
class FakeInfo:
    """The true answers, keyed by engine id (1-based, as in the files)."""
    train_life: dict[int, int]    # total cycles until failure (= rows in train file)
    test_life: dict[int, int]     # total cycles the test engine would have lived
    test_len: dict[int, int]      # cycles actually written to the test file
    test_rul: dict[int, int]      # test_life - test_len (= RUL file value)


def make_fake_cmapss(
    out_dir: str | Path,
    *,
    fd: str = "FD004",
    n_conditions: int = 6,
    n_train: int = 40,
    n_test: int = 40,
    min_life: int = 60,
    max_life: int = 150,
    n_short_test: int = 2,
    seed: int = 0,
) -> FakeInfo:
    """Write train_{fd}.txt, test_{fd}.txt, RUL_{fd}.txt into out_dir and return the true answers."""
    if min_life <= WINDOW:
        raise ValueError(f"min_life must exceed the window ({WINDOW}) so normal test engines fit a window")
    if not 1 <= n_conditions <= len(CONDITIONS):
        raise ValueError(f"n_conditions must be 1..{len(CONDITIONS)}")
    if n_short_test > n_test:
        raise ValueError("n_short_test cannot exceed n_test")

    rng = np.random.default_rng(seed)        # ONE generator, used in a fixed order -> same seed, same files
    conditions = np.array(CONDITIONS[:n_conditions])

    # Fixed "physics" shared by all engines: each condition has its own sensor levels,
    # each trending sensor has its own direction and size of drift.
    levels = rng.uniform(10.0, 1000.0, size=(n_conditions, N_SENSORS)).round(2)
    slopes = np.zeros(N_SENSORS)
    slopes[TRENDING_SENSORS] = rng.choice([-1.0, 1.0], len(TRENDING_SENSORS)) * rng.uniform(3.0, 6.0, len(TRENDING_SENSORS))
    noisy = np.zeros(N_SENSORS)
    noisy[TRENDING_SENSORS + NOISE_ONLY_SENSORS] = NOISE_SD

    def engine_rows(unit: int, life: int) -> np.ndarray:
        """All `life` cycles of one engine, as a (life, 26) array."""
        cycles = np.arange(1, life + 1)
        cond = rng.integers(0, n_conditions, size=life)
        settings = conditions[cond] + rng.uniform(-1, 1, size=(life, 3)) * [0.003, 0.0003, 0.0]   # tiny jitter, like real files
        wear = (cycles / life) ** 2
        sensors = levels[cond] + np.outer(wear, slopes) + rng.normal(size=(life, N_SENSORS)) * noisy
        return np.column_stack([np.full(life, unit), cycles, settings, sensors])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_life, train_rows = {}, []
    for unit in range(1, n_train + 1):
        life = int(rng.integers(min_life, max_life + 1))
        train_life[unit] = life
        train_rows.append(engine_rows(unit, life))

    short_units = set((rng.choice(n_test, size=n_short_test, replace=False) + 1).tolist())
    test_life, test_len, test_rul, test_rows = {}, {}, {}, []
    for unit in range(1, n_test + 1):
        life = int(rng.integers(min_life, max_life + 1))
        if unit in short_units:
            seen = int(rng.integers(10, WINDOW))          # 10..29 cycles: shorter than one window
        else:
            seen = int(rng.integers(WINDOW, life))        # 30..life-1: always stops before failure
        test_life[unit], test_len[unit], test_rul[unit] = life, seen, life - seen
        test_rows.append(engine_rows(unit, life)[:seen])

    _write_rows(out / f"train_{fd}.txt", np.vstack(train_rows))
    _write_rows(out / f"test_{fd}.txt", np.vstack(test_rows))
    (out / f"RUL_{fd}.txt").write_text("".join(f"{test_rul[u]}\n" for u in range(1, n_test + 1)))
    return FakeInfo(train_life, test_life, test_len, test_rul)


def _write_rows(path: Path, rows: np.ndarray) -> None:
    """Same text format as NASA: integer id and cycle, 4-decimal floats, trailing spaces."""
    with open(path, "w", newline="\n") as f:
        for r in rows:
            f.write(f"{int(r[0])} {int(r[1])} " + " ".join(f"{v:.4f}" for v in r[2:]) + "  \n")
