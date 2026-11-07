"""Cut a small, fixed sample of the real NASA C-MAPSS files into tests/data/ for the test suite.

Run once (output is committed):   python tests/make_slice.py
Rows are copied byte-for-byte from the NASA files; only the engine id is renumbered 1..N so the
slice has the same layout as a real file. SLICE.json records which original engines were taken
and the hash of every slice file. Never hand-edit tests/data/; rerun this script instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 0
WINDOW = 30
# How many engines per file. FD004 test includes N_SHORT engines shorter than one window.
SIZES = {"FD004": {"train": 25, "test": 25}, "FD001": {"train": 10, "test": 10}}
N_SHORT = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_raw(raw_dir: Path, names: list[str]) -> dict[str, str]:
    """Refuse to slice files that don't match data/raw/CHECKSUMS.txt."""
    expected = {}
    for line in (raw_dir / "CHECKSUMS.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            digest, name = line.split()
            expected[name.lstrip("*")] = digest
    hashes = {}
    for name in names:
        hashes[name] = _sha256(raw_dir / name)
        if hashes[name] != expected.get(name):
            raise ValueError(f"{name} does not match CHECKSUMS.txt; re-download the NASA files")
    return hashes


def _read_engines(path: Path) -> dict[int, list[str]]:
    """{original engine id: its lines, exactly as written (line endings kept)}"""
    engines: dict[int, list[str]] = {}
    with open(path, newline="") as f:
        for line in f:
            engines.setdefault(int(line.split(maxsplit=1)[0]), []).append(line)
    return engines


def _write_renumbered(path: Path, engines: dict[int, list[str]], chosen: list[int]) -> None:
    """Write the chosen engines in original-id order, renumbered 1..N; rest of each line untouched."""
    with open(path, "w", newline="") as f:
        for new_id, old_id in enumerate(chosen, start=1):
            for line in engines[old_id]:
                f.write(f"{new_id} {line.split(' ', 1)[1]}")


def make_slice(raw_dir: str | Path, out_dir: str | Path, seed: int = SEED) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sources = [f"{kind}_{fd}.txt" for fd in SIZES for kind in ("train", "test", "RUL")]
    meta = {"seed": seed, "source_sha256": _verify_raw(raw_dir, sources), "engines": {}, "files": {}}

    for fd, size in SIZES.items():
        # Train: a random sample of engines.
        train = _read_engines(raw_dir / f"train_{fd}.txt")
        chosen_train = sorted(rng.choice(sorted(train), size=size["train"], replace=False).tolist())

        # Test: guarantee some engines shorter than one window (FD001 has none), fill the rest at random.
        test = _read_engines(raw_dir / f"test_{fd}.txt")
        short = sorted(e for e, rows in test.items() if len(rows) < WINDOW)
        n_short = min(N_SHORT, len(short))
        picked_short = rng.choice(short, size=n_short, replace=False).tolist() if n_short else []
        rest = sorted(set(test) - set(picked_short))
        picked_rest = rng.choice(rest, size=size["test"] - n_short, replace=False).tolist()
        chosen_test = sorted(picked_short + picked_rest)

        # RUL file: line i belongs to original test engine i.
        with open(raw_dir / f"RUL_{fd}.txt", newline="") as f:
            rul_lines = f.readlines()

        _write_renumbered(out_dir / f"train_{fd}.txt", train, chosen_train)
        _write_renumbered(out_dir / f"test_{fd}.txt", test, chosen_test)
        with open(out_dir / f"RUL_{fd}.txt", "w", newline="") as f:
            f.writelines(rul_lines[e - 1] for e in chosen_test)

        meta["engines"][f"train_{fd}"] = chosen_train
        meta["engines"][f"test_{fd}"] = chosen_test

    for name in sources:
        meta["files"][name] = _sha256(out_dir / name)
    with open(out_dir / "SLICE.json", "w", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    m = make_slice(repo / "data" / "raw", repo / "tests" / "data")
    for k, v in m["engines"].items():
        print(f"{k}: {len(v)} engines, original ids {v}")
