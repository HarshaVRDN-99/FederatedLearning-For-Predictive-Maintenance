"""Shared test helpers. pytest loads this file automatically before any test."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLICE_DIR = REPO / "tests" / "data"      # small committed sample of the real NASA files
RAW_DIR = REPO / "data" / "raw"          # full NASA files: local only, never committed


@pytest.fixture(scope="session")
def slice_dir() -> Path:
    """Folder with train/test/RUL files for FD004 and FD001, same format as data/raw."""
    return SLICE_DIR
