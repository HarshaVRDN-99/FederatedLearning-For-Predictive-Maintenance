"""Smoke test: the package installs, imports, and the shared config holds the agreed values.

Replaced in importance by real component tests later; kept as a cheap 'is the setup broken?' check.
"""
import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]   # repo root, so the test works from any folder
SUBPACKAGES = ["data", "model", "fl", "flower_app", "partition", "eval"]


def test_package_and_subpackages_import():
    for name in ["rulfl"] + [f"rulfl.{s}" for s in SUBPACKAGES]:
        importlib.import_module(name)


def test_base_config_has_agreed_values():
    with open(ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["window"] == 30
    assert cfg["data"]["rul_cap"] == 125
    assert cfg["fl"]["K"] == 5
