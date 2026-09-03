"""Test fixtures. Run from the `training/` directory: `pytest tests/ -q`."""

import sys
from pathlib import Path

import pytest

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT))

from driftless_train.paths import PROC_DIR  # noqa: E402


@pytest.fixture(scope="session")
def processed_runs():
    """Real prepared runs, or skip. Keeps the suite runnable on a clean clone."""
    files = sorted(PROC_DIR.glob("*.npz"))
    if not files:
        pytest.skip("no prepared runs; run python -m driftless_train.prepare")
    return files
