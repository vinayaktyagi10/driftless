"""Reproducibility of the augmentation RNG streams.

Raised in review: `rotate_aug` and `lowpass_aug` drew from one shared generator,
consumed in an order that depended on which flags were set. Enabling the
low-pass augmentation therefore shifted the rotation draws, so the same seed did
not mean the same rotations and no two flag combinations were comparable.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftless_train.dataset import (
    OUT_WIN,
    STRIDE_TRAIN,
    WIN,
    WindowDataset,
    load_index,
    make_splits,
    split_runs,
)

ACC_NORM = 9      # rotation-INVARIANT channel: |acc|, so it identifies the tier
CUTOFFS = (4.0, 2.0, 1.0)
N_ITEMS = 24


def _train_runs(k: int = 3):
    runs, _ = load_index()
    if not runs:
        pytest.skip("no processed runs available")
    return split_runs(runs, make_splits(runs))["train"][:k]


def _tier_sequence(ds) -> list[int]:
    """Which low-pass tier each of the first N windows was drawn from.

    Identified through `acc_norm`, which a mount rotation leaves unchanged, so
    this works whether or not rotation augmentation is enabled.
    """
    seq = []
    for i in range(min(N_ITEMS, len(ds))):
        run_id, end = ds.items[i]
        got = ds[i][0].numpy()[ACC_NORM]
        cands = [ds.arrays[run_id]["features"]] + list(ds.tiers.get(run_id, []))
        errs = [np.abs(c[end - ds.win:end].T[ACC_NORM] - got).max() for c in cands]
        seq.append(int(np.argmin(errs)))
    return seq


def test_rotation_flag_does_not_shift_the_lowpass_draws():
    """The property that was broken: toggling one augmentation must not move
    the other's random draws for a fixed seed."""
    runs = _train_runs()
    kw = dict(win=WIN, stride=STRIDE_TRAIN, out_win=OUT_WIN, seed=7,
              lowpass_aug=CUTOFFS)
    plain = _tier_sequence(WindowDataset(runs, rotate_aug=False, **kw))
    rotated = _tier_sequence(WindowDataset(runs, rotate_aug=True,
                                           max_tilt_deg=60.0, **kw))
    assert plain == rotated, (
        f"enabling rotate_aug changed the low-pass tier sequence:\n"
        f"  without rotation: {plain}\n  with rotation:    {rotated}")


def test_same_seed_reproduces_the_same_tier_sequence():
    runs = _train_runs()
    kw = dict(win=WIN, stride=STRIDE_TRAIN, out_win=OUT_WIN,
              lowpass_aug=CUTOFFS, rotate_aug=True, max_tilt_deg=60.0)
    a = _tier_sequence(WindowDataset(runs, seed=11, **kw))
    b = _tier_sequence(WindowDataset(runs, seed=11, **kw))
    assert a == b, "same seed gave different tier draws"


def test_the_tier_sequence_is_not_degenerate():
    """Guards the two tests above from passing on an all-native sequence."""
    runs = _train_runs()
    seq = _tier_sequence(WindowDataset(
        runs, win=WIN, stride=STRIDE_TRAIN, out_win=OUT_WIN, seed=7,
        lowpass_aug=CUTOFFS, rotate_aug=True, max_tilt_deg=60.0))
    assert len(set(seq)) > 1, f"tier draws are degenerate: {seq}"
