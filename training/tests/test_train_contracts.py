"""Contracts around training that review found broken.

Both were silent: nothing crashed, the numbers just quietly stopped meaning what
the files said they meant.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from driftless_train.dataset import (
    OUT_WIN,
    WIN,
    WindowDataset,
    compute_stats,
    load_index,
    make_splits,
    split_runs,
)
from driftless_train.train import FitConfig, fit


def _tiny_dataset(lowpass=(4.0, 2.0), stride=400):
    runs, _ = load_index()
    if not runs:
        pytest.skip("no processed runs available")
    train = split_runs(runs, make_splits(runs))["train"][:1]
    ds = WindowDataset(train, win=WIN, stride=stride, out_win=OUT_WIN,
                       rotate_aug=True, max_tilt_deg=60.0, seed=0,
                       lowpass_aug=lowpass)
    if len(ds) < 8:
        pytest.skip("not enough windows for a tiny fit")
    return ds


def test_compute_stats_is_not_idempotent_under_augmentation():
    """The hazard that makes the next test necessary.

    `compute_stats` draws through `__getitem__`, which consumes the augmentation
    RNGs. Calling it twice therefore returns DIFFERENT numbers -- so training
    must compute it once and pass it in, not compute it separately for the file
    and for the model.
    """
    ds = _tiny_dataset()
    a, b = compute_stats(ds), compute_stats(ds)
    drift = np.abs(np.array(a["x_mean"]) - np.array(b["x_mean"])).max()
    assert drift > 0, (
        "compute_stats appears idempotent under augmentation -- if the dataset "
        "is now deterministic, this test and the `stats=` argument to fit() "
        "can both go")


def test_fit_uses_the_stats_it_is_given():
    """What the model normalises with must be what got written to stats.json."""
    ds = _tiny_dataset()
    stats = compute_stats(ds)
    out = fit(ds, ds, n_channels=ds[0][0].shape[0],
              cfg=FitConfig(epochs=1, batch_size=8, device="cpu"), stats=stats)
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        assert np.array_equal(np.array(out["stats"][key]),
                              np.array(stats[key])), (
            f"fit() recomputed {key} instead of using the stats passed in")


def test_fit_reports_every_improvement_so_a_caller_can_persist_it():
    """An interrupted run must not lose its best weights.

    `fit` used to hold the best state in memory and hand it back only after the
    final epoch, so an interrupted 40-epoch run left no checkpoint at all --
    and crossval chains five of them.
    """
    ds = _tiny_dataset()
    seen = []

    def on_best(state_dict, epoch):
        assert isinstance(state_dict, dict) and state_dict, "empty state dict"
        assert all(isinstance(v, torch.Tensor) for v in state_dict.values())
        seen.append(epoch)

    fit(ds, ds, n_channels=ds[0][0].shape[0],
        cfg=FitConfig(epochs=2, batch_size=8, device="cpu"),
        stats=compute_stats(ds), on_best=on_best)

    assert seen, "on_best was never called, so nothing could have been saved"
    assert seen == sorted(seen), f"improvements reported out of order: {seen}"
    assert seen[0] == 1, "the first epoch always improves on +inf"
