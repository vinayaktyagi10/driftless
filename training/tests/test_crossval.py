"""Fold construction invariants.

`build_folds` is the only new logic in the cross-validation path that can be
wrong silently: a bad deal still produces five plausible-looking numbers. The
invariants that actually matter are (a) every route is tested exactly once and
(b) folds are balanced by DURATION, since route lengths here span 64 s to
10598 s and a count-balanced deal would put a 64 s fold next to a 3 h one.
"""

from __future__ import annotations

import pytest

from driftless_train.crossval import build_folds


def _routes(seconds: list[float]) -> dict[str, float]:
    return {f"r{i}": s for i, s in enumerate(seconds)}


def test_every_route_appears_exactly_once():
    rs = _routes([10598, 5400, 3600, 2500, 1800, 900, 600, 300, 117, 101, 64])
    folds = build_folds(rs, 5)
    flat = [r for f in folds for r in f]
    assert sorted(flat) == sorted(rs)
    assert len(flat) == len(set(flat)), "a route was dealt into two folds"


def test_fold_count_is_respected():
    for k in (2, 3, 5, 7):
        folds = build_folds(_routes([100] * 14), k)
        assert len(folds) == k
        assert all(folds), "empty fold: k too large for the route pool"


def test_snake_deal_balances_duration_better_than_naive_chunking():
    # Real trusted-pool shape: a few long drives and a tail of very short ones.
    secs = [10598, 8600, 5400, 3600, 2500, 1800, 900, 600, 300, 117, 101, 64]
    rs = _routes(secs)
    k = 5
    folds = build_folds(rs, k)
    tot = [sum(rs[r] for r in f) for f in folds]

    # Naive: sort longest-first and cut into k contiguous chunks.
    order = sorted(rs, key=lambda r: -rs[r])
    chunks = [order[i::k] for i in range(k)]  # round-robin, no snake
    naive = [sum(rs[r] for r in c) for c in chunks]

    spread = max(tot) - min(tot)
    assert spread <= max(naive) - min(naive) + 1e-9
    # And no fold may be starved to the point its median is noise.
    assert min(tot) > 0.5 * (sum(secs) / k)


def test_duration_balance_within_tolerance():
    secs = [10598, 8600, 5400, 3600, 2500, 1800, 900, 600, 300, 117, 101, 64]
    rs = _routes(secs)
    folds = build_folds(rs, 5)
    tot = [sum(rs[r] for r in f) for f in folds]
    target = sum(secs) / 5
    # Whole routes cannot be split, so exact balance is impossible; the longest
    # single route (10598 s) bounds how good any grouping can be.
    assert max(abs(t - target) for t in tot) < max(secs)


def test_deal_is_deterministic():
    rs = _routes([500, 400, 300, 200, 100, 50])
    assert build_folds(rs, 3) == build_folds(rs, 3)


def test_ties_broken_by_name_not_dict_order():
    a = build_folds({"z": 100, "a": 100, "m": 100, "b": 100}, 2)
    b = build_folds({"a": 100, "b": 100, "m": 100, "z": 100}, 2)
    assert a == b, "fold assignment depends on dict insertion order"


@pytest.mark.parametrize("k", [2, 3, 4])
def test_no_route_is_lost_for_odd_pool_sizes(k):
    for n in range(k, k * 3 + 1):
        rs = _routes([100 * (n - i) for i in range(n)])
        flat = [r for f in build_folds(rs, k) for r in f]
        assert sorted(flat) == sorted(rs), f"n={n} k={k}"
