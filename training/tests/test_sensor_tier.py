"""Guards on the simulated cleaner-sensor tier.

The experiment's conclusion rests on the simulation being what it claims: a
low-pass that removes high-frequency content and nothing else, with the derived
channels RE-DERIVED rather than filtered. Both are easy to get subtly wrong and
neither would announce itself in the output numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftless_train.dataset import DT
from driftless_train.preprocess import DERIVED_CHANNELS, imu_derived
from driftless_train.schema import IMU_CHANNELS
from driftless_train.sensor_tier import (
    ACC_IDX,
    CH_IDX,
    GYR_IDX,
    lowpass,
    simulate_tier,
)

N_CH = len(IMU_CHANNELS) + len(DERIVED_CHANNELS)


def _fake(n=2000, seed=0):
    """Slow motion (0.3 Hz) plus broadband noise, plus gravity on z."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    F = np.zeros((n, N_CH))
    slow = np.sin(2 * np.pi * 0.3 * t)
    for k, i in enumerate(ACC_IDX):
        F[:, i] = slow * (1.0 + k) + rng.normal(0, 0.5, n)
    F[:, ACC_IDX[2]] += 9.80665
    for i in GYR_IDX:
        F[:, i] = 0.2 * slow + rng.normal(0, 0.05, n)
    F[:, CH_IDX["grav_z"]] = 9.80665
    d = imu_derived(F[:, ACC_IDX], F[:, GYR_IDX], DT)
    for name, col in d.items():
        F[:, CH_IDX[name]] = col
    return F


def test_lowpass_attenuates_high_frequency_and_keeps_low():
    n = 4000
    t = np.arange(n) * DT
    lo = np.sin(2 * np.pi * 0.2 * t)          # well inside the passband
    hi = np.sin(2 * np.pi * 4.5 * t)          # well inside the stopband
    x = (lo + hi).reshape(-1, 1)
    y = lowpass(x, DT, 1.0)[:, 0]
    # Ignore edges: filtfilt padding makes the first/last samples unreliable.
    core = slice(200, n - 200)
    assert np.std(y[core] - lo[core]) < 0.15, "passband content was distorted"
    assert np.std(y[core]) < np.std(x[core, 0]), "no attenuation at all"


def test_lowpass_is_zero_phase():
    """A real cleaner sensor has no group delay, so the filter must not add one."""
    n = 3000
    t = np.arange(n) * DT
    x = np.sin(2 * np.pi * 0.3 * t).reshape(-1, 1)
    y = lowpass(x, DT, 1.5)[:, 0]
    core = slice(300, n - 300)
    # Cross-correlate over +-10 samples; the peak must be at zero lag.
    lags = range(-10, 11)
    best = max(lags, key=lambda k: float(
        np.dot(x[core, 0], np.roll(y, k)[core])))
    assert best == 0, f"filter introduced a {best}-sample group delay"


@pytest.mark.parametrize("cutoff", [4.0, 2.0, 1.0])
def test_simulate_tier_rederives_and_does_not_filter_derived(cutoff):
    F = _fake()
    out = simulate_tier(F, DT, cutoff)
    assert out.shape == F.shape

    # Derived channels must equal imu_derived of the FILTERED raw axes -- not the
    # filtered version of the original derived channels. acc_norm is nonlinear,
    # so the two differ and confusing them would silently change the experiment.
    expect = imu_derived(out[:, ACC_IDX], out[:, GYR_IDX], DT,
                         grav_fallback=out[:, [CH_IDX[c] for c in
                                               ("grav_x", "grav_y", "grav_z")]])
    for name in DERIVED_CHANNELS:
        assert np.allclose(out[:, CH_IDX[name]], expect[name], atol=1e-12), name

    wrong = lowpass(F[:, [CH_IDX["acc_norm"]]], DT, cutoff)[:, 0]
    core = slice(200, len(F) - 200)
    assert not np.allclose(out[core, CH_IDX["acc_norm"]], wrong[core], atol=1e-6), \
        "acc_norm looks filtered rather than re-derived"


def test_simulate_tier_removes_energy_monotonically_with_cutoff():
    F = _fake()
    core = slice(200, len(F) - 200)

    def hf(arr):
        x = arr[core][:, ACC_IDX]
        return float(np.var(x - x.mean(axis=0)))

    e = [hf(simulate_tier(F, DT, c)) for c in (4.0, 2.0, 1.0, 0.5)]
    assert all(a > b for a, b in zip(e, e[1:], strict=False)), \
        f"variance not monotonically decreasing with cutoff: {e}"
    assert e[0] < hf(F), "the loosest cutoff removed nothing"


def test_simulate_tier_leaves_untouched_channels_alone():
    F = _fake()
    out = simulate_tier(F, DT, 2.0)
    for name in ("grav_x", "grav_y", "grav_z"):
        assert np.allclose(out[:, CH_IDX[name]], F[:, CH_IDX[name]]), name


def test_simulate_tier_does_not_mutate_its_input():
    F = _fake()
    before = F.copy()
    simulate_tier(F, DT, 1.0)
    assert np.array_equal(F, before), "simulate_tier mutated the caller's array"


def test_cutoff_must_be_below_nyquist():
    F = _fake(n=500)
    with pytest.raises(ValueError):
        simulate_tier(F, DT, 5.0)     # exactly Nyquist
    with pytest.raises(ValueError):
        simulate_tier(F, DT, 9.0)     # above Nyquist
