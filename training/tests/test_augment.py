"""Mount-rotation invariance, asserted on real recorded data.

The claim these tests defend: the five gravity-projected channels are EXACTLY
invariant to how the phone is mounted, so rotation augmentation only has to
touch the nine raw channels. If that were merely approximate, augmentation would
be injecting label-inconsistent inputs and the mount-invariant model variant
would be built on sand.
"""

import numpy as np
import pytest

from driftless_train.augment import (GYRO_XYZ_IDX, invariant_channel_indices,
                                     random_rotation, rotate_batch, rotate_window,
                                     tilt_rotation)
from driftless_train.preprocess import (DERIVED_CHANNELS, FEATURE_CHANNELS,
                                        GYRO_XYZ_COLUMNS)


def test_rotations_are_proper():
    rng = np.random.default_rng(0)
    for _ in range(20):
        for R in (random_rotation(rng), tilt_rotation(rng)):
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_tilt_rotation_keeps_the_phone_upright():
    """A dash mount does not invert the handset; up must stay up."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        R = tilt_rotation(rng, max_tilt_deg=60.0)
        # The rotated vertical axis must retain a positive vertical component.
        assert (R @ np.array([0.0, 0.0, 1.0]))[2] > np.cos(np.deg2rad(60.5))


def test_gyro_block_uses_the_physical_axis_order_not_the_label_order():
    """The vertical gyro is the 'Pitch'-labelled column; the rotation block must
    use the physical XYZ permutation, not the stored slot order."""
    f = list(FEATURE_CHANNELS)
    assert [f[i] for i in GYRO_XYZ_IDX] == list(GYRO_XYZ_COLUMNS)
    # Sanity: that really is a permutation of the three stored gyro slots.
    assert set(GYRO_XYZ_IDX) == {f.index(c) for c in
                                 ("gyro_yaw", "gyro_pitch", "gyro_roll")}


def test_rotation_leaves_derived_channels_untouched():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(len(FEATURE_CHANNELS), 20))
    R = tilt_rotation(rng)
    y = rotate_window(x, R)
    inv = invariant_channel_indices()
    assert np.array_equal(x[inv], y[inv])
    # ...and that it DID change the raw ones, or the test proves nothing.
    raw = np.setdiff1d(np.arange(len(FEATURE_CHANNELS)), inv)
    assert not np.allclose(x[raw], y[raw])


def test_rotate_batch_matches_rotate_window():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, len(FEATURE_CHANNELS), 20))
    R = tilt_rotation(rng)
    got = rotate_batch(X, R)
    for i in range(len(X)):
        assert np.allclose(got[i], rotate_window(X[i], R), atol=1e-12)


def test_derived_channels_are_invariant_on_real_data():
    """The real claim: rotate a whole recorded run's raw signal, recompute the
    derived channels through the actual preprocessing, and they must come back
    identical -- because the gravity filter is linear and commutes with R.
    """
    from driftless_train.preprocess import _add_imu_features
    from driftless_train.schema import load_raw_csv
    from driftless_train.paths import RAW_DIR

    csvs = sorted(RAW_DIR.glob("S-S1.csv")) or sorted(RAW_DIR.glob("S-*.csv"))
    if not csvs:
        pytest.skip("no raw CSVs on disk")

    df = load_raw_csv(csvs[0]).iloc[:6000].copy()
    df["t_s"] = np.arange(len(df)) * 0.1
    base = _add_imu_features(df.copy())

    rng = np.random.default_rng(7)
    R = tilt_rotation(rng, max_tilt_deg=60.0)

    rot = df.copy()
    for cols in (("acc_x", "acc_y", "acc_z"),
                 GYRO_XYZ_COLUMNS,
                 ("grav_x", "grav_y", "grav_z")):
        v = rot[list(cols)].to_numpy(dtype=float)
        # einsum rather than `v @ R.T`: on NumPy 2.2 the BLAS gemm path emits
        # spurious "divide by zero encountered in matmul" warnings from SIMD lane
        # flags. Verified harmless -- the gemm result is bit-identical to this
        # einsum and matches an explicit per-row loop to 1.8e-15, with norms
        # preserved exactly -- but an unexplained warning in a test suite is
        # worse than one extra character here.
        rot[list(cols)] = np.einsum("ij,kj->ik", v, R)
    rot = _add_imu_features(rot)

    for ch in DERIVED_CHANNELS:
        a = base[ch].to_numpy(dtype=float)
        b = rot[ch].to_numpy(dtype=float)
        scale = max(np.abs(a).max(), 1e-9)
        err = np.abs(a - b).max() / scale
        assert err < 1e-9, f"{ch} is not rotation-invariant: rel err {err:.2e}"
