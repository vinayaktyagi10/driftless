"""Pins the dataset quirks that silently corrupt everything if they regress.

Each test here corresponds to a trap that was actually hit while building the
pipeline. They are cheap and they protect numbers that appear in the report.
"""

import numpy as np
import pandas as pd

from driftless_train import schema
from driftless_train.geo import latlon_to_enu, wrap_pi
from driftless_train.preprocess import split_runs


def test_s_speed_column_is_named_for_metres_per_second():
    """The raw header says "(Kmh)" but the values are m/s.

    Verified against position-differentiated speed on S-M / S-S1 / S-S2 (median
    ratio 1.000). If someone "fixes" this name back to kmh and divides by 3.6,
    every speed label silently shrinks 3.6x.
    """
    assert "gps_speed_ms" in schema.COLUMNS
    assert "gps_speed_kmh" not in schema.COLUMNS
    assert len(schema.COLUMNS) == 24


def test_v_speed_column_is_named_for_kmh():
    """The vehicle file's velocity really IS km/h -- opposite to the S file."""
    assert "v_speed_kmh" in schema.V_COLUMNS
    assert len(schema.V_COLUMNS) == 29


def test_no_global_sampling_rate_constant():
    """Rate is a measured per-sequence property, not an assumption."""
    assert not hasattr(schema, "FS_HZ")
    assert schema.FS_HZ_TARGET == 10.0


def _fake_raw(n_a: int, n_b: int) -> pd.DataFrame:
    """Two concatenated runs, the second restarting its clock -- as S-M does."""
    t = np.r_[np.arange(n_a) * 100, np.arange(n_b) * 100]
    df = pd.DataFrame({"t_ms": t.astype(float)})
    df["seq_id"] = "FAKE"
    return df


def test_split_runs_cuts_at_clock_reset():
    df = _fake_raw(1000, 800)
    runs = split_runs(df)
    assert len(runs) == 2
    assert [len(r) for r in runs] == [1000, 800]
    for r in runs:
        assert np.all(np.diff(r["t_ms"].to_numpy()) > 0)


def test_split_runs_drops_fragments_below_min_rows():
    df = _fake_raw(1000, 50)          # 50 rows = 5 s, not worth labelling
    runs = split_runs(df)
    assert len(runs) == 1
    assert len(runs[0]) == 1000


def test_split_runs_never_reorders_within_a_run():
    """Sorting across a reset is what produced a 648,000 km path length."""
    df = _fake_raw(600, 600)
    runs = split_runs(df)
    joined = np.concatenate([r["t_ms"].to_numpy() for r in runs])
    assert np.array_equal(joined, df["t_ms"].to_numpy())


def test_enu_is_metric_and_origin_centred():
    lat = np.array([52.4000, 52.4010])
    lon = np.array([-1.5000, -1.5000])
    e, n, lat0, lon0 = latlon_to_enu(lat, lon)
    # 0.001 deg of latitude is ~111.2 m
    assert np.isclose(n[1] - n[0], 111.2, rtol=0.01)
    assert np.allclose(e, e[0])
    assert np.isclose((n[0] + n[1]) / 2, 0.0, atol=1e-6)


def test_enu_east_uses_cos_latitude():
    lat = np.full(2, 52.4)
    lon = np.array([-1.5000, -1.4990])
    e, n, *_ = latlon_to_enu(lat, lon)
    expected = 111_320 * 0.001 * np.cos(np.deg2rad(52.4))
    assert np.isclose(e[1] - e[0], expected, rtol=0.01)


def test_wrap_pi_range():
    a = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi])
    w = wrap_pi(a)
    assert np.all(w > -np.pi - 1e-9) and np.all(w <= np.pi + 1e-9)


def test_gyro_vertical_axis_is_the_pitch_labelled_column():
    """IO-VNBD's gyro axis labels do not match its accelerometer's XYZ order.

    Measured against paired vehicle CAN yaw rate on every available run, the
    column headed "GYROSCOPE Pitch" is the vertical-axis rate (corr +0.977 to
    +0.987, consistently). "Yaw" and "Roll" are inconsistent and sign-unstable.
    Reverting this mapping feeds the model noise in place of its single most
    informative channel, and makes good runs look like badly-mounted ones.
    """
    from driftless_train.preprocess import GYRO_XYZ_COLUMNS
    assert GYRO_XYZ_COLUMNS[2] == "gyro_pitch", (
        "the vertical (Z) gyro component must be the 'Pitch'-labelled column")
    assert set(GYRO_XYZ_COLUMNS) == {"gyro_yaw", "gyro_pitch", "gyro_roll"}


def test_gravity_comes_from_a_causal_lowpass_of_the_accelerometer():
    """Attitude must come from a CAUSAL low-pass of the accelerometer.

    Two separate requirements, both previously violated:
      * not the GRAVITY column -- it is near-constant (0, 0, 9.8066) on this
        dataset and carries no attitude information;
      * not a centred filter -- the original np.convolve(..., mode="same")
        averaged ~10 s of FUTURE samples into the current row, which breaks the
        causality guarantee the windowing relies on and cannot be reproduced on
        a phone in real time.
    """
    import inspect

    from driftless_train import preprocess

    # The derived channels are computed by `imu_derived`, which
    # `_add_imu_features` delegates to; check the whole chain so moving the maths
    # between them cannot quietly drop the causal estimator.
    src = (inspect.getsource(preprocess._add_imu_features)
           + inspect.getsource(preprocess.imu_derived))
    assert "_causal_gravity" in src, "attitude must use the causal estimator"
    assert 'mode="same"' not in src, "centred filter reintroduced -- non-causal"

    helper = inspect.getsource(preprocess._causal_gravity)
    assert "lfilter" in helper, "expected a one-pole recursive (causal) filter"


def test_causal_gravity_uses_no_future_samples():
    """Behavioural, not textual: perturbing a future sample must not change the
    gravity estimate at an earlier index."""
    import numpy as np

    from driftless_train.preprocess import _causal_gravity

    rng = np.random.default_rng(0)
    acc = rng.normal(size=(500, 3)) + np.array([0.0, 0.0, 9.81])
    base = _causal_gravity(acc, dt_s=0.1)

    bumped = acc.copy()
    bumped[300:] += 50.0
    after = _causal_gravity(bumped, dt_s=0.1)

    assert np.allclose(base[:300], after[:300], atol=1e-12), \
        "gravity estimate at t depends on samples after t"
    assert not np.allclose(base[300:], after[300:]), "perturbation had no effect"


def test_causal_gravity_commutes_with_rotation():
    """Linearity is what makes the derived channels exactly mount-invariant."""
    import numpy as np

    from driftless_train.augment import tilt_rotation
    from driftless_train.preprocess import _causal_gravity

    rng = np.random.default_rng(1)
    acc = rng.normal(size=(400, 3)) + np.array([0.0, 0.0, 9.81])
    R = tilt_rotation(rng)

    lp_then_rot = np.einsum("ij,kj->ik", _causal_gravity(acc, 0.1), R)
    rot_then_lp = _causal_gravity(np.einsum("ij,kj->ik", acc, R), 0.1)
    assert np.abs(lp_then_rot - rot_then_lp).max() < 1e-9
