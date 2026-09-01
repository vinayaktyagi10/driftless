"""The S<->V alignment must recover a known offset and lag from synthetic data."""

import numpy as np
import pandas as pd
import pytest

from driftless_train.pair import (
    MIN_ALIGN_CORR,
    estimate_alignment,
    sod_from_date,
    split_v_runs,
)


def _make_pair(tz_offset_s: float, lag_s: float, n: int = 3000, fs: float = 10.0):
    """A shared turn signal, sampled by a 'phone' and a 'vehicle' with a lag."""
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    turn = np.sin(2 * np.pi * t / 40.0) * 0.4            # slow weaving
    turn += rng.normal(0, 0.01, n)

    base = 32000.0
    v = pd.DataFrame({
        "v_t_sod": base + t,
        # ISO sign: vehicle yaw rate is the negative of the phone's
        "v_yaw_rate_dps": np.rad2deg(-turn),
        "v_heading_deg": np.rad2deg(np.cumsum(turn) / fs) % 360,
        "v_speed_kmh": np.full(n, 36.0),
        "v_ws_fl_rads": np.full(n, 36.0),
        "v_ws_fr_rads": np.full(n, 36.0),
        "v_ws_rl_rads": np.full(n, 36.0),
        "v_ws_rr_rads": np.full(n, 36.0),
        "v_lat_deg": 52.4 + np.zeros(n),
        "v_lon_deg": -1.5 + np.zeros(n),
    })

    # Phone timestamps: local time = UTC + tz, and its samples arrive `lag_s` late.
    sod = base + tz_offset_s + t + lag_s
    ts = pd.to_datetime(sod, unit="s", origin=pd.Timestamp("2019-09-08"))
    s = pd.DataFrame({
        "date_str": ts.strftime("%Y-%m-%d %H:%M:%S:") + (
            (ts.microsecond // 1000).astype(str).str.zfill(3)),
        "gyro_vert": turn,
    })
    return s, v


@pytest.mark.parametrize("tz,lag", [(3600.0, 0.0), (3600.0, 0.5),
                                    (7200.0, -0.3), (19800.0, 0.2)])
def test_alignment_recovers_offset_and_lag(tz, lag):
    s, v = _make_pair(tz, lag)
    al = estimate_alignment(s, v)
    assert al.tz_offset_s == pytest.approx(tz, abs=1e-6)
    assert al.fine_lag_s == pytest.approx(lag, abs=0.15)
    assert abs(al.corr) > 0.9
    assert al.ok


def test_alignment_rejects_unrelated_signals():
    s, v = _make_pair(3600.0, 0.0)
    rng = np.random.default_rng(1)
    s = s.copy()
    s["gyro_vert"] = rng.normal(0, 0.4, len(s))     # no shared motion at all
    al = estimate_alignment(s, v)
    assert abs(al.corr) < MIN_ALIGN_CORR
    assert not al.ok


def test_sod_from_date_parses_the_millisecond_colon_format():
    sr = pd.Series(["2019-09-08 10:07:49:546"])
    sod = sod_from_date(sr)
    assert sod[0] == pytest.approx(10 * 3600 + 7 * 60 + 49.546, abs=1e-3)


def test_split_v_runs_cuts_on_large_time_holes():
    n = 1500
    t = np.arange(n) / 10.0
    t[800:] += 60.0                                  # a one-minute hole
    v = pd.DataFrame({"v_t_sod": t})
    runs = split_v_runs(v, min_rows=100)
    assert len(runs) == 2
    assert [len(r) for r in runs] == [800, 700]
