"""Pair phone sensors (S-*) with vehicle ground truth (V-*) on one clock.

Why this module is the centre of the data pipeline
--------------------------------------------------
The phone's own GNSS in these files updates only every 9 s and its position is
held constant in between -- far too coarse to label a 2 s window. The vehicle
file for the same drive carries survey GNSS at 7 decimal places, CAN velocity,
four wheel speeds and yaw rate, all at a genuine 10 Hz. In the Synchronised set
the two sides are the same length and share a clock, so V can label S
sample-for-sample.

Alignment is not assumed, it is measured:
  1. Coarse: the S file timestamps in local time, V in UTC seconds-of-day. The
     offset is recovered and snapped to a quarter-hour (covers UK/France/Nigeria
     and, later, India's +5:30).
  2. Fine: a cross-correlation of |phone gyro about vertical| against |vehicle
     yaw rate| over +/-5 s. These are the same physical quantity measured by two
     independent devices, so the peak is the true residual lag.
  3. The peak correlation is kept as an alignment-quality score, and pairs that
     score badly are rejected rather than silently trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import latlon_to_enu, wrap_pi
from .schema import load_raw_csv, load_v_csv

# Alignment search settings.
TZ_SNAP_S = 900.0        # quarter-hour: every real UTC offset is a multiple
MAX_FINE_LAG_S = 5.0
SMOOTH_N = 10            # 1 s boxcar at 10 Hz, to suppress mount vibration
# Two thresholds, because "usable at all" and "trustworthy enough to evaluate on"
# are different questions.
#
# Measured across all 72 routes with the gyro axis corrected, coupling splits
# cleanly BY DRIVER, not by route:
#   drivers A / B / D (families S, M, Y -- 8 routes):  0.96 - 1.00
#   driver  E (families Vta, Vtb, Vw, Vf -- 64 routes): 0.22 - 0.61
# Driver E's accelerometer noise is also ~2x higher (|acc| std 0.81-1.07 vs 0.52),
# and a wide +/-300 s lag search finds no better alignment -- so this is a loose
# phone, not a clock problem. Its heading labels are close to unlearnable, but its
# speed cues may not be, so we keep the runs and let the split policy confine them
# to training. Nothing weakly coupled is ever evaluated on.
MIN_COUPLING_CORR = 0.20   # below this the phone tells us nothing about the car
TRUSTED_COUPLING_CORR = 0.70  # above this a run may appear in val/test
MIN_ALIGN_CORR = MIN_COUPLING_CORR   # backwards-compatible alias

# Target plausibility gates.
MAX_SPEED_MS = 60.0
MAX_YAW_RATE_RADS = 2.0  # ~115 deg/s; beyond this is a CAN glitch
MOVING_MS = 2.0


@dataclass
class Alignment:
    """Result of aligning one S run to one V run.

    `corr` deserves care: a wide (+/-300 s) lag search on the failing runs found
    their best correlation already at lag 0, so a low value is NOT evidence of a
    clock error. It means the phone is not rigidly coupled to the vehicle on that
    drive -- loose in a cupholder or handheld -- so its gyro does not track the
    car's yaw. That run is unusable for learning vehicle dynamics from phone IMU
    regardless of how well its clocks line up. Hence `coupling_ok`, not
    `alignment_ok`.

    Measured on the pairs available: good runs score 0.67-0.79, bad ones
    0.17-0.38, with nothing in between.
    """

    tz_offset_s: float     # snapped local-time offset removed from the S clock
    residual_s: float      # what was left over before fine alignment
    fine_lag_s: float      # how many seconds the S clock runs AHEAD of V's.
                           # Subtract it from the S timestamps to align them.
    corr: float            # peak correlation of phone gyro vs vehicle yaw rate
    n_overlap: int
    turn_std_rads: float = 0.0   # how much turning there was to correlate at all

    @property
    def coupling_ok(self) -> bool:
        return abs(self.corr) >= MIN_COUPLING_CORR

    @property
    def ok(self) -> bool:
        return self.coupling_ok and self.n_overlap > 600


def sod_from_date(date_str: pd.Series) -> np.ndarray:
    """Seconds-of-day from the S file's 'YYYY-MM-DD HH-MI-SS_SSS' column."""
    txt = date_str.astype(str).str.replace(r":(\d{3})$", r".\1", regex=True)
    ts = pd.to_datetime(txt, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
            + ts.dt.microsecond / 1e6).to_numpy(dtype=float)


def split_v_runs(v: pd.DataFrame, min_rows: int = 600) -> list[pd.DataFrame]:
    """Split a V file into runs at clock decreases or holes longer than 5 s."""
    t = v["v_t_sod"].to_numpy(dtype=float)
    dt = np.diff(t)
    cuts = np.flatnonzero((dt < 0) | (dt > 5.0)) + 1
    bounds = np.r_[0, cuts, len(v)]
    return [v.iloc[a:b].copy().reset_index(drop=True)
            for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_rows]


def _smooth(x: np.ndarray, n: int = SMOOTH_N) -> np.ndarray:
    """1 s boxcar. Keeps the sign: signed turn signals give a far sharper
    correlation peak than rectified ones (0.74 vs 0.37 on route S1)."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    return np.convolve(x, np.ones(n) / n, mode="same")


def estimate_file_tz(s_all: pd.DataFrame, v_all: pd.DataFrame) -> tuple[float, float]:
    """Local-time offset between an S file and its V file, from FIRST timestamps.

    Must be computed per FILE, not per run. A run that covers only part of the V
    span has a wildly wrong median, and snapping that to a quarter-hour lands on
    a bogus offset: estimating per-run put S-M#0 at 900 s and S-M#1 at 5400 s
    when both are 3600 s, and destroyed the correlation (0.04 instead of 0.79).
    Both files start within a few seconds of each other, so the first sample is
    the reliable anchor.
    """
    sod = sod_from_date(s_all["date_str"])
    t_v = v_all["v_t_sod"].to_numpy(dtype=float)
    raw = float(sod[0] - t_v[0])
    tz = float(np.round(raw / TZ_SNAP_S) * TZ_SNAP_S)
    return tz, raw - tz


def estimate_alignment(s_run: pd.DataFrame, v_run: pd.DataFrame,
                       tz: float | None = None) -> Alignment:
    """Measure the residual lag and the phone/vehicle coupling for one run pair.

    `tz` should come from `estimate_file_tz` on the whole files. When omitted it
    is estimated from the frames given, which is only correct if they span the
    same interval.
    """
    sod_s = sod_from_date(s_run["date_str"])
    t_v = v_run["v_t_sod"].to_numpy(dtype=float)

    if tz is None:
        raw_offset = float(sod_s[0] - t_v[0])
        tz = float(np.round(raw_offset / TZ_SNAP_S) * TZ_SNAP_S)
        residual = raw_offset - tz
    else:
        residual = float(sod_s[0] - t_v[0]) - tz

    # Common time axis, S mapped into V's UTC frame.
    t_s = sod_s - tz

    # Resample both rotation signals onto a shared 10 Hz grid over the overlap.
    lo = max(np.nanmin(t_s), np.nanmin(t_v))
    hi = min(np.nanmax(t_s), np.nanmax(t_v))
    if not np.isfinite(lo) or hi - lo < 60:
        return Alignment(tz, residual, 0.0, 0.0, 0)

    grid = np.arange(lo, hi, 0.1)
    # Both are rotation rate about the vertical, measured by two independent
    # devices. Phone gyro_vert is positive clockwise (about gravity/down); the
    # vehicle CAN yaw rate is ISO (positive left), so they anti-correlate. We
    # take |corr| at the end rather than assuming which way round it is.
    a = np.interp(grid, t_s, _smooth(s_run["gyro_vert"].to_numpy(dtype=float)))
    b = np.interp(grid, t_v, _smooth(
        np.deg2rad(v_run["v_yaw_rate_dps"].to_numpy(dtype=float))))
    a = a - a.mean()
    b = b - b.mean()

    max_lag = int(MAX_FINE_LAG_S / 0.1)
    best_lag, best_c = 0, 0.0
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom > 0:
        for lag in range(-max_lag, max_lag + 1):
            aa = np.roll(a, lag)
            c = float((aa * b).sum() / denom)
            if abs(c) > abs(best_c):
                best_lag, best_c = lag, c

    # Sign convention, which a unit test pins because getting it backwards makes
    # build_paired double the misalignment instead of removing it:
    # roll(a, k) matches b when a's samples are shifted LATER by k, i.e. the S
    # clock is running k*0.1 s BEHIND. So the amount S runs ahead is -k*0.1.
    return Alignment(tz, residual, -best_lag * 0.1, best_c, len(grid),
                     turn_std_rads=float(np.std(b)))


def _yaw_rate_sign(v: pd.DataFrame) -> float:
    """Vehicle yaw rate uses ISO sign; heading is a compass bearing. Resolve it.

    Rather than hard-coding -1 we measure it per run, so a differently-wired
    logger in a future capture cannot silently invert every heading label.
    """
    t = v["v_t_sod"].to_numpy(dtype=float)
    head = np.unwrap(np.deg2rad(v["v_heading_deg"].to_numpy(dtype=float)))
    yr = np.deg2rad(v["v_yaw_rate_dps"].to_numpy(dtype=float))
    speed = v["v_speed_kmh"].to_numpy(dtype=float) / 3.6

    # Forward difference with a positive-dt mask. np.gradient divides by dt
    # directly and these files do contain repeated timestamps, which produced
    # divide-by-zero warnings and NaNs.
    dt = np.diff(t)
    dpsi = np.full_like(t, np.nan)
    good = dt > 1e-9
    dpsi[:-1][good] = np.diff(head)[good] / dt[good]

    m = (np.abs(yr) > np.deg2rad(2.0)) & (speed > MOVING_MS) & np.isfinite(dpsi)
    if m.sum() < 50:
        return -1.0
    ratio = float(np.median(dpsi[m] / yr[m]))
    return -1.0 if ratio < 0 else 1.0


def _wheel_radius_m(v: pd.DataFrame) -> float:
    """Effective rolling radius from CAN speed / wheel angular rate."""
    ws = v[["v_ws_fl_rads", "v_ws_fr_rads", "v_ws_rl_rads", "v_ws_rr_rads"]]
    omega = ws.to_numpy(dtype=float).mean(axis=1)
    speed = v["v_speed_kmh"].to_numpy(dtype=float) / 3.6
    m = (omega > 1.0) & (speed > MOVING_MS)
    if m.sum() < 50:
        return np.nan
    return float(np.median(speed[m] / omega[m]))


def build_paired(s_run: pd.DataFrame, v_run: pd.DataFrame,
                 align: Alignment) -> pd.DataFrame:
    """Interpolate V ground truth onto the S sample grid and derive dense targets."""
    sod_s = sod_from_date(s_run["date_str"])
    t_s = sod_s - align.tz_offset_s - align.fine_lag_s
    t_v = v_run["v_t_sod"].to_numpy(dtype=float)

    out = s_run.copy()
    out["t_utc"] = t_s

    sign = _yaw_rate_sign(v_run)
    radius = _wheel_radius_m(v_run)

    lat = np.interp(t_s, t_v, v_run["v_lat_deg"].to_numpy(dtype=float))
    lon = np.interp(t_s, t_v, v_run["v_lon_deg"].to_numpy(dtype=float))
    east, north, lat0, lon0 = latlon_to_enu(lat, lon)

    speed = np.interp(t_s, t_v, v_run["v_speed_kmh"].to_numpy(dtype=float)) / 3.6
    yaw = sign * np.deg2rad(
        np.interp(t_s, t_v, v_run["v_yaw_rate_dps"].to_numpy(dtype=float)))
    head = np.interp(t_s, t_v,
                     np.unwrap(np.deg2rad(v_run["v_heading_deg"].to_numpy(dtype=float))))
    omega = np.interp(t_s, t_v, v_run[["v_ws_fl_rads", "v_ws_fr_rads",
                                       "v_ws_rl_rads", "v_ws_rr_rads"]]
                      .to_numpy(dtype=float).mean(axis=1))

    out["tgt_lat"] = lat
    out["tgt_lon"] = lon
    out["tgt_east_m"] = east
    out["tgt_north_m"] = north
    out["tgt_speed_ms"] = speed
    out["tgt_yaw_rate_rads"] = yaw
    out["tgt_heading_rad"] = head
    out["tgt_wheel_speed_ms"] = omega * radius if np.isfinite(radius) else np.nan
    out["tgt_moving"] = speed > MOVING_MS

    # Outside the V time span np.interp clamps to the endpoints, which would
    # invent flat ground truth. Mark those samples unusable.
    in_span = (t_s >= t_v[0]) & (t_s <= t_v[-1])
    plausible = (np.isfinite(speed) & (speed >= 0) & (speed <= MAX_SPEED_MS)
                 & np.isfinite(yaw) & (np.abs(yaw) <= MAX_YAW_RATE_RADS))
    out["valid_target"] = in_span & plausible
    out["valid_all"] = out["valid_target"] & out["valid_imu"]

    out.attrs.update({
        "enu_origin": (lat0, lon0),
        "yaw_sign": sign,
        "wheel_radius_m": radius,
        "align_tz_offset_s": align.tz_offset_s,
        "align_fine_lag_s": align.fine_lag_s,
        "align_corr": align.corr,
        "coupling_corr": align.corr,
        "turn_std_rads": align.turn_std_rads,
    })
    return out


def match_runs(s_runs: list[pd.DataFrame], v_runs: list[pd.DataFrame],
               tz: float | None = None):
    """Match each S run to the V run it overlaps most in absolute time.

    Matching is by time overlap, not by correlation: correlation measures phone
    mount quality, so using it to choose the partner would pick the wrong V run
    for a badly mounted drive instead of reporting the drive as badly mounted.
    """
    pairs = []
    for s_run in s_runs:
        t_s = sod_from_date(s_run["date_str"]) - (tz or 0.0)
        best, best_overlap = None, 0.0
        for v_run in v_runs:
            t_v = v_run["v_t_sod"].to_numpy(dtype=float)
            overlap = min(t_s[-1], t_v[-1]) - max(t_s[0], t_v[0])
            if overlap > best_overlap:
                best, best_overlap = v_run, overlap
        if best is not None:
            pairs.append((s_run, best, estimate_alignment(s_run, best, tz)))
    return pairs


def pair_route(route_s_path: Path, route_v_path: Path) -> list[pd.DataFrame]:
    """Load one drive's S and V files and return the paired, labelled runs."""
    from .preprocess import preprocess_run, split_runs

    s_all = load_raw_csv(route_s_path)
    v_all = load_v_csv(route_v_path)
    tz, _ = estimate_file_tz(s_all, v_all)

    s_runs = [preprocess_run(r) for r in split_runs(s_all)]
    v_runs = split_v_runs(v_all)

    out = []
    for s_run, v_run, al in match_runs(s_runs, v_runs, tz):
        if not al.ok:
            continue
        out.append(build_paired(s_run, v_run, al))
    return out
