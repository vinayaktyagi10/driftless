"""Clean one IO-VNBD sequence into a per-sample feature/label table.

What this stage is responsible for
---------------------------------
1. A single monotonic clock, with true sample gaps flagged (not interpolated away).
2. Attitude-invariant IMU features, because a dashboard-mounted phone sits at an
   arbitrary angle and we refuse to depend on a hand calibration.
3. GNSS-derived ground truth, aware that GPS in these files updates at ~1 Hz and
   is held constant between fixes while the IMU runs at 10 Hz.
4. Honest per-sample validity flags, so the windowing stage can drop bad spans
   instead of training on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import latlon_to_enu, wrap_deg180, wrap_pi
from .schema import DT_S_TARGET, FS_HZ_TARGET, IMU_CHANNELS, load_raw_csv

# Derived, rotation-invariant channels appended to the raw IMU channels.
DERIVED_CHANNELS: tuple[str, ...] = (
    "acc_norm",     # |a|, independent of phone orientation
    "acc_vert",     # a projected on the gravity direction (still contains g)
    "acc_horiz",    # in-plane acceleration magnitude
    "gyro_vert",    # yaw rate about TRUE vertical -- the heading-change driver
    "gyro_horiz",   # out-of-plane rotation magnitude (pitch/roll of the body)
)

FEATURE_CHANNELS: tuple[str, ...] = IMU_CHANNELS + DERIVED_CHANNELS

# Quality thresholds. Deliberately conservative: the label is GNSS, so a window
# labelled from a bad fix teaches the model the wrong thing.
MAX_DT_MS = 500.0          # a real sampling hole, not jitter
MAX_GPS_ACCURACY_M = 10.0
MIN_SATS_USED = 4
MOVING_SPEED_MS = 2.0      # below this, GNSS course-over-ground is meaningless
MAX_PLAUSIBLE_SPEED_MS = 60.0   # 216 km/h -- anything above is a glitch


MIN_RUN_ROWS = 600      # 60 s at 10 Hz; shorter fragments are not worth labelling


def split_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split one raw file into independent recording runs.

    Several IO-VNBD files are two or more logging sessions concatenated, marked
    by TIME SINCE START resetting to ~0 (S-M: 4422 s + 6176 s; S-S2: 186 s +
    9201 s). Sorting the file by t_ms instead of splitting it interleaves two
    physically separate drives, which fabricates kilometre-scale position jumps
    -- it inflated S-M's path length to 648,000 km before this was caught. So:
    never sort across a reset, cut there.
    """
    t_ms = df["t_ms"].to_numpy(dtype=float)
    resets = np.flatnonzero(np.diff(t_ms) < 0) + 1
    bounds = np.r_[0, resets, len(df)]

    runs: list[pd.DataFrame] = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b - a < MIN_RUN_ROWS:
            continue
        run = df.iloc[a:b].copy().reset_index(drop=True)
        run["run_id"] = f"{df['seq_id'].iat[0]}#{k}"
        run["run_index"] = k
        runs.append(run)
    return runs


def _add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Establish a monotonic clock in seconds and flag sampling gaps.

    Assumes `df` is a single run (see `split_runs`). Duplicate timestamps are
    dropped; genuine holes are flagged, never interpolated over.
    """
    df = df[~df["t_ms"].duplicated(keep="first")].reset_index(drop=True)

    t_ms = df["t_ms"].to_numpy(dtype=float)
    if np.any(np.diff(t_ms) < 0):
        raise ValueError(
            f"{df.get('run_id', pd.Series(['?'])).iat[0]}: t_ms still decreases; "
            "call split_runs() before preprocessing"
        )
    df["t_s"] = (t_ms - t_ms[0]) / 1000.0
    dt_ms = np.r_[np.nan, np.diff(t_ms)]
    df["dt_ms"] = dt_ms
    df["gap"] = np.r_[False, dt_ms[1:] > MAX_DT_MS]
    return df


# IO-VNBD's gyroscope axis labels do not correspond to its accelerometer's
# (X, Y, Z) order, and taking them at face value silently destroys the single
# most informative channel in the dataset.
#
# Measured against the paired vehicle CAN yaw rate on all 9 runs available:
#
#     column              corr with vehicle yaw rate
#     GYROSCOPE Yaw       -0.09 .. +0.69   (inconsistent)
#     GYROSCOPE Pitch     +0.977 .. +0.987 (consistent, every run)
#     GYROSCOPE Roll      -0.79 .. +0.59   (sign flips between runs)
#
# So the column headed "Pitch" is the vertical-axis (yaw) rate. Projecting the
# gyro triple onto the gravity vector in naive (Yaw, Pitch, Roll) order picks up
# "Roll" instead -- which is why runs appeared to have randomly-signed, weak
# phone/vehicle coupling and were being rejected as badly mounted.
#
# The accelerometer IS in (X, Y, Z) with Z vertical: its mean is ~(0.04, 0.06,
# 9.85) with |mean| = 9.85, i.e. gravity on Z.
#
# The X/Y assignment between the two remaining gyro columns is not resolved and
# does not need to be: it enters only through gyro_horiz, which is a magnitude.
GYRO_XYZ_COLUMNS: tuple[str, str, str] = ("gyro_yaw", "gyro_roll", "gyro_pitch")


def _add_imu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Project the IMU onto the gravity frame to get attitude-invariant channels.

    The vertical direction comes from a low-passed ACCELEROMETER rather than the
    GRAVITY column. On IO-VNBD the GRAVITY column is near-constant at
    (0, 0, 9.8066) -- per-axis std ~0.02 and as few as 18 distinct values in a
    whole run -- so it carries no real attitude information. The accelerometer
    does, since it includes gravity. On this dataset the projection therefore
    degenerates to selecting the vertical axis, but the machinery is what makes
    the same code correct for our own captures, where a dashboard-mounted phone
    is genuinely tilted.
    """
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyr = df[list(GYRO_XYZ_COLUMNS)].to_numpy(dtype=float)
    grv = df[["grav_x", "grav_y", "grav_z"]].to_numpy(dtype=float)

    # Low-pass the accelerometer over ~20 s to recover the gravity direction.
    n = min(len(df), 201)
    if n >= 5:
        k = np.ones(n) / n
        lp = np.stack([np.convolve(acc[:, i], k, mode="same") for i in range(3)],
                      axis=1)
    else:
        lp = grv
    g_norm = np.linalg.norm(lp, axis=1, keepdims=True)
    # Where the estimate is degenerate, fall back to the reported gravity, then
    # to +Z, so the projection stays finite. Such rows are flagged invalid anyway.
    g_hat = np.where(g_norm > 1e-3, lp / np.maximum(g_norm, 1e-9),
                     np.array([0.0, 0.0, 1.0]))

    acc_norm = np.linalg.norm(acc, axis=1)
    acc_vert = np.einsum("ij,ij->i", acc, g_hat)
    acc_horiz = np.sqrt(np.maximum(acc_norm**2 - acc_vert**2, 0.0))

    gyro_vert = np.einsum("ij,ij->i", gyr, g_hat)
    gyro_norm = np.linalg.norm(gyr, axis=1)
    gyro_horiz = np.sqrt(np.maximum(gyro_norm**2 - gyro_vert**2, 0.0))

    df["acc_norm"] = acc_norm
    df["acc_vert"] = acc_vert
    df["acc_horiz"] = acc_horiz
    df["gyro_vert"] = gyro_vert
    df["gyro_horiz"] = gyro_horiz
    return df


def _add_gnss_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ground truth from GNSS, respecting the ~1 Hz fix rate.

    Two independent speed sources are kept:
      * `gt_speed_ms`   -- the receiver's own (Doppler) speed, held between fixes.
      * `gps_speed_pos_ms` -- speed differentiated from position between *distinct*
        fixes, used only to cross-check the above.
    Disagreement between them is the cheapest available signal that a fix is bad.
    """
    lat = df["gps_lat_deg"].to_numpy(dtype=float)
    lon = df["gps_lon_deg"].to_numpy(dtype=float)
    east, north, lat0, lon0 = latlon_to_enu(lat, lon)
    df["gps_east_m"] = east
    df["gps_north_m"] = north
    df.attrs["enu_origin"] = (lat0, lon0)

    # A "new fix" is a row where the reported position actually changed.
    new_fix = np.r_[True, (np.diff(lat) != 0) | (np.diff(lon) != 0)]
    df["gps_new_fix"] = new_fix

    # No unit conversion: the column is already m/s despite its "(Kmh)" header.
    # See the note in schema.COLUMNS -- verified ratio 1.000 against position.
    df["gt_speed_ms"] = df["gps_speed_ms"].to_numpy(dtype=float)

    # Position-differentiated speed, computed only across distinct fixes and then
    # held forward, so it lives on the same 10 Hz grid as everything else.
    idx = np.flatnonzero(new_fix)
    pos_speed = np.full(len(df), np.nan)
    if idx.size >= 2:
        t = df["t_s"].to_numpy(dtype=float)
        d = np.hypot(np.diff(east[idx]), np.diff(north[idx]))
        dt = np.diff(t[idx])
        v = np.divide(d, dt, out=np.full_like(d, np.nan), where=dt > 1e-6)
        pos_speed[idx[1:]] = v
    df["gps_speed_pos_ms"] = pd.Series(pos_speed).ffill().to_numpy()

    # Course over ground. Reported in degrees; only trustworthy while moving.
    course = df["gps_orientation_deg"].to_numpy(dtype=float)
    moving = df["gt_speed_ms"].to_numpy() >= MOVING_SPEED_MS
    course_valid = np.where(moving, course, np.nan)
    course_rad = np.deg2rad(pd.Series(course_valid).ffill().bfill().to_numpy())
    # Unwrap so a window's heading change can be a plain difference.
    df["gt_heading_rad"] = np.unwrap(course_rad)
    df["gps_moving"] = moving
    return df


def _add_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample validity: is this row safe to use inside a labelled window?"""
    acc_ok = df["gps_accuracy_m"].to_numpy(dtype=float) <= MAX_GPS_ACCURACY_M
    sats_ok = df["gps_sats_used"].to_numpy(dtype=float) >= MIN_SATS_USED
    speed = df["gt_speed_ms"].to_numpy(dtype=float)
    speed_ok = np.isfinite(speed) & (speed >= 0) & (speed <= MAX_PLAUSIBLE_SPEED_MS)
    feat_ok = np.isfinite(df[list(FEATURE_CHANNELS)].to_numpy(dtype=float)).all(axis=1)
    dt_ok = ~df["gap"].to_numpy(dtype=bool)

    df["valid_gnss"] = acc_ok & sats_ok & speed_ok
    df["valid_imu"] = feat_ok & dt_ok
    df["valid"] = df["valid_gnss"] & df["valid_imu"]
    return df


def preprocess_run(run: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for ONE monotonic run."""
    df = _add_time(run)
    df = _add_imu_features(df)
    df = _add_gnss_truth(df)
    df = _add_validity(df)
    return df


def preprocess_frame(raw: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a raw file into runs and clean each one."""
    return [preprocess_run(r) for r in split_runs(raw)]


def preprocess_file(path) -> list[pd.DataFrame]:
    return preprocess_frame(load_raw_csv(path))


def preprocess_report(df: pd.DataFrame) -> dict:
    """Numbers that go straight into the data-quality section of the writeup."""
    v_doppler = df.loc[df["gps_moving"], "gt_speed_ms"].to_numpy()
    v_pos = df.loc[df["gps_moving"], "gps_speed_pos_ms"].to_numpy()
    both = np.isfinite(v_doppler) & np.isfinite(v_pos)
    resid = v_doppler[both] - v_pos[both]

    east = df["gps_east_m"].to_numpy()
    north = df["gps_north_m"].to_numpy()
    fix = df["gps_new_fix"].to_numpy()
    path_len = float(np.hypot(np.diff(east[fix]), np.diff(north[fix])).sum())

    return {
        "seq_id": df["seq_id"].iat[0],
        "run_id": str(df["run_id"].iat[0]) if "run_id" in df else df["seq_id"].iat[0],
        "rows": int(len(df)),
        "duration_s": float(df["t_s"].iat[-1]),
        "path_len_m": path_len,
        "valid_frac": float(df["valid"].mean()),
        "valid_gnss_frac": float(df["valid_gnss"].mean()),
        "valid_imu_frac": float(df["valid_imu"].mean()),
        "moving_frac": float(df["gps_moving"].mean()),
        "gps_fix_hz": float(df["gps_new_fix"].sum() / max(df["t_s"].iat[-1], 1e-9)),
        "speed_ms_mean_moving": float(np.nanmean(v_doppler)) if v_doppler.size else 0.0,
        "speed_ms_max": float(np.nanmax(df["gt_speed_ms"])),
        "speed_resid_mae_ms": float(np.abs(resid).mean()) if resid.size else np.nan,
        "speed_resid_p95_ms": float(np.percentile(np.abs(resid), 95)) if resid.size else np.nan,
        "gap_count": int(df["gap"].sum()),
    }
