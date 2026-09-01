"""Canonical schema for IO-VNBD smartphone (S-*) CSV files.

The raw headers are latin-1 encoded, have inconsistent leading spaces, and use
mojibake for the superscript in "m/s^2". We normalise them once, here, so no
other module has to care.

Verified against the real S-S1.csv (2019-09-08, Driver A, 9.6 MB, 24 columns).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# The README claims the smartphone logs "at 10HZ", but that is not true of every
# file: measured median sample intervals across the S-* set include 50 ms (20 Hz)
# and 100 ms (10 Hz), and the GNSS fix interval ranges from per-sample to ~9 s.
# So rate is a *per-sequence measured property*, never a constant. These are only
# the resampling target used to put every sequence on one grid.
FS_HZ_TARGET = 10.0
DT_S_TARGET = 1.0 / FS_HZ_TARGET

# Ordered canonical names, positionally matching the raw 24-column layout.
COLUMNS: tuple[str, ...] = (
    "gps_lat_deg",
    "gps_lon_deg",
    "gps_alt_m",
    # VERIFIED UNIT CORRECTION: the raw header says "GPS SPEED (Kmh)" but the
    # values are metres per second. Checked against position-differentiated speed
    # over distinct GNSS fixes on S-M / S-S1 / S-S2: median ratio 1.000 (a genuine
    # km/h column would give 3.6). Trusting the header would scale every speed
    # label by 3.6x. Named for the real unit so the mistake cannot come back.
    "gps_speed_ms",
    "gps_accuracy_m",
    "gps_orientation_deg",
    "gps_satellites",       # raw form "27 / 28" -> split into used/in_range
    "t_ms",                 # TIME SINCE START (ms), monotonic
    "date_str",
    "acc_x", "acc_y", "acc_z",          # m/s^2, includes gravity
    "grav_x", "grav_y", "grav_z",       # m/s^2, low-passed gravity estimate
    "gyro_yaw", "gyro_pitch", "gyro_roll",   # rad/s
    "mag_x", "mag_y", "mag_z",          # microtesla
    "ori_yaw_deg", "ori_pitch_deg", "ori_roll_deg",
)

# Channels fed to the network. Gravity is included so the model can infer the
# phone's mounting attitude instead of us hard-coding a calibration.
IMU_CHANNELS: tuple[str, ...] = (
    "acc_x", "acc_y", "acc_z",
    "gyro_yaw", "gyro_pitch", "gyro_roll",
    "grav_x", "grav_y", "grav_z",
)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in COLUMNS if c not in ("gps_satellites", "date_str")
)

_WS = re.compile(r"\s+")


def _norm_raw_header(h: str) -> str:
    """Collapse whitespace and lowercase a raw header for fuzzy matching."""
    return _WS.sub(" ", h.strip()).lower()


def _parse_satellites(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split the "used / in_range" satellite string into two integer series."""
    txt = series.astype(str)
    parts = txt.str.extract(r"(\d+)\s*/\s*(\d+)")
    used = pd.to_numeric(parts[0], errors="coerce")
    in_range = pd.to_numeric(parts[1], errors="coerce")
    return used, in_range


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read one IO-VNBD S-*.csv into a canonically-named DataFrame.

    Positional column mapping is used rather than name matching, because the raw
    headers contain encoding damage that varies between files. We assert the
    column count so a schema change fails loudly instead of silently shifting
    every channel by one.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        encoding="latin-1",
        engine="c",
        header=0,
        low_memory=False,
    )

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}. "
            f"Raw headers: {[_norm_raw_header(c) for c in df.columns]}"
        )

    df.columns = list(COLUMNS)

    sats_used, sats_range = _parse_satellites(df["gps_satellites"])
    df["gps_sats_used"] = sats_used
    df["gps_sats_in_range"] = sats_range
    df = df.drop(columns=["gps_satellites"])

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["seq_id"] = path.stem
    return df


def sequence_summary(df: pd.DataFrame) -> dict:
    """One-row health summary of a loaded sequence, for the data audit report."""
    t = df["t_ms"].to_numpy(dtype=float)
    dt = np.diff(t)
    lat, lon = df["gps_lat_deg"].to_numpy(), df["gps_lon_deg"].to_numpy()
    gps_moved = np.r_[False, (np.diff(lat) != 0) | (np.diff(lon) != 0)]

    return {
        "seq_id": df["seq_id"].iat[0],
        "rows": len(df),
        "duration_s": float((t[-1] - t[0]) / 1000.0) if len(t) > 1 else 0.0,
        "median_dt_ms": float(np.median(dt)) if dt.size else np.nan,
        "dt_p99_ms": float(np.percentile(dt, 99)) if dt.size else np.nan,
        "dt_gaps_gt_500ms": int((dt > 500).sum()),
        "gps_fix_updates": int(gps_moved.sum()),
        "gps_update_hz": float(gps_moved.sum() / max((t[-1] - t[0]) / 1000.0, 1e-9)),
        "gps_speed_ms_max": float(np.nanmax(df["gps_speed_ms"])),
        "gps_speed_ms_mean": float(np.nanmean(df["gps_speed_ms"])),
        "gps_accuracy_m_median": float(np.nanmedian(df["gps_accuracy_m"])),
        "nan_frac_imu": float(df[list(IMU_CHANNELS)].isna().to_numpy().mean()),
        "lat_span_deg": float(np.nanmax(lat) - np.nanmin(lat)),
        "lon_span_deg": float(np.nanmax(lon) - np.nanmin(lon)),
        "date_start": str(df["date_str"].iat[0]),
    }


# ---------------------------------------------------------------------------
# Vehicle side (V-*.csv): CAN bus + survey-grade GNSS/INS, 10 Hz, 29 columns.
# This is the ground-truth source. Note the unit conventions differ from the S
# files: here "Velocity (km/hr)" really is km/h (verified ratio 1/3.6 = 0.2778
# against position-differentiated speed), whereas the S file's "GPS SPEED (Kmh)"
# is m/s. The dataset is not internally consistent, so both are pinned by test.
# ---------------------------------------------------------------------------

V_COLUMNS: tuple[str, ...] = (
    "v_sats",
    "v_t_sod",            # Time Since Start of Day (s), UTC, monotonic
    "v_lat_deg",          # 7 decimal places -- the position ground truth
    "v_lon_deg",
    "v_speed_kmh",        # genuinely km/h
    "v_heading_deg",      # compass bearing, clockwise from north
    "v_height_km",
    "v_vert_speed_kmh",
    "v_sample_period_s",
    "v_steering_angle_deg",
    "v_ws_fl_rads", "v_ws_fr_rads", "v_ws_rl_rads", "v_ws_rr_rads",
    "v_yaw_rate_dps",     # ISO sign (positive left) -> OPPOSITE to d(heading)/dt
    "v_indicated_speed_kmh",
    "v_long_acc_g",
    "v_lat_acc_g",
    "v_handbrake",
    "v_gear_requested",
    "v_gear",
    "v_engine_rpm",
    "v_coolant_temp_c",
    "v_clutch",
    "v_brake_pressure_psi",
    "v_brake_position",
    "v_battery_v",
    "v_air_temp_c",
    "v_accel_pedal",
)


def load_v_csv(path: str | Path) -> pd.DataFrame:
    """Read one IO-VNBD V-*.csv into a canonically-named DataFrame."""
    path = Path(path)
    df = pd.read_csv(path, encoding="latin-1", engine="c", header=0, low_memory=False)
    if df.shape[1] != len(V_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(V_COLUMNS)} V columns, got {df.shape[1]}"
        )
    df.columns = list(V_COLUMNS)
    for c in V_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["v_seq_id"] = path.stem
    return df
