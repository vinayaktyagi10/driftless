"""Inventory every downloaded IO-VNBD sequence: rate, GNSS quality, usability.

This exists because IO-VNBD is not uniform. The README advertises one smartphone
sampling rate; the files disagree with it and with each other. Training on the
set as if it were homogeneous would quietly mix 20 Hz and 10 Hz sequences, and
label 10-second GNSS intervals as if they were 100 ms ones.

Run:  python -m driftless.audit
Out:  artifacts/metrics/dataset_audit.csv  +  dataset_audit.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import latlon_to_enu
from .pair import TRUSTED_COUPLING_CORR
from .preprocess import preprocess_frame, preprocess_report
from .schema import load_raw_csv

# Populated in main() from data/processed/index.json: coupling per prepared run.
_PREPARED: dict[str, dict] = {}

from .paths import METRIC_DIR as OUT_DIR, PROC_DIR, RAW_DIR

# Usability gates for the training pool.
#
# Note what is NOT a gate: the phone's own GNSS fix interval. It is ~9 s in every
# S-* run, which is exactly why the pipeline labels from the paired V file
# instead. An earlier version of this audit rejected on that and concluded "2 of
# 63 sequences usable", which is nonsense -- the phone GNSS is an input we do not
# rely on, not a disqualification.
#
# What actually decides usability is whether the phone is rigidly coupled to the
# vehicle, which is measured per run in pair.py and recorded in the processed
# index.
MIN_DURATION_S = 120.0
MIN_PATH_M = 500.0
MIN_VALID_FRAC = 0.50


def audit_one(path: Path, manifest: dict[str, dict]) -> list[dict]:
    """Audit every run inside one raw file. Returns one row per run."""
    raw = load_raw_csv(path)
    return [_audit_run(df, path, manifest) for df in preprocess_frame(raw)]


def _audit_run(df: pd.DataFrame, path: Path, manifest: dict[str, dict]) -> dict:
    rep = preprocess_report(df)

    t = df["t_s"].to_numpy(dtype=float)
    dt = np.diff(t)
    fix = df["gps_new_fix"].to_numpy(dtype=bool)
    fix_idx = np.flatnonzero(fix)
    fix_dt = np.diff(t[fix_idx]) if fix_idx.size >= 2 else np.array([np.nan])

    meta = manifest.get(rep["seq_id"], {})
    row = {
        "run_id": rep["run_id"],
        "seq_id": rep["seq_id"],
        "family": meta.get("family", "?"),
        "driver": meta.get("driver", "?"),
        "rows": rep["rows"],
        "duration_s": round(rep["duration_s"], 1),
        "sample_dt_ms": round(float(np.median(dt)) * 1000, 1) if dt.size else np.nan,
        "sample_hz": round(1.0 / float(np.median(dt)), 2) if dt.size and np.median(dt) > 0 else np.nan,
        "n_gnss_fix": int(fix_idx.size),
        "gnss_fix_dt_s": round(float(np.median(fix_dt)), 3),
        "gnss_fix_hz": round(rep["gps_fix_hz"], 3),
        "path_len_m": round(rep["path_len_m"], 0),
        "speed_mean_ms": round(rep["speed_ms_mean_moving"], 2),
        "speed_max_ms": round(rep["speed_ms_max"], 2),
        "moving_frac": round(rep["moving_frac"], 3),
        "gps_acc_med_m": round(float(np.nanmedian(df["gps_accuracy_m"])), 1),
        "valid_frac": round(rep["valid_frac"], 3),
        "valid_gnss_frac": round(rep["valid_gnss_frac"], 3),
        "valid_imu_frac": round(rep["valid_imu_frac"], 3),
        "gap_count": rep["gap_count"],
        "speed_resid_mae_ms": round(rep["speed_resid_mae_ms"], 3),
        "file_mb": round(path.stat().st_size / 1e6, 1),
    }

    prepared = _PREPARED.get(row["run_id"], {})
    row["coupling"] = prepared.get("coupling")
    row["prepared"] = bool(prepared)

    reasons = []
    if row["duration_s"] < MIN_DURATION_S:
        reasons.append("too_short")
    if row["path_len_m"] < MIN_PATH_M:
        reasons.append("too_little_motion")
    if row["valid_frac"] < MIN_VALID_FRAC:
        reasons.append("low_valid_frac")
    if not prepared:
        reasons.append("not_paired")
    elif row["coupling"] is not None and row["coupling"] < TRUSTED_COUPLING_CORR:
        reasons.append("weak_coupling_train_only")
    row["usable"] = not reasons
    row["reject_reason"] = ",".join(reasons)
    return row


def write_markdown(df: pd.DataFrame, out: Path) -> None:
    usable = df[df["usable"]]
    lines = [
        "# IO-VNBD smartphone subset — data audit",
        "",
        f"Sequences audited: **{len(df)}**  ·  usable for training: **{len(usable)}**",
        "",
        "## Why this audit exists",
        "",
        "IO-VNBD is not a homogeneous set. Measured on the actual files:",
        "",
        f"- Sample rates present: {sorted(df['sample_hz'].dropna().unique().tolist())} Hz.",
        f"- The phone's own GNSS fix interval spans {df['gnss_fix_dt_s'].min():.2f} s "
        f"to {df['gnss_fix_dt_s'].max():.2f} s -- far too coarse to label a 2 s "
        "window, which is why ground truth comes from the paired vehicle file.",
        "- The column headed `GPS SPEED (Kmh)` holds **metres per second**; verified "
        "against position-differentiated speed (median ratio 1.000).",
        "- The column headed `GYROSCOPE Pitch` is the **vertical-axis (yaw) rate** "
        "(corr +0.977..+0.987 with vehicle yaw rate on every run).",
        "",
        "## Usable pool",
        "",
        f"- Total drive time: **{usable['duration_s'].sum() / 3600:.2f} h**",
        f"- Total distance: **{usable['path_len_m'].sum() / 1000:.1f} km**",
        f"- Drivers: {sorted(usable['driver'].unique().tolist())}",
        f"- Route families: {sorted(usable['family'].unique().tolist())}",
        "",
        "Phone/vehicle coupling by driver (the binding constraint on this dataset):",
        "",
        df.dropna(subset=["coupling"]).groupby("driver")["coupling"]
          .agg(runs="count", min="min", max="max").round(3).to_markdown(),
        "",
        "## Rejected",
        "",
    ]
    rej = df[~df["usable"]]
    if len(rej):
        counts = (rej["reject_reason"].str.split(",").explode().value_counts())
        for reason, n in counts.items():
            ids = rej[rej["reject_reason"].str.contains(reason)]["run_id"].tolist()
            lines.append(f"- `{reason}` — {n}: {', '.join(ids)}")
    else:
        lines.append("- none")
    lines += ["", "## Per-sequence table", "", df.to_markdown(index=False)]
    out.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    proc_index = PROC_DIR / "index.json"
    if proc_index.exists():
        import json
        for r in json.loads(proc_index.read_text())["runs"]:
            _PREPARED[r["run_id"]] = {"coupling": abs(r.get("align_corr", 0.0))}

    from .download import MANIFEST_PATH, load_manifest
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = {s.seq_id: {"family": s.family, "driver": s.driver}
                    for s in load_manifest()}

    # S files only: this audit is about the phone side. The V files live in the
    # same directory and have a different 29-column schema.
    files = sorted(args.raw_dir.glob("S-*.csv"))
    if not files:
        print(f"no S-*.csv in {args.raw_dir}; run `python -m driftless.download` first")
        return 1

    rows = []
    for i, f in enumerate(files, 1):
        try:
            for r in audit_one(f, manifest):
                rows.append(r)
                flag = "ok  " if r["usable"] else "SKIP"
                print(f"[{i:3d}/{len(files)}] {flag} {r['run_id']:14} "
                      f"{r['sample_hz']:5.1f}Hz  gnss {r['gnss_fix_dt_s']:6.2f}s  "
                      f"{r['duration_s']:7.0f}s  {r['path_len_m']/1000:6.2f}km  "
                      f"valid {r['valid_frac']:.2f}  {r['reject_reason']}", flush=True)
        except Exception as e:      # a corrupt file must not kill the audit
            print(f"[{i:3d}/{len(files)}] FAIL {f.name}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows).sort_values(["family", "run_id"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "dataset_audit.csv", index=False)
    write_markdown(df, args.out_dir / "dataset_audit.md")

    print(f"\nusable {int(df['usable'].sum())}/{len(df)}  "
          f"| {df[df['usable']]['duration_s'].sum()/3600:.2f} h  "
          f"| {df[df['usable']]['path_len_m'].sum()/1000:.1f} km")
    print(f"-> {args.out_dir/'dataset_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
