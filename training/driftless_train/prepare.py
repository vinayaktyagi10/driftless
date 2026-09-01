"""Materialise paired runs into compact per-run arrays for training.

Stores per-SAMPLE features and targets rather than pre-cut windows, so window
length and stride stay free parameters at training time instead of being baked
into a regenerated 300 MB dataset every time we change our mind.

Run:  python -m driftless_train.prepare
Out:  data/processed/<run_id>.npz   +   data/processed/index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pair import pair_route
from .paths import PROC_DIR, RAW_DIR
from .preprocess import FEATURE_CHANNELS

TARGET_KEYS = ("speed_ms", "yaw_rate_rads", "heading_rad", "east_m", "north_m")


def available_routes(raw_dir: Path = RAW_DIR) -> list[tuple[str, Path, Path]]:
    """Every drive for which BOTH the S and V file are on disk."""
    from .download import MANIFEST_PATH, load_manifest

    out: list[tuple[str, Path, Path]] = []
    if MANIFEST_PATH.exists():
        by_route: dict[str, dict[str, str]] = {}
        for seq in load_manifest():
            by_route.setdefault(f"{seq.family}/{seq.route}", {})[seq.side] = seq.seq_id
        for route, sides in sorted(by_route.items()):
            if "S" not in sides or "V" not in sides:
                continue
            sp, vp = raw_dir / f"{sides['S']}.csv", raw_dir / f"{sides['V']}.csv"
            if sp.exists() and vp.exists():
                out.append((route, sp, vp))
    return out


def materialise_run(df: pd.DataFrame) -> dict:
    feats = df[list(FEATURE_CHANNELS)].to_numpy(dtype=np.float32)
    return {
        "features": feats,
        "speed_ms": df["tgt_speed_ms"].to_numpy(dtype=np.float32),
        "yaw_rate_rads": df["tgt_yaw_rate_rads"].to_numpy(dtype=np.float32),
        "heading_rad": df["tgt_heading_rad"].to_numpy(dtype=np.float32),
        "east_m": df["tgt_east_m"].to_numpy(dtype=np.float64),
        "north_m": df["tgt_north_m"].to_numpy(dtype=np.float64),
        "t_utc": df["t_utc"].to_numpy(dtype=np.float64),
        "valid": df["valid_all"].to_numpy(dtype=bool),
        "moving": df["tgt_moving"].to_numpy(dtype=bool),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--routes", nargs="*", default=None)
    args = ap.parse_args(argv)

    routes = available_routes(args.raw_dir)
    if args.routes:
        want = set(args.routes)
        routes = [r for r in routes if r[0] in want or r[0].split("/")[-1] in want]
    if not routes:
        print("no complete S+V pairs on disk yet; "
              "run `python -m driftless_train.download`")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (route, sp, vp) in enumerate(routes, 1):
        family = route.split("/")[0]
        try:
            paired = pair_route(sp, vp)
        except Exception as e:
            print(f"[{i:3d}/{len(routes)}] FAIL {route}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        if not paired:
            print(f"[{i:3d}/{len(routes)}] drop {route}: no run passed alignment",
                  flush=True)
            continue

        for df in paired:
            run_id = str(df["run_id"].iat[0])
            arrays = materialise_run(df)
            np.savez_compressed(args.out_dir / f"{run_id}.npz", **arrays)
            e, n = arrays["east_m"], arrays["north_m"]
            path_m = float(np.hypot(np.diff(e), np.diff(n)).sum())
            index.append({
                "run_id": run_id,
                "route": route,
                "family": family,
                "rows": int(len(arrays["valid"])),
                "duration_s": round(float(arrays["t_utc"][-1] - arrays["t_utc"][0]), 1),
                "path_len_m": round(path_m, 1),
                "valid_frac": round(float(arrays["valid"].mean()), 4),
                "moving_frac": round(float(arrays["moving"].mean()), 4),
                "align_corr": round(float(df.attrs["align_corr"]), 4),
                "align_lag_s": round(float(df.attrs["align_fine_lag_s"]), 2),
                "wheel_radius_m": round(float(df.attrs["wheel_radius_m"]), 4),
            })
            print(f"[{i:3d}/{len(routes)}] ok   {run_id:14} "
                  f"{index[-1]['duration_s']:7.0f}s "
                  f"{path_m/1000:6.2f}km corr {df.attrs['align_corr']:+.2f} "
                  f"valid {index[-1]['valid_frac']:.3f}", flush=True)

    if not index:
        print("nothing materialised")
        return 1

    idx = {"channels": list(FEATURE_CHANNELS), "targets": list(TARGET_KEYS),
           "fs_hz": 10.0, "runs": index}
    (args.out_dir / "index.json").write_text(json.dumps(idx, indent=2))

    tot_h = sum(r["duration_s"] for r in index) / 3600
    tot_km = sum(r["path_len_m"] for r in index) / 1000
    print(f"\n{len(index)} runs  |  {tot_h:.2f} h  |  {tot_km:.1f} km  "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
