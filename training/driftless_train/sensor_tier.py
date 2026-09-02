"""Does the phone-trained model transfer to a cleaner IMU tier?

Role 04/05 asked whether the edge engine can reuse this model on FOG-grade
inertial input, or whether that tier needs its own training set. The concern
raised was specific and correct in shape: decimating 200 Hz FOG to 10 Hz does not
produce phone-like 10 Hz MEMS data, it produces heavily low-passed data -- a
second distribution shift, not a fix for the first.

That is measurable without any FOG hardware, the same way the mount-rotation
shift was measured: apply the shift to held-out data and re-run the blackout
evaluation. A cleaner sensor differs from the phone mainly by having far less
broadband noise, and vehicle dynamics live below ~2 Hz while MEMS noise and
engine vibration are broadband -- so low-passing the phone channels simulates a
sensor whose noise floor is much lower, and simultaneously simulates the
anti-alias filtering that decimation would apply.

WHY ZERO-PHASE. `filtfilt` is used deliberately. A real FOG does not lag the
truth; it reports the same motion with less noise. A causal filter would add
group delay that no real sensor has, and the model would then be penalised for a
timing shift rather than for the noise change under test. This filter
manufactures a test input offline -- it is not, and must not become, part of any
online path.

WHAT THIS CANNOT SHOW. There is no FOG-grade data in IO-VNBD to validate the
simulation against. The vehicle CAN channels are not a substitute: they carry
only 3 inertial signals at 10 Hz, quantised to 0.1 deg/s and 0.0092 g, which
measurement puts BELOW one LSB of their own noise -- so their true noise density
is unmeasurable from this dataset. This experiment therefore bounds how the model
responds to losing high-frequency content. It does not certify FOG performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .dataset import DT, load_index, make_splits, split_runs
from .evaluate import (
    BLACKOUT_S,
    blackout_errors,
    load_model,
    longest_valid_span,
    window_predictions,
)
from .pair import TRUSTED_COUPLING_CORR
from .paths import METRIC_DIR, MODEL_DIR
from .preprocess import DERIVED_CHANNELS, GYRO_XYZ_COLUMNS, imu_derived
from .schema import IMU_CHANNELS

CH_IDX = {c: i for i, c in enumerate(IMU_CHANNELS + DERIVED_CHANNELS)}
ACC_IDX = [CH_IDX["acc_x"], CH_IDX["acc_y"], CH_IDX["acc_z"]]
GYR_IDX = [CH_IDX[c] for c in GYRO_XYZ_COLUMNS]
GRV_IDX = [CH_IDX["grav_x"], CH_IDX["grav_y"], CH_IDX["grav_z"]]

# Cutoffs to sweep, Hz. 5 Hz is the Nyquist of the 10 Hz stream (no-op-ish);
# 0.5 Hz is aggressive enough to remove most of what a MEMS sensor adds.
CUTOFFS_HZ = (4.0, 2.0, 1.0, 0.5)


def lowpass(x: np.ndarray, dt_s: float, cutoff_hz: float,
            order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0. See module docstring."""
    nyq = 0.5 / dt_s
    wn = cutoff_hz / nyq
    if not 0 < wn < 1:
        raise ValueError(f"cutoff {cutoff_hz} Hz outside (0, {nyq}) Hz")
    b, a = butter(order, wn, btype="low")
    # padlen must fit the shortest run; filtfilt's default can exceed it.
    padlen = min(3 * (max(len(a), len(b)) - 1), x.shape[0] - 1)
    return filtfilt(b, a, x, axis=0, padlen=max(padlen, 0))


def simulate_tier(features: np.ndarray, dt_s: float,
                  cutoff_hz: float) -> np.ndarray:
    """Return a copy of `features` as a lower-noise sensor would have recorded it.

    The raw accelerometer and gyroscope axes are low-passed and the five derived
    channels are then RE-DERIVED from them. Filtering the derived channels
    directly would be wrong: `acc_norm` is a nonlinear function of the axes, so
    filter-then-derive and derive-then-filter are different signals.
    """
    out = np.array(features, dtype=float, copy=True)
    out[:, ACC_IDX] = lowpass(out[:, ACC_IDX], dt_s, cutoff_hz)
    out[:, GYR_IDX] = lowpass(out[:, GYR_IDX], dt_s, cutoff_hz)
    derived = imu_derived(out[:, ACC_IDX], out[:, GYR_IDX], dt_s,
                          grav_fallback=out[:, GRV_IDX])
    for name, col in derived.items():
        out[:, CH_IDX[name]] = col
    return out


def hf_energy_removed(native: np.ndarray, filtered: np.ndarray,
                      idx: list[int]) -> float:
    """Fraction of variance removed from the given channels, as a noise proxy."""
    n = native[:, idx] - native[:, idx].mean(axis=0)
    f = filtered[:, idx] - filtered[:, idx].mean(axis=0)
    vn = float(np.sum(n ** 2))
    return float(1.0 - np.sum(f ** 2) / vn) if vn > 0 else float("nan")


SPEED_BANDS = ((0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30))


def vibration_speed_coupling(a: dict, win: int, out_win: int,
                             cutoff_hz: float = 2.0) -> dict:
    """Measure how strongly high-frequency energy encodes speed.

    This is the mechanism behind the degradation measured above. The band the
    low-pass removes is not just noise: on a phone in a car, road, tyre and
    engine vibration amplitude grows with speed, so that band is a genuine speed
    cue -- and a sensor- and vehicle-specific one.
    """
    F = a["features"].astype(float)
    v = a["speed_ms"].astype(float)
    valid = a["valid"].astype(bool)
    hf_acc = F[:, ACC_IDX] - lowpass(F[:, ACC_IDX], DT, cutoff_hz)
    hf_gyr = F[:, GYR_IDX] - lowpass(F[:, GYR_IDX], DT, cutoff_hz)

    rows = []
    for e in range(win, len(F), out_win):
        sl = slice(e - win, e)
        if not valid[sl].all():
            continue
        rows.append((float(np.sqrt((hf_acc[sl] ** 2).sum(axis=1).mean())),
                     float(np.sqrt((hf_gyr[sl] ** 2).sum(axis=1).mean())),
                     float(v[e - 1])))
    if len(rows) < 50:
        return {}
    r = np.array(rows)
    bands = []
    for lo, hi in SPEED_BANDS:
        m = (r[:, 2] >= lo) & (r[:, 2] < hi)
        if m.sum() > 20:
            bands.append({"lo_ms": lo, "hi_ms": hi, "n": int(m.sum()),
                          "hf_acc_rms": round(float(r[m, 0].mean()), 4),
                          "hf_gyro_rms": round(float(r[m, 1].mean()), 5)})
    return {
        "cutoff_hz": cutoff_hz, "n_windows": len(r),
        "corr_hf_acc_speed": round(float(np.corrcoef(r[:, 0], r[:, 2])[0, 1]), 3),
        "corr_hf_gyro_speed": round(float(np.corrcoef(r[:, 1], r[:, 2])[0, 1]), 3),
        "bands": bands,
    }


def _eval_arrays(model, a: dict, win: int, out_win: int, dev,
                 channels) -> dict | None:
    span = longest_valid_span(a["valid"])
    if span[1] - span[0] < win + out_win * 20:
        return None
    pr = window_predictions(model, a, win, out_win, span, dev, channels=channels)
    if not pr:
        return None
    rows = blackout_errors(pr, a)
    res = {
        "speed_mae_ms": float(np.abs(pr["pred_speed"] - pr["true_speed"]).mean()),
        "dpsi_mae_deg": float(np.rad2deg(
            np.abs(pr["pred_dpsi"] - pr["true_dpsi"]).mean())),
        "n_windows": int(len(pr["starts"])),
    }
    for T in BLACKOUT_S:
        sub = [r["model_err_m"] for r in rows if r["duration_s"] == T]
        res[f"med_{int(T)}s"] = round(float(np.median(sub)), 2) if sub else None
        res[f"n_{int(T)}s"] = len(sub)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="+", default=["test", "val"],
                    help="which held-out splits to measure on")
    ap.add_argument("--cutoffs", type=float, nargs="+", default=list(CUTOFFS_HZ))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", type=Path, default=MODEL_DIR / "tcn_best.pt")
    args = ap.parse_args(argv)

    import torch
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                       else "cpu" if args.device == "auto" else args.device)
    model, win, out_win, channels = load_model(args.ckpt, dev)

    all_runs, idx = load_index()
    coupling = {r["run_id"]: abs(r.get("align_corr", 0.0) or 0.0)
                for r in idx["runs"]}
    by_split = split_runs(all_runs, make_splits(all_runs))
    runs = [r for sp in args.splits for r in by_split.get(sp, [])
            if coupling.get(r.run_id, 0.0) >= TRUSTED_COUPLING_CORR]
    if not runs:
        print(f"no trusted runs in splits {args.splits}")
        return 1
    print(f"{len(runs)} trusted held-out run(s): "
          f"{', '.join(sorted({r.route for r in runs}))}\n")

    rows = []
    for run in runs:
        a = run.load()
        base = _eval_arrays(model, a, win, out_win, dev, channels)
        if base is None:
            continue
        rows.append({"run_id": run.run_id, "route": run.route,
                     "tier": "native", "cutoff_hz": None,
                     "hf_removed_acc": 0.0, "hf_removed_gyro": 0.0, **base})
        for hz in args.cutoffs:
            sim = dict(a)
            sim["features"] = simulate_tier(a["features"], DT, hz)
            got = _eval_arrays(model, sim, win, out_win, dev, channels)
            if got is None:
                continue
            rows.append({
                "run_id": run.run_id, "route": run.route,
                "tier": f"lowpass_{hz:g}Hz", "cutoff_hz": hz,
                "hf_removed_acc": round(hf_energy_removed(
                    a["features"], sim["features"], ACC_IDX), 4),
                "hf_removed_gyro": round(hf_energy_removed(
                    a["features"], sim["features"], GYR_IDX), 4),
                **got})

    # Paired comparison: each simulated tier is compared with the SAME route's
    # native result, so route difficulty cancels out. That matters here -- the
    # cross-validation showed per-route 30 s medians spanning a factor of 1.8.
    native = {r["run_id"]: r for r in rows if r["tier"] == "native"}
    summary = []
    for hz in args.cutoffs:
        sub = [r for r in rows if r["cutoff_hz"] == hz]
        if not sub:
            continue
        def ratio(key, sub=sub):
            vals = [r[key] / native[r["run_id"]][key] for r in sub
                    if native[r["run_id"]].get(key)]
            return round(float(np.mean(vals)), 3) if vals else None
        summary.append({
            "cutoff_hz": hz,
            "hf_removed_acc": round(float(np.mean(
                [r["hf_removed_acc"] for r in sub])), 4),
            "hf_removed_gyro": round(float(np.mean(
                [r["hf_removed_gyro"] for r in sub])), 4),
            "speed_mae_ratio": ratio("speed_mae_ms"),
            "dpsi_mae_ratio": ratio("dpsi_mae_deg"),
            "med_30s_ratio": ratio("med_30s"),
            "med_60s_ratio": ratio("med_60s"),
            "med_30s_m": round(float(np.mean(
                [r["med_30s"] for r in sub if r["med_30s"]])), 2),
        })

    # Mechanism, measured on EVERY held-out run: the correlation is
    # route-dependent (it varies with road surface), so one route would
    # misrepresent it in either direction.
    per_run_mech = []
    for run in runs:
        got = vibration_speed_coupling(run.load(), win, out_win)
        if got:
            got["route"] = run.route
            got["run_id"] = run.run_id
            per_run_mech.append(got)
    mech = {}
    if per_run_mech:
        ca = [x["corr_hf_acc_speed"] for x in per_run_mech]
        cg = [x["corr_hf_gyro_speed"] for x in per_run_mech]
        # Pool the band table across runs, weighting by window count.
        pooled = []
        for lo, hi in SPEED_BANDS:
            hits = [(b["n"], b["hf_acc_rms"], b["hf_gyro_rms"])
                    for x in per_run_mech for b in x["bands"]
                    if b["lo_ms"] == lo and b["hi_ms"] == hi]
            if not hits:
                continue
            n = sum(h[0] for h in hits)
            pooled.append({
                "lo_ms": lo, "hi_ms": hi, "n": n,
                "hf_acc_rms": round(sum(h[0] * h[1] for h in hits) / n, 4),
                "hf_gyro_rms": round(sum(h[0] * h[2] for h in hits) / n, 5)})
        mech = {
            "cutoff_hz": per_run_mech[0]["cutoff_hz"],
            "n_runs": len(per_run_mech),
            "n_windows": sum(x["n_windows"] for x in per_run_mech),
            "corr_hf_acc_speed_mean": round(float(np.mean(ca)), 3),
            "corr_hf_acc_speed_min": min(ca), "corr_hf_acc_speed_max": max(ca),
            "corr_hf_gyro_speed_mean": round(float(np.mean(cg)), 3),
            "corr_hf_gyro_speed_min": min(cg), "corr_hf_gyro_speed_max": max(cg),
            "bands": pooled, "per_run": per_run_mech,
        }

    res = {"splits": args.splits, "n_runs": len(native), "mechanism": mech,
           "routes": sorted({r["route"] for r in rows}),
           "context_s": round(win * DT, 2), "rows": rows, "summary": summary,
           "native_med_30s_m": round(float(np.mean(
               [r["med_30s"] for r in native.values() if r["med_30s"]])), 2)}
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "sensor_tier.json").write_text(json.dumps(res, indent=2))
    write_markdown(res)

    print(f"{'cutoff':>8}{'HF gone (a/g)':>18}{'speed MAE':>11}"
          f"{'dpsi':>9}{'30 s':>9}{'60 s':>9}")
    print(f"{'native':>8}{'--':>18}{'1.00x':>11}{'1.00x':>9}"
          f"{'1.00x':>9}{'1.00x':>9}   ({res['native_med_30s_m']} m)")
    for s in summary:
        print(f"{s['cutoff_hz']:7g}H"
              f"{s['hf_removed_acc'] * 100:8.1f}%/{s['hf_removed_gyro'] * 100:.1f}%"
              f"{s['speed_mae_ratio']:10.2f}x{s['dpsi_mae_ratio']:8.2f}x"
              f"{s['med_30s_ratio']:8.2f}x{s['med_60s_ratio']:8.2f}x"
              f"   ({s['med_30s_m']} m)")
    if mech:
        print(f"\nmechanism ({mech['n_runs']} runs, {mech['n_windows']} windows): "
              f"corr(HF accel RMS, speed) = "
              f"{mech['corr_hf_acc_speed_mean']:+.3f} "
              f"[{mech['corr_hf_acc_speed_min']:+.3f}, "
              f"{mech['corr_hf_acc_speed_max']:+.3f}], "
              f"gyro {mech['corr_hf_gyro_speed_mean']:+.3f}")
        for b in mech["bands"]:
            print(f"  {b['lo_ms']:2d}-{b['hi_ms']:2d} m/s  n={b['n']:5d}  "
                  f"HF acc {b['hf_acc_rms']:.4f} m/s^2  "
                  f"HF gyro {b['hf_gyro_rms']:.5f} rad/s")
    print(f"\n-> {METRIC_DIR / 'sensor_tier.md'}")
    return 0


def write_markdown(res: dict) -> None:
    L = [
        "# Sensor-tier transfer: can the phone model serve a cleaner IMU?",
        "",
        f"Measured on {res['n_runs']} trusted held-out run(s) "
        f"({', '.join(res['routes'])}) with the shipped checkpoint, unmodified. "
        f"A cleaner sensor is simulated by low-passing the raw accelerometer and "
        f"gyroscope axes and re-deriving the attitude-invariant channels — which "
        f"also simulates the anti-alias filtering that decimating a 200 Hz FOG "
        f"stream to 10 Hz would apply.",
        "",
        "Ratios are **paired per route** against that route's own native result, "
        "so route difficulty cancels; the cross-validation showed per-route 30 s "
        "medians spanning a factor of 1.8, so unpaired numbers would be noise.",
        "",
        "| tier | acc HF removed | gyro HF removed | speed MAE | Δψ MAE | "
        "30 s | 60 s |",
        "|---|---|---|---|---|---|---|",
        f"| **native phone** | — | — | 1.00× | 1.00× | 1.00× "
        f"({res['native_med_30s_m']} m) | 1.00× |",
    ]
    for s in res["summary"]:
        L.append(
            f"| low-pass {s['cutoff_hz']:g} Hz | "
            f"{s['hf_removed_acc'] * 100:.1f} % | "
            f"{s['hf_removed_gyro'] * 100:.1f} % | "
            f"{s['speed_mae_ratio']}× | {s['dpsi_mae_ratio']}× | "
            f"**{s['med_30s_ratio']}×** ({s['med_30s_m']} m) | "
            f"{s['med_60s_ratio']}× |")
    m = res.get("mechanism") or {}
    if m:
        L += [
            "",
            "## Why the speed head breaks and the heading head does not",
            "",
            f"The band the low-pass removes is not noise — it is a speed cue. "
            f"Across {m['n_runs']} held-out runs ({m['n_windows']} windows), "
            f"high-frequency energy above {m['cutoff_hz']:g} Hz correlates with "
            f"true speed at **{m['corr_hf_acc_speed_mean']:+.3f}** "
            f"(accelerometer, range {m['corr_hf_acc_speed_min']:+.3f} to "
            f"{m['corr_hf_acc_speed_max']:+.3f} by route) and "
            f"**{m['corr_hf_gyro_speed_mean']:+.3f}** (gyroscope):",
            "",
            "| speed band | n | HF accel RMS (m/s²) | HF gyro RMS (rad/s) |",
            "|---|---|---|---|",
        ]
        for b in m["bands"]:
            L.append(f"| {b['lo_ms']}–{b['hi_ms']} m/s | {b['n']} | "
                     f"{b['hf_acc_rms']} | {b['hf_gyro_rms']} |")
        L += [
            "",
            "Road, tyre and engine vibration grows with speed, so the model "
            "reads vibration amplitude as a partial speedometer. Note the cue "
            "**saturates above ~10 m/s** — it separates 0 from 10 m/s well and "
            "30 from 20 m/s barely — and its strength varies by route, which is "
            "road surface. It is a real cue, and it is specific to this sensor, "
            "mount, vehicle and surface. Yaw rate needs no such cue: heading "
            "dynamics live below 2 Hz, which is why Δψ survives every cutoff "
            "tested.",
            "",
            "**This is a limitation of the shipped model, not only an argument "
            "about the edge tier.** The cross-validation varied route and driver "
            "but held vehicle, handset and mount fixed, so it cannot see this "
            "dependence. A different car or phone is a shift the reported numbers "
            "do not cover.",
        ]

    L += [
        "",
        "## What this does and does not establish",
        "",
        "- It bounds how this architecture responds to losing high-frequency "
        "content, which is the dominant difference between a phone MEMS stream "
        "and a decimated FOG stream.",
        "- It does **not** certify FOG performance. There is no FOG-grade data in "
        "IO-VNBD to validate the simulation against.",
        "- The vehicle CAN channels are not a substitute: 3 inertial signals at "
        "10 Hz, quantised to 0.1 °/s and 0.0092 g, which measurement puts below "
        "one LSB of their own noise — so their noise density cannot be measured "
        "from this dataset at all.",
        "- `filtfilt` is zero-phase on purpose: a real cleaner sensor has no "
        "group delay, so a causal filter would penalise the model for a timing "
        "shift instead of the noise change under test. It is an offline test "
        "input generator, not an online filter.",
    ]
    (METRIC_DIR / "sensor_tier.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
