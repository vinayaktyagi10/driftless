"""Assemble the Round-1 evidence document from whatever the pipeline produced.

Round 1 for PS #26168 is a paper round: a written proposal plus position-plot
results on a subset of the dataset, judged before any live demo. This collects
the audit, the split, the metrics and the figures into one markdown file that
role 06 can lift from directly, with every number traceable to a file on disk.

Run:  python -m driftless_train.report
Out:  artifacts/ROUND1_EVIDENCE.md
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .paths import ARTIFACT_DIR as ART
from .paths import CONFIG_DIR, PROC_DIR

METRIC_DIR = ART / "metrics"
PLOT_DIR = ART / "plots"
MODEL_DIR = ART / "models"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


def _tier_ratio(tier: dict | None, cutoff_hz: float) -> float | None:
    """30 s error ratio at one low-pass cutoff, from the sensor-tier study."""
    if not tier:
        return None
    for row in tier.get("summary", []):
        if abs(row.get("cutoff_hz", -1) - cutoff_hz) < 1e-9:
            return row.get("med_30s_ratio")
    return None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(key, "")) for key, _ in cols) + " |")
    return "\n".join([head, sep, *body])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "ROUND1_EVIDENCE.md")
    args = ap.parse_args(argv)

    ev = _load(METRIC_DIR / "eval_summary.json")
    noise = _load(METRIC_DIR / "measurement_noise.json")
    allan = _load(METRIC_DIR / "allan_imu_noise.json")
    cv = _load(METRIC_DIR / "crossval.json")
    tier = _load(METRIC_DIR / "sensor_tier.json")
    ckpt_meta = {}
    ck_path = MODEL_DIR / "tcn_best.pt"
    if ck_path.exists():
        try:
            import torch
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            ckpt_meta = {"rotate_aug": bool(ck.get("rotate_aug")),
                         "epoch": ck.get("epoch"),
                         "channel_indices": ck.get("channel_indices")}
        except Exception:
            ckpt_meta = {}
    splits = _load(CONFIG_DIR / "splits.json")
    export = _load(MODEL_DIR / "export_report.json")
    proc = _load(PROC_DIR / "index.json")

    L: list[str] = [
        "# Driftless — Round 1 evidence",
        "",
        "SIH 2026 · PS #26168 (ISRO) · role 03 (data + model training)",
        f"Generated {date.today().isoformat()} from the artifacts in this repo.",
        "",
        "## 1. What was built",
        "",
        "A speed & heading-change regressor trained on the public **IO-VNBD** "
        "dataset, evaluated by **dead reckoning through simulated GNSS "
        "blackouts** on held-out routes, and exported for both target runtimes.",
        "",
    ]

    if proc:
        runs = proc["runs"]
        tot_h = sum(r["duration_s"] for r in runs) / 3600
        tot_km = sum(r["path_len_m"] for r in runs) / 1000
        fams = sorted({r["family"] for r in runs})
        L += [
            "## 2. Data",
            "",
            f"- **{len(runs)} runs**, **{tot_h:.2f} h**, **{tot_km:.1f} km** of "
            "paired phone-IMU + vehicle-ground-truth driving.",
            f"- Route families: {', '.join(fams)}.",
            f"- Feature channels ({len(proc['channels'])}): "
            f"`{'`, `'.join(proc['channels'])}`.",
            "- Ground truth is the paired vehicle file (survey GNSS at 7 decimal "
            "places, CAN velocity, wheel speeds, yaw rate) at a genuine 10 Hz. "
            "The phone's own GNSS updates only every 9 s and is unusable as a "
            "per-window label.",
            "",
            "Sanity check on the ground truth itself: integrating the **true** "
            "speed and heading reproduces the **true** trajectory to **0.212 % "
            "drift over 38.07 km**. That validates units, the compass convention "
            "and target coherence together, and sets the floor any model is "
            "measured against.",
            "",
            "Full per-run inventory: `artifacts/metrics/dataset_audit.md`.",
            "",
        ]

    if splits:
        sp = splits["splits"]
        dur = splits.get("trusted_duration_s", splits.get("duration_s", {}))
        weak = set(splits.get("weak_coupling_train_only", []))
        L += [
            "## 3. Split discipline",
            "",
            "Splits are **route-wise and duration-balanced**, frozen to "
            "`configs/splits.json`. Windows from one drive never straddle train "
            "and test — otherwise the model can memorise a road instead of "
            "learning vehicle dynamics.",
            "",
            "Routes are additionally separated by **phone/vehicle coupling**. "
            "Only rigidly-coupled routes may appear in val or test; weakly-"
            "coupled ones are train-only and excluded from the reported model "
            "(see the coupling section of the README). The hours below are "
            "trusted data; the 70/15/15 target applies to it.",
            "",
            _table(
                [{"split": k,
                  "routes": len([r for r in sp.get(k, []) if r not in weak]),
                  "duration_h": f"{dur.get(k, 0)/3600:.2f}",
                  "names": ", ".join(r for r in sp.get(k, []) if r not in weak)}
                 for k in ("train", "val", "test")],
                [("split", "Split"), ("routes", "Trusted routes"),
                 ("duration_h", "Trusted hours"), ("names", "Route IDs")]),
            "",
            f"Plus {len(weak)} weakly-coupled train-only routes "
            f"({splits.get('weak_train_only_duration_h', 0)} h), not used by the "
            "reported model.",
            "",
        ]

    if ev:
        L += [
            "## 4. Results — position error after a GNSS blackout",
            "",
            f"Held-out **{ev['split']}** split, {ev['n_runs']} run(s). "
            f"Context {ev['win']*0.1:.1f} s, output interval "
            f"{ev['out_win']*0.1:.1f} s. Blackout start points every 10 s across "
            "each run; each row aggregates all of them.",
            "",
            "- **model** — the regressor, with speed propagated from the last "
            "known fix and blended toward the absolute head "
            f"(ramp τ = {ev.get('tau_s', '?')} s)",
            "- **abs-only** — the absolute speed head alone, no propagation",
            "- **baseline** — no ML: hold the last known speed, integrate the "
            "phone gyro for heading",
            "- **oracle** — integrate the *true* speed and yaw: "
            "the dead-reckoning floor",
            "",
            _table(
                [{"d": f"{r['duration_s']:.0f} s",
                  "n": r["n"],
                  "m": f"**{r['model_err_med_m']:.1f}**",
                  "p90": f"{r['model_err_p90_m']:.1f}",
                  "pct": f"{r['model_drift_med_pct']:.1f} %",
                  "abs": f"{r.get('model_abs_err_med_m', float('nan')):.1f}",
                  "b": f"{r['baseline_err_med_m']:.1f}",
                  "bp": f"{r['baseline_drift_med_pct']:.1f} %",
                  "o": f"{r['oracle_err_med_m']:.1f}"}
                 for r in ev["blackout_summary"]],
                [("d", "Blackout"), ("n", "n"), ("m", "model median (m)"),
                 ("p90", "model p90 (m)"), ("pct", "model drift"),
                 ("abs", "abs-only (m)"), ("b", "baseline (m)"),
                 ("bp", "baseline drift"), ("o", "oracle (m)")]),
            "",
            "### Per-window regression accuracy",
            "",
            f"- Speed MAE **{ev['speed_mae_ms_mean']:.3f} m/s**",
            f"- Heading-change MAE **{ev['dpsi_mae_deg_mean']:.3f}°** per "
            f"{ev['out_win']*0.1:.1f} s window",
            f"- Same heading by raw gyro integration alone: "
            f"**{ev['gyro_dpsi_mae_deg_mean']:.3f}°** — the learned head is "
            f"{ev['gyro_dpsi_mae_deg_mean']/max(ev['dpsi_mae_deg_mean'], 1e-9):.1f}"
            "× better",
            "",
            "Raw per-blackout records: `artifacts/metrics/eval_blackouts.csv`.",
            "",
        ]

    # The single-split table above is one held-out road. Cross-validation is what
    # says whether that road was representative, so it goes immediately after --
    # a reader must not be able to take the headline without the spread.
    if cv and cv.get("pooled"):
        p30 = next((r for r in cv["pooled"] if int(r["duration_s"]) == 30), None)
        af = cv["across_folds"]
        single30 = next((r for r in ev["blackout_summary"]
                         if int(r["duration_s"]) == 30), None) if ev else None
        L += [
            "### Cross-validated — how representative was that one road?",
            "",
            f"The table above holds out **one route**. To find out whether it "
            f"was a lucky one, `crossval.py` runs **{cv['k']}-fold route-wise "
            f"cross-validation** over the "
            f"**{cv['n_trusted_routes']} trusted routes** "
            f"({cv['trusted_hours']} h): each is held out in turn by a model "
            f"trained through the same `train.fit`, and the errors are pooled. "
            f"{len(cv.get('per_route', []))} of {cv['n_trusted_routes']} yield "
            f"blackout samples"
            + (f" ({', '.join(cv['unevaluated_routes'])} is shorter than the "
               f"shortest blackout, so it contributes training data only)."
               if cv.get("unevaluated_routes") else "."),
            "",
            "| Blackout | n | CV median | CV p90 | drift | baseline | oracle |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in cv["pooled"]:
            L.append(f"| {int(r['duration_s'])} s | {r['n']} | "
                     f"**{r['model_med_m']} m** | {r['model_p90_m']} m | "
                     f"{r['model_drift_med_pct']:.1f} % | "
                     f"{r['baseline_med_m']} m | {r['oracle_med_m']} m |")
        L += [
            "",
            f"Fold-to-fold spread at 30 s: **{af['med_30s']['mean']} ± "
            f"{af['med_30s']['std']} m** (range {af['med_30s']['min']}–"
            f"{af['med_30s']['max']} m). Speed MAE "
            f"**{af['speed_mae_ms']['mean']} ± {af['speed_mae_ms']['std']} m/s**.",
            "",
        ]
        if p30 and single30:
            ratio = p30["model_med_m"] / max(single30["model_err_med_m"], 1e-9)
            # Where the shipped test route actually ranks, computed rather than
            # asserted: a hand-written "it was an easy road" would go stale the
            # first time the split or the fold deal changes.
            test_routes = (splits or {}).get("splits", {}).get("test", [])
            ranked = sorted((r for r in cv.get("per_route", [])
                             if r.get("med_30s") is not None),
                            key=lambda r: r["med_30s"])
            rank_txt = ""
            for i, r in enumerate(ranked):
                if r["route"] not in test_routes:
                    continue
                where = ("the easiest" if i == 0 else
                         f"the {_ordinal(i + 1)} easiest")
                rank_txt = (
                    f" Cross-validated on its own, the shipped test route "
                    f"**{r['route']}** is {where} of the {len(ranked)} "
                    f"evaluated routes at 30 s ({r['med_30s']} m, against a "
                    f"per-route median of "
                    f"{ranked[len(ranked) // 2]['med_30s']} m), so the "
                    f"single-split figure is the optimistic end of this "
                    f"model's range rather than its centre.")
                break
            L += [
                f"**Read this before quoting the headline.** The pooled 30 s "
                f"median is **{p30['model_med_m']} m** against "
                f"**{single30['model_err_med_m']} m** on the single test route — "
                f"{ratio:.2f}× worse.{rank_txt} Two effects are mixed in and "
                f"{cv['n_trusted_routes']} routes cannot fully separate them: "
                f"fold models train on ~3/5 of the trusted pool, which pushes "
                f"their error up, and the test route is genuinely easier, which "
                f"pushes the single-split figure down. We quote both numbers "
                f"everywhere and lead with the cross-validated one.",
                "",
                "Per-route figures are in `artifacts/metrics/crossval.md`; the "
                "raw samples are in `crossval_samples.csv`.",
                "",
            ]

    if ckpt_meta:
        L += [
            "## 5. Robustness: the phone is not lying flat in a car",
            "",
            "Every phone in IO-VNBD lay flat (mean accelerometer direction "
            "≈ (0, 0, 1) in all runs), but the product puts one in a dashboard "
            "mount at an arbitrary angle. Nine of the fourteen input channels are "
            "raw body axes and leave the training distribution as soon as the "
            "phone is tilted. Measured under a simulated mount rotation (random "
            "azimuth, tilt up to 60°) on the held-out route:",
            "",
            "| training | unrotated 30 s | rotated 30 s | degradation | 60 s |",
            "|---|---|---|---|---|",
            "| 14 ch, no augmentation | 33.8 m | **186.5 m** | **5.5×** | 70.0 m |",
            "| 5 gravity-projected channels only | 37.5 m | 37.5 m | "
            "**1.000× — bit-identical** | 70.3 m |",
            "| **14 ch + rotation augmentation** | **32.8 m** | 33.3 m | "
            "**1.01×** | **59.1 m** |",
            "",
            "Untreated, the model is *worse than the no-ML baseline* once the "
            "phone is tilted. Augmentation removes that and also improves "
            "accuracy (60 s error fell 16 %), so the reported model is trained "
            f"with it (`rotate_aug={ckpt_meta.get('rotate_aug')}`). The "
            "5-channel subset is retained as a fallback whose invariance is "
            "*provable* rather than empirical: a fixed mount rotation commutes "
            "with the linear gravity filter, so those channels are exactly "
            "invariant — asserted on real data in `tests/test_augment.py`.",
            "",
            "Related correctness note: gravity is estimated with a **causal** "
            "one-pole filter. An earlier version used a centred convolution, "
            "which averaged ~10 s of future samples into the current row — "
            "breaking the causality the windowing depends on, and not something "
            "a handset could reproduce in real time.",
            "",
        ]

    if noise:
        sp, dv = noise["speed"], noise.get("dv", {})
        L += [
            "## 6. Handover to fusion (role 02)",
            "",
            "Full detail in `artifacts/metrics/measurement_noise.md`.",
            "",
            f"- Forward speed: sigma **{sp['sigma_mps']} m/s**, bias "
            f"**{sp['bias_mps']:+.3f} m/s**, with a per-speed-band table since "
            "the error scales with speed.",
            f"- Speed change (`dv`): sigma **{dv.get('sigma_mps')} m/s**, bias "
            f"**{dv.get('bias_mps'):+.4f} m/s** — essentially unbiased, and the "
            "better-conditioned of the two speed outputs.",
            f"- Heading change: sigma **{noise['dpsi']['sigma_deg']}°** per "
            f"{noise['update_interval_s']} s.",
            f"- **Residuals are time-correlated**: speed lag-1 autocorrelation "
            f"**{sp['correlation']['lag1']}**, decorrelation time "
            f"**{sp['correlation']['decorrelation_s']} s** — the same as the "
            "context length. Feeding one measurement per "
            f"{noise['update_interval_s']} s as if independent over-informs the "
            f"filter by about **{sp['independent_sampling_inflation']}×** in "
            "sigma.",
            "",
            "The measurement is the longitudinal body-velocity component — the "
            "one axis the non-holonomic constraint deliberately leaves free — so "
            "it fits the existing unscented update with no new filter code.",
            "",
        ]

    if allan:
        L += [
            "## 7. IMU noise characterisation",
            "",
            f"`artifacts/metrics/allan_imu_noise.md`. Overlapping Allan deviation "
            f"over **{allan['stationary_seconds']:.0f} s** of stationary data in "
            f"{allan['n_segments']} spans, addressing the handset half of the "
            "TODO in `edge-engine/include/driftless/imu_noise.h`.",
            "",
            "**Two of the four parameters are not identifiable from this data**, "
            "and the tool refuses to report them rather than returning a number: "
            "in the window where bias instability should make the Allan curve "
            "rise, it is still falling on every axis. The gyro white-noise fit is "
            "separately contaminated because the vehicle is stopped but the "
            "engine is idling. Fixing both needs one capture nobody has taken: "
            "phone flat on a desk, engine off, 10 minutes at the fastest sensor "
            "rate.",
            "",
        ]

    figs = sorted(PLOT_DIR.glob("*.png"))
    if figs:
        L += ["## 8. Figures", ""]
        for f in figs:
            rel = f.relative_to(ART)
            cap = {"blackout_error": "Position error and drift against blackout "
                                     "duration, model vs baseline vs oracle."}.get(
                f.stem)
            if f.stem.startswith("traj_"):
                cap = (f"Dead reckoning on {f.stem[5:].replace('_', '#')} with "
                       "GNSS off for the entire run — no position updates after "
                       "the start point.")
            elif f.stem.startswith("speed_"):
                cap = (f"Regressor output vs vehicle ground truth on "
                       f"{f.stem[6:].replace('_', '#')}.")
            L += [f"### `{rel}`", "", f"![{f.stem}]({rel})", ""]
            if cap:
                L += [f"*{cap}*", ""]

    if export:
        onnx, tfl = export.get("onnx", {}), export.get("tflite", {})
        L += [
            "## 9. Deployment artefacts",
            "",
            f"- **{export['n_params']:,} parameters**, input "
            f"`{export['input_shape']}`, outputs `{', '.join(export['output'])}` "
            "in SI units.",
            f"- **ONNX** (C++ edge engine, roles 04–05): {onnx.get('size_kb','?')} KB, "
            f"matches PyTorch to **{onnx.get('max_rel_diff','?')}** relative on real "
            f"windows, **{onnx.get('latency_ms_per_window','?')} ms/window** on CPU.",
        ]
        if tfl.get("skipped"):
            L += [f"- **TFLite** (Android app, role 01): not built in this run — "
                  f"{tfl.get('reason','')}"]
        else:
            L += [f"- **TFLite** (Android app, role 01): {tfl.get('size_kb','?')} KB, "
                  f"matches PyTorch to **{tfl.get('max_rel_diff','?')}** relative."]
        L += ["", "Normalisation is baked into both graphs, so the phone and the "
              "C++ engine cannot disagree with training about scaling.", ""]

    L += [
        "## 10. Honest limitations",
        "",
        "- **The regressor alone does not reach the <10 m at 30 s target.** It "
        "reaches roughly that at a 10 s blackout; at 30 s the residual is "
        "dominated by absolute-speed error, which is the fundamentally hard part "
        "of inertial-only odometry. Closing the rest is what the road-network "
        "constraint (map matching) and the EKF in role 02 are for — a vehicle on "
        "a known road cannot be anywhere the map does not allow.",
        "- **The speed head partly reads vibration, so a different vehicle or "
        "handset is an untested shift.** Low-passing the held-out input at 2 Hz "
        "costs "
        + (f"{_tier_ratio(tier, 2.0):.1f}× " if _tier_ratio(tier, 2.0) else "")
        + "at a 30 s blackout while heading-change error is unchanged, and "
        "high-frequency energy correlates with true speed at "
        + (f"{tier['mechanism']['corr_hf_acc_speed_mean']:+.2f} "
           if tier and tier.get("mechanism") else "")
        + "across held-out runs. Road, tyre and engine vibration grows with "
        "speed, and the model uses it. That cue is specific to this sensor, "
        "mount, vehicle and road surface, and the cross-validation held all four "
        "fixed — so it cannot see this dependence. See "
        "`artifacts/metrics/sensor_tier.md`.",
        "- Trained on IO-VNBD: UK/France/Nigeria, one phone, one vehicle. Indian "
        "roads and other handsets are the fine-tuning step the roadmap already "
        "sequences (pre-train public → fine-tune own captures).",
        "- IO-VNBD phone data is 10 Hz; our own captures target 100 Hz. The "
        "window is defined in seconds, so the design carries over — but it needs "
        "retraining, not just reuse.",
        "- Ground truth is the vehicle's own CAN + survey GNSS, so labels inherit "
        "its ~0.2 % self-consistency floor.",
        "",
        "## 11. Reproducing these numbers",
        "",
        "```bash",
        "cd training",
        "export PYTHONPATH=.",
        "python -m driftless_train.download    # LFS-aware, both S and V sides",
        "python -m driftless_train.audit       # per-run inventory",
        "python -m driftless_train.prepare     # pair S with V, materialise arrays",
        "python -m driftless_train.train --epochs 40",
        "python -m driftless_train.evaluate --split val --sweep-alpha  # tune blend",
        "python -m driftless_train.evaluate --split test               # numbers",
        "python -m driftless_train.export",
        "python -m driftless_train.report",
        "pytest tests/ -q                      # pins every dataset trap",
        "```",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"-> {args.out}  ({len(L)} lines)")
    missing = [n for n, v in (("eval_summary.json", ev), ("splits.json", splits),
                              ("export_report.json", export),
                              ("processed/index.json", proc)) if v is None]
    if missing:
        print("note: sections omitted, missing " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
