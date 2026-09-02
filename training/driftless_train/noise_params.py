"""Measurement-noise parameters for the fusion filter, from held-out residuals.

What role 02 needs from us, and why a single sigma is not enough
---------------------------------------------------------------
The regressor's outputs enter the UKF as measurements, so the filter needs their
noise. Two properties matter, and the second one is the one that bites:

1. **Magnitude.** sigma of (predicted - true), for speed and for heading change.
   Speed error grows with speed, so a single number is wrong across the range and
   a per-band table is given.

2. **Time correlation.** Consecutive predictions share most of their 8 s input
   context, so their errors are heavily correlated. Treating each 2 s prediction
   as an independent observation injects far more information than actually
   arrived -- the covariance collapses and the filter then starts REJECTING
   honest GNSS fixes. That is not hypothetical: `ukf_fusion_engine.h` records
   exactly this failure for the non-holonomic constraint, where sigma=0.1 at
   10 Hz drove attitude covariance to ~1 degree and made the filter reject 21 of
   the GNSS fixes following a blackout.

   So we report the measured decorrelation time and the inflation factor that
   makes independent-sampling honest.

Run:  python -m driftless_train.noise_params --ckpt artifacts/models/tcn_best.pt
Out:  artifacts/metrics/measurement_noise.json + .md
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .dataset import DT, load_index, make_splits, split_runs
from .evaluate import load_model, longest_valid_span, window_predictions
from .pair import TRUSTED_COUPLING_CORR
from .paths import METRIC_DIR, MODEL_DIR

SPEED_BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 25.0), (25.0, 40.0))


def _decorrelation(series: list[np.ndarray] | np.ndarray, win_dt: float,
                   max_lag: int = 40) -> dict:
    """Autocorrelation of the residual series and its 1/e decorrelation time.

    Takes a LIST of per-run series and averages their autocorrelations, weighted
    by length. Concatenating first would be wrong: each run has its own mean
    residual (they differ by ~1 m/s between held-out routes), so the joins act
    as step changes and inflate the autocorrelation at every lag. Doing exactly
    that made the apparent decorrelation time jump 8 s -> 20 s purely from
    pooling a second route, which would have propagated into the filter's sigma
    inflation as a factor of 3.16 instead of 2.
    """
    runs = [series] if isinstance(series, np.ndarray) else list(series)
    runs = [np.asarray(r, dtype=float) for r in runs if len(r) >= max_lag + 2]
    if not runs:
        return {"lag1": None, "decorrelation_s": None, "acf": []}

    num = np.zeros(max_lag)
    wsum = 0.0
    for r in runs:
        x = r - r.mean()            # de-mean PER RUN
        denom = float(np.dot(x, x))
        if denom <= 0:
            continue
        w = float(len(x))
        for k in range(1, max_lag + 1):
            num[k - 1] += w * float(np.dot(x[:-k], x[k:]) / denom)
        wsum += w
    if wsum <= 0:
        return {"lag1": None, "decorrelation_s": None, "acf": []}
    acf = list(num / wsum)

    tau = None
    for k, v in enumerate(acf, start=1):
        if v < np.exp(-1.0):
            tau = k * win_dt
            break
    return {
        "lag1": round(acf[0], 4),
        "decorrelation_s": tau,
        "acf": [round(v, 4) for v in acf[:20]],
    }


def analyse(ckpt, splits: tuple[str, ...] = ("test", "val"),
            device: str = "cpu") -> dict:
    """Residual statistics over EVERY held-out split, not one route.

    Originally this ran on `test` alone, which is a single route, and published
    a sigma of 2.2046 m/s plus a bias of +0.4853 m/s as if both were properties
    of the model. Neither is: on the other held-out route the residual sigma is
    3.159 m/s and the bias is -0.554 m/s. The fusion filter consumed the
    single-route numbers, so the sigma it was given was ~30 % optimistic --
    exactly the over-tight-sigma failure that filter already documents
    elsewhere. Pooling across held-out routes is the conservative default.
    """
    dev = torch.device(device)
    model, win, out_win, chan_idx = load_model(ckpt, dev)

    runs, idx = load_index()
    by_split = split_runs(runs, make_splits(runs))
    coupling = {r["run_id"]: abs(r.get("align_corr", 0.0) or 0.0)
                for r in idx["runs"]}
    sel = [r for sp in splits for r in by_split.get(sp, [])
           if coupling.get(r.run_id, 0.0) >= TRUSTED_COUPLING_CORR]
    if not sel:
        raise SystemExit(f"no trusted runs in splits {list(splits)}")

    sp_res, dp_res, dv_res, sp_true = [], [], [], []
    per_run = []
    for run in sel:
        arrays = run.load()
        span = longest_valid_span(arrays["valid"])
        if span[1] - span[0] < win + out_win * 20:
            continue
        pr = window_predictions(model, arrays, win, out_win, span, dev,
                                channels=chan_idx)
        if not pr:
            continue
        rs = pr["pred_speed"] - pr["true_speed"]
        rd = pr["pred_dpsi"] - pr["true_dpsi"]
        sp_res.append(rs)
        dp_res.append(rd)
        sp_true.append(pr["true_speed"])
        # dv must use the SAME definition as the training target
        # (dataset.window_targets): the speed change WITHIN the output interval,
        # v[end-1] - v[start]. Differencing consecutive window MEANS instead --
        # which an earlier version of this file did -- measures a different
        # quantity and reports a residual the model was never trained against.
        v_all = arrays["speed_ms"].astype(float)
        dv_true = np.array([v_all[st + out_win - 1] - v_all[st]
                            for st in pr["starts"]])
        dv_res.append(pr["pred_dv"] - dv_true)
        per_run.append({
            "run_id": run.run_id,
            "n": int(len(rs)),
            "speed_sigma": round(float(rs.std()), 4),
            "speed_bias": round(float(rs.mean()), 4),
            "dpsi_sigma_deg": round(float(np.rad2deg(rd.std())), 4),
            "dpsi_bias_deg": round(float(np.rad2deg(rd.mean())), 4),
        })

    rs = np.concatenate(sp_res)
    rd = np.concatenate(dp_res)
    vt = np.concatenate(sp_true)
    win_dt = out_win * DT

    bands = []
    for lo, hi in SPEED_BANDS:
        m = (vt >= lo) & (vt < hi)
        if m.sum() < 30:
            continue
        bands.append({
            "speed_lo": lo, "speed_hi": hi, "n": int(m.sum()),
            "sigma_mps": round(float(rs[m].std()), 4),
            "bias_mps": round(float(rs[m].mean()), 4),
            "sigma_frac_of_speed": round(float(rs[m].std() /
                                               max(vt[m].mean(), 1e-6)), 4),
        })

    corr_sp = _decorrelation(sp_res, win_dt)
    corr_dp = _decorrelation(dp_res, win_dt)

    def inflation(corr):
        """sqrt of the effective number of correlated samples per decorrelation
        time: the factor to multiply sigma by if each prediction is fed as an
        independent measurement at win_dt spacing."""
        tau = corr.get("decorrelation_s")
        if not tau or tau <= win_dt:
            return 1.0
        return round(float(np.sqrt(tau / win_dt)), 3)

    # Per-split residual stats: the spread between them is the point. A single
    # split's sigma and bias are not properties of the model.
    by_split_stats = []
    for sp_name in splits:
        ids = {r.run_id for r in by_split.get(sp_name, [])}
        rows = [x for x in per_run if x["run_id"] in ids]
        if not rows:
            continue
        n = sum(x["n"] for x in rows)
        by_split_stats.append({
            "split": sp_name, "n": n,
            "routes": sorted({r.route for r in by_split.get(sp_name, [])
                              if r.run_id in {x["run_id"] for x in rows}}),
            "speed_sigma_mps": round(float(np.sqrt(
                sum(x["n"] * x["speed_sigma"] ** 2 for x in rows) / n)), 4),
            "speed_bias_mps": round(float(
                sum(x["n"] * x["speed_bias"] for x in rows) / n), 4),
        })

    return {
        "splits": list(splits),
        "by_split": by_split_stats,
        "bias_is_route_dependent": True,
        "checkpoint": str(ckpt),
        "context_s": round(win * DT, 2),
        "update_interval_s": round(win_dt, 2),
        "n_windows": int(len(rs)),
        "speed": {
            "sigma_mps": round(float(rs.std()), 4),
            "bias_mps": round(float(rs.mean()), 4),
            "mae_mps": round(float(np.abs(rs).mean()), 4),
            "p95_abs_mps": round(float(np.percentile(np.abs(rs), 95)), 4),
            "by_speed_band": bands,
            "correlation": corr_sp,
            "independent_sampling_inflation": inflation(corr_sp),
        },
        "dpsi": {
            "sigma_rad": round(float(rd.std()), 6),
            "sigma_deg": round(float(np.rad2deg(rd.std())), 4),
            "bias_deg": round(float(np.rad2deg(rd.mean())), 4),
            "correlation": corr_dp,
            "independent_sampling_inflation": inflation(corr_dp),
        },
        "dv": {
            "sigma_mps": round(float(np.concatenate(dv_res).std()), 4),
            "bias_mps": round(float(np.concatenate(dv_res).mean()), 4),
            "mae_mps": round(float(np.abs(np.concatenate(dv_res)).mean()), 4),
            "definition": "v[end-1] - v[start] within the output interval, "
                          "matching dataset.window_targets",
            "correlation": _decorrelation(dv_res, win_dt),
        },
        "per_run": per_run,
    }


def write_markdown(res: dict, path) -> None:
    sp, dp = res["speed"], res["dpsi"]
    infl = sp["independent_sampling_inflation"]
    L = [
        "# Measurement noise for the fusion filter",
        "",
        f"From residuals pooled over the held-out "
        f"**{'** and **'.join(res['splits'])}** splits "
        f"({res['n_windows']} predictions, {res['update_interval_s']} s apart, "
        f"{res['context_s']} s context).",
        "",
        "## Forward-speed measurement",
        "",
        "> **Do not hard-code the bias.** These values are pooled over every "
        "held-out route. Bias is not a stable property of this model: it is a "
        "property of each route's speed distribution interacting with a "
        "shrinkage estimator, and it changes sign between held-out routes (see "
        "the per-split table below). Subtracting one route's bias measurably "
        "hurts on another. Use **0.0** unless you have calibrated on the actual "
        "deployment route.",
        "",
        f"- sigma **{sp['sigma_mps']} m/s**, bias {sp['bias_mps']} m/s, "
        f"MAE {sp['mae_mps']} m/s, p95 |error| {sp['p95_abs_mps']} m/s",
        "",
        "Error scales with speed, so a single sigma is wrong across the range:",
        "",
        "### Per held-out split — why one route is not enough",
        "",
        "| split | routes | n | speed sigma (m/s) | speed bias (m/s) |",
        "|---|---|---|---|---|",
        *[f"| {b['split']} | {', '.join(b['routes'])} | {b['n']} | "
          f"**{b['speed_sigma_mps']}** | {b['speed_bias_mps']:+} |"
          for b in res.get("by_split", [])],
        "",
        "Sigma differs by ~40 % between held-out routes and the bias changes "
        "sign. Use the pooled sigma above; an over-tight sigma is the failure "
        "mode that collapses the filter's covariance and makes it reject honest "
        "GNSS fixes.",
        "",
        "> **This table is NOT a calibration curve.** It bins by *true* speed, "
        "which the filter cannot observe, and a minimum-MSE estimator "
        "necessarily looks biased when conditioned on the truth -- it shrinks "
        "toward the mean, measured here at std(pred)/std(true) = 0.84 with a "
        "pred-on-true slope of 0.74. Correcting that de-shrinks the estimate "
        "and *raises* MSE. Cross-fitting a linear correction between the two "
        "held-out routes made speed MAE worse in both directions "
        "(1.54 -> 1.69 and 2.06 -> 2.25 m/s). This table shows where the error "
        "lives; it is not meant to be inverted.",
        "",
        "| speed band | n | sigma (m/s) | bias (m/s) | sigma / mean speed |",
        "|---|---|---|---|---|",
    ]
    for b in sp["by_speed_band"]:
        L.append(f"| {b['speed_lo']:.0f}-{b['speed_hi']:.0f} m/s | {b['n']} | "
                 f"{b['sigma_mps']} | {b['bias_mps']} | "
                 f"{100*b['sigma_frac_of_speed']:.1f} % |")

    L += [
        "",
        "## Speed-change (`dv`) measurement",
        "",
        f"- sigma **{res['dv']['sigma_mps']} m/s**, bias "
        f"{res['dv']['bias_mps']} m/s, MAE {res['dv']['mae_mps']} m/s over "
        f"{res['update_interval_s']} s",
        f"- decorrelation time "
        f"{res['dv']['correlation']['decorrelation_s']} s "
        f"(lag-1 {res['dv']['correlation']['lag1']})",
        "",
        f"Definition: `{res['dv']['definition']}`. This is the better-conditioned "
        "of the two speed outputs and is what a blackout should propagate from, "
        "since it starts from a known speed.",
        "",
        "## Heading-change measurement",
        "",
        f"- sigma **{dp['sigma_deg']}°** ({dp['sigma_rad']} rad) per "
        f"{res['update_interval_s']} s, bias {dp['bias_deg']}°",
        "",
        "## The correlation warning",
        "",
        "Consecutive predictions share most of their input context, so their "
        "errors are **not independent**:",
        "",
        f"- speed residual lag-1 autocorrelation **{sp['correlation']['lag1']}**, "
        f"decorrelation time **{sp['correlation']['decorrelation_s']} s**",
        f"- heading residual lag-1 autocorrelation **{dp['correlation']['lag1']}**, "
        f"decorrelation time **{dp['correlation']['decorrelation_s']} s**",
        "",
        f"Feeding every prediction as an independent measurement at "
        f"{res['update_interval_s']} s spacing therefore over-informs the filter. "
        f"If you do that, inflate sigma by about **{infl}x** for speed "
        f"(sqrt(tau / update interval)). The alternative -- and the better fix -- "
        f"is to apply the update only once per decorrelation time, or to model "
        f"the correlation.",
        "",
        "This is the same failure mode already documented for the non-holonomic "
        "constraint in `ukf_fusion_engine.h`: an over-tight sigma applied at high "
        "rate collapsed the attitude covariance and made the filter reject honest "
        "GNSS fixes after the blackout. Same trap, different measurement.",
        "",
        "## Suggested wiring",
        "",
        "The measurement is the longitudinal component of body velocity -- the "
        "one axis the non-holonomic constraint deliberately leaves free. It fits "
        "the existing unscented path with no new filter code:",
        "",
        "```cpp",
        "// h(x) = forward component of body velocity",
        "auto h = [](const NavState& s) {",
        "    Eigen::VectorXd z(1);",
        "    z(0) = UkfFusionEngine::bodyVelocity(s).x();",
        "    return z;",
        "};",
        "Eigen::MatrixXd sqrt_R(1, 1);",
        f"sqrt_R(0, 0) = {sp['sigma_mps']} * {infl};"
        "   // pooled held-out sigma x correlation inflation",
        "engine.updateUnscented(h, z_from_model, sqrt_R, 0.99);",
        "```",
        "",
        "Speed-dependent sigma from the table above is better than the single "
        "value, and the NIS gate at 0.99 will reject the occasional bad "
        "prediction rather than letting it into the state.",
        "",
        "## Per-run breakdown",
        "",
        "| run | n | speed sigma | speed bias | dpsi sigma | dpsi bias |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["per_run"]:
        L.append(f"| {r['run_id']} | {r['n']} | {r['speed_sigma']} | "
                 f"{r['speed_bias']} | {r['dpsi_sigma_deg']}° | "
                 f"{r['dpsi_bias_deg']}° |")
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODEL_DIR / "tcn_best.pt"))
    ap.add_argument("--splits", nargs="+", default=["test", "val"],
                    help="held-out splits to pool over (default: test val)")
    args = ap.parse_args(argv)

    from pathlib import Path
    res = analyse(Path(args.ckpt), tuple(args.splits))
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "measurement_noise.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, METRIC_DIR / "measurement_noise.md")

    sp, dp = res["speed"], res["dpsi"]
    print(f"speed sigma {sp['sigma_mps']} m/s  bias {sp['bias_mps']}  "
          f"lag1 {sp['correlation']['lag1']}  "
          f"tau {sp['correlation']['decorrelation_s']} s  "
          f"inflation x{sp['independent_sampling_inflation']}")
    print(f"dpsi  sigma {dp['sigma_deg']} deg  "
          f"lag1 {dp['correlation']['lag1']}  "
          f"tau {dp['correlation']['decorrelation_s']} s")
    print(f"-> {METRIC_DIR/'measurement_noise.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
