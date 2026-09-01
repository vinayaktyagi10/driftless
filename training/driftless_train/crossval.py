"""Route-wise cross-validation over the trusted pool.

Why
---
Every headline number so far rests on a SINGLE held-out route (S1, 1.44 h). For a
paper round that is the softest point in the submission: "how do you know that
isn't one lucky road?" is the obvious question and a single split cannot answer
it. This trains one model per fold so that every trusted route is held out
exactly once, and reports the spread as well as the centre.

Fold design
-----------
Leave-one-route-out is not usable here: of the 12 trusted routes, three are 64 s,
101 s and 117 s long, and a 64 s test set yields about two 30 s blackout samples.
So folds are **duration-balanced groups of whole routes** -- routes sorted longest
first and dealt out snake-wise, which keeps each fold near 1/k of the total.

Routes stay intact and never straddle train and test. For fold i:
  test  = fold i
  val   = fold (i+1) mod k      (rotating, so val is always disjoint from test)
  train = the remaining folds
Weakly-coupled routes are excluded throughout, matching the shipped model.

Training goes through `train.fit`, the same function the canonical run uses, so
this estimates the model that actually ships rather than a lookalike.

Run:  python -m driftless_train.crossval --k 5 --epochs 40
Out:  artifacts/metrics/crossval.json + crossval.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from .dataset import DT, OUT_WIN, STRIDE_TRAIN, WIN, WindowDataset, load_index
from .evaluate import (
    BLACKOUT_S,
    blackout_errors,
    longest_valid_span,
    window_predictions,
)
from .pair import TRUSTED_COUPLING_CORR
from .paths import METRIC_DIR
from .train import FitConfig, fit

# A per-route median computed from a handful of blackout start points is noise.
# Two trusted routes are ~30 s long and yield single-digit sample counts at the
# longer durations; their rows are kept for completeness but marked, so nobody
# reads "140.7 m from n=1" as a measurement.
MIN_SAMPLES = 30


def build_folds(route_seconds: dict[str, float], k: int) -> list[list[str]]:
    """Deal routes into k duration-balanced groups, longest first, snake-wise."""
    order = sorted(route_seconds, key=lambda r: (-route_seconds[r], r))
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, route in enumerate(order):
        lap, pos = divmod(i, k)
        folds[pos if lap % 2 == 0 else k - 1 - pos].append(route)
    return folds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--out-win", type=int, default=OUT_WIN)
    ap.add_argument("--stride", type=int, default=STRIDE_TRAIN)
    ap.add_argument("--no-rotate-aug", action="store_true",
                    help="disable the augmentation the shipped model uses")
    ap.add_argument("--folds-only", action="store_true",
                    help="print the fold assignment and exit without training")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rebuild-docs", action="store_true",
                    help="recompute every table from a previous run's "
                         "crossval_samples.csv and rewrite crossval.json/.md "
                         "without retraining anything")
    args = ap.parse_args(argv)

    runs, idx = load_index()
    channels = idx["channels"]
    trusted = {}
    for r in json.loads((METRIC_DIR.parent.parent / "data" / "processed" /
                         "index.json").read_text())["runs"]:
        if abs(r.get("align_corr", 0.0)) >= TRUSTED_COUPLING_CORR:
            trusted[r["run_id"]] = r["route"]

    by_route: dict[str, float] = defaultdict(float)
    run_of_route: dict[str, list] = defaultdict(list)
    for run in runs:
        if run.run_id in trusted:
            by_route[run.route] += 0.0
            run_of_route[run.route].append(run)
    # durations from the index, so folds balance time rather than route count
    for r in json.loads((METRIC_DIR.parent.parent / "data" / "processed" /
                         "index.json").read_text())["runs"]:
        if r["run_id"] in trusted:
            by_route[r["route"]] += r["duration_s"]

    if len(by_route) < args.k + 1:
        print(f"only {len(by_route)} trusted routes; need at least k+1={args.k+1}")
        return 1

    folds = build_folds(dict(by_route), args.k)
    print(f"{len(by_route)} trusted routes, "
          f"{sum(by_route.values())/3600:.2f} h, into {args.k} folds:")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {sum(by_route[r] for r in f)/3600:5.2f} h  "
              f"{', '.join(f)}")
    if args.folds_only:
        return 0

    if args.rebuild_docs:
        return rebuild_docs(args, by_route, run_of_route)

    route_of_run = {run.run_id: route
                    for route, runs_ in run_of_route.items() for run in runs_}

    cfg = FitConfig(epochs=args.epochs, device=args.device)
    rotate = not args.no_rotate_aug

    fold_rows, pooled = [], []
    for i, test_routes in enumerate(folds):
        val_routes = folds[(i + 1) % args.k]
        train_routes = [r for j, f in enumerate(folds)
                        if j not in (i, (i + 1) % args.k) for r in f]

        tr = [x for r in train_routes for x in run_of_route[r]]
        va = [x for r in val_routes for x in run_of_route[r]]
        te = [x for r in test_routes for x in run_of_route[r]]

        ds_tr = WindowDataset(tr, win=args.win, stride=args.stride,
                              out_win=args.out_win, rotate_aug=rotate, seed=i)
        ds_va = WindowDataset(va, win=args.win, stride=args.out_win,
                              out_win=args.out_win)
        if not len(ds_tr) or not len(ds_va):
            print(f"fold {i}: empty window set, skipping")
            continue

        print(f"\n--- fold {i}/{args.k - 1}: test {[r for r in test_routes]} "
              f"({len(ds_tr)} train / {len(ds_va)} val windows) ---", flush=True)
        out = fit(ds_tr, ds_va, len(channels), cfg)
        model = out["model"].eval()
        dev = out["device"]

        sp_mae, dp_mae, gy_mae, errs, n_win = [], [], [], [], 0
        for run in te:
            a = run.load()
            span = longest_valid_span(a["valid"])
            if span[1] - span[0] < args.win + args.out_win * 20:
                continue
            pr = window_predictions(model, a, args.win, args.out_win, span, dev)
            if not pr:
                continue
            n_win += len(pr["starts"])
            sp_mae.append(np.abs(pr["pred_speed"] - pr["true_speed"]).mean())
            dp_mae.append(np.rad2deg(np.abs(pr["pred_dpsi"] - pr["true_dpsi"]).mean()))
            gy_mae.append(np.rad2deg(np.abs(pr["gyro_dpsi"] - pr["true_dpsi"]).mean()))
            rows = blackout_errors(pr, a)
            for row in rows:
                row["fold"] = i
                row["run_id"] = run.run_id
            errs += rows

        if not errs:
            print(f"fold {i}: no evaluable test span, skipping")
            continue
        pooled += errs

        row = {
            "fold": i,
            "test_routes": test_routes,
            "test_hours": round(sum(by_route[r] for r in test_routes) / 3600, 3),
            "train_windows": len(ds_tr),
            "val_windows": len(ds_va),
            "best_val_loss": round(out["best_val"], 5),
            "n_test_windows": n_win,
            "speed_mae_ms": round(float(np.mean(sp_mae)), 4),
            "dpsi_mae_deg": round(float(np.mean(dp_mae)), 4),
            "gyro_dpsi_mae_deg": round(float(np.mean(gy_mae)), 4),
        }
        for T in BLACKOUT_S:
            sub = [r["model_err_m"] for r in errs if r["duration_s"] == T]
            base = [r["baseline_err_m"] for r in errs if r["duration_s"] == T]
            row[f"med_{int(T)}s"] = round(float(np.median(sub)), 2) if sub else None
            row[f"base_med_{int(T)}s"] = (round(float(np.median(base)), 2)
                                          if base else None)
        fold_rows.append(row)
        print(f"  fold {i}: speed MAE {row['speed_mae_ms']:.3f} m/s  "
              f"dpsi {row['dpsi_mae_deg']:.2f} deg  "
              f"30s med {row['med_30s']} m", flush=True)

    if not fold_rows:
        print("no folds produced results")
        return 1

    res = summarise(pooled, fold_rows, by_route, route_of_run,
                    {"k": args.k, "epochs": args.epochs, "rotate_aug": rotate,
                     "win": args.win, "out_win": args.out_win})
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "crossval.json").write_text(json.dumps(res, indent=2))
    # Raw per-blackout rows: re-aggregating these costs nothing, retraining the
    # five folds to recover them costs the whole run.
    with (METRIC_DIR / "crossval_samples.csv").open("w", newline="") as fh:
        cols = ["fold", "run_id", "route", "duration_s", "model_err_m",
                "baseline_err_m", "oracle_err_m", "model_drift_pct"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in pooled:
            w.writerow({**r, "route": route_of_run.get(r["run_id"], "?")})
    write_markdown(res)

    print(f"\n{'':6} {'pooled med':>11} {'p90':>8} {'drift':>7} "
          f"{'baseline':>9} {'oracle':>7}")
    for p in res["pooled"]:
        print(f"{int(p['duration_s']):5}s {p['model_med_m']:11.2f} "
              f"{p['model_p90_m']:8.2f} {p['model_drift_med_pct']:6.2f}% "
              f"{p['baseline_med_m']:9.2f} {p['oracle_med_m']:7.2f}")
    af = res["across_folds"]
    print(f"\nacross folds: speed MAE {af['speed_mae_ms']['mean']} "
          f"+/- {af['speed_mae_ms']['std']} m/s | "
          f"30 s median {af['med_30s']['mean']} +/- {af['med_30s']['std']} m "
          f"(range {af['med_30s']['min']}-{af['med_30s']['max']})")
    print(f"-> {METRIC_DIR/'crossval.md'}")
    return 0


def summarise(pooled: list[dict], fold_rows: list[dict],
              by_route: dict[str, float], route_of_run: dict[str, str],
              meta: dict) -> dict:
    """Aggregate raw per-blackout rows into the reported tables.

    Split out from the training loop so `--rebuild-docs` can regenerate
    every table from a saved crossval_samples.csv. Without this the claim
    that re-aggregating is free would be false: fixing a formatting bug in
    the per-route table would mean retraining five models.
    """
    # Pooled: every trusted route appears in test exactly once, so this is the
    # estimate over the whole trusted pool rather than over one lucky road.
    pooled_summary = []
    for T in BLACKOUT_S:
        sub = [r for r in pooled if r["duration_s"] == T]
        if not sub:
            continue
        m = np.array([r["model_err_m"] for r in sub])
        b = np.array([r["baseline_err_m"] for r in sub])
        o = np.array([r["oracle_err_m"] for r in sub])
        d = np.array([r["model_drift_pct"] for r in sub])
        pooled_summary.append({
            "duration_s": T, "n": len(sub),
            "model_med_m": round(float(np.median(m)), 2),
            "model_p90_m": round(float(np.percentile(m, 90)), 2),
            "model_drift_med_pct": round(float(np.median(d)), 3),
            "baseline_med_m": round(float(np.median(b)), 2),
            "oracle_med_m": round(float(np.median(o)), 2),
        })

    # Per route. Every trusted route is tested exactly once, by the one fold
    # that held it out, so these are all out-of-sample. This is the table that
    # shows WHICH roads are hard -- the pooled median hides it, and a single-split
    # headline is just one row of it.
    per_route: list[dict] = []
    for route in sorted(by_route):
        rows = [r for r in pooled if route_of_run.get(r["run_id"]) == route]
        if not rows:
            continue
        entry = {"route": route,
                 "hours": round(by_route[route] / 3600, 3),
                 "fold": rows[0]["fold"]}
        for T in BLACKOUT_S:
            sub_ = [r["model_err_m"] for r in rows if r["duration_s"] == T]
            base_ = [r["baseline_err_m"] for r in rows if r["duration_s"] == T]
            entry[f"n_{int(T)}s"] = len(sub_)
            entry[f"med_{int(T)}s"] = (round(float(np.median(sub_)), 2)
                                       if sub_ else None)
            entry[f"base_med_{int(T)}s"] = (round(float(np.median(base_)), 2)
                                            if base_ else None)
        per_route.append(entry)

    tested = {r["route"] for r in per_route}
    unevaluated = sorted(set(by_route) - tested)

    def across(key):
        v = [r[key] for r in fold_rows if r.get(key) is not None]
        return {"mean": round(float(np.mean(v)), 3),
                "std": round(float(np.std(v)), 3),
                "min": round(float(np.min(v)), 3),
                "max": round(float(np.max(v)), 3)}

    res = {
        "k": meta["k"], "epochs": meta["epochs"], "rotate_aug": meta["rotate_aug"],
        "context_s": round(meta["win"] * DT, 2),
        "output_interval_s": round(meta["out_win"] * DT, 2),
        "n_trusted_routes": len(by_route),
        "trusted_hours": round(sum(by_route.values()) / 3600, 3),
        "folds": fold_rows,
        "per_route": per_route,
        "unevaluated_routes": unevaluated,
        "min_samples_for_median": MIN_SAMPLES,
        "pooled": pooled_summary,
        "across_folds": {k: across(k) for k in
                         ("speed_mae_ms", "dpsi_mae_deg", "med_10s", "med_30s",
                          "med_60s", "med_120s")},
    }
    return res


def rebuild_docs(args, by_route: dict[str, float], run_of_route: dict) -> int:
    """Regenerate the tables from a saved crossval_samples.csv.

    Per-fold aggregates (speed/heading MAE, best val loss) are not recoverable
    from the samples file -- it holds position errors only -- so those are carried
    over from the existing crossval.json rather than invented.
    """
    samples = METRIC_DIR / "crossval_samples.csv"
    prev = METRIC_DIR / "crossval.json"
    if not samples.exists():
        print(f"no {samples}; run the folds first")
        return 1
    with samples.open(newline="") as fh:
        pooled = []
        for row in csv.DictReader(fh):
            pooled.append({
                "fold": int(row["fold"]), "run_id": row["run_id"],
                "duration_s": float(row["duration_s"]),
                "model_err_m": float(row["model_err_m"]),
                "baseline_err_m": float(row["baseline_err_m"]),
                "oracle_err_m": float(row["oracle_err_m"]),
                "model_drift_pct": float(row["model_drift_pct"]),
            })
    route_of_run = {r["run_id"]: r["route"] for r in
                    csv.DictReader(samples.open(newline=""))}

    old = json.loads(prev.read_text()) if prev.exists() else {}
    fold_rows = old.get("folds", [])
    meta = {"k": old.get("k", args.k), "epochs": old.get("epochs", args.epochs),
            "rotate_aug": old.get("rotate_aug", not args.no_rotate_aug),
            "win": args.win, "out_win": args.out_win}

    res = summarise(pooled, fold_rows, by_route, route_of_run, meta)
    prev.write_text(json.dumps(res, indent=2))
    write_markdown(res)
    print(f"rebuilt from {len(pooled)} saved samples "
          f"({len(res['per_route'])} routes) -> {METRIC_DIR/'crossval.md'}")
    return 0


def _short_route_note(res: dict) -> str:
    """The leave-one-route-out argument, stated from the actual route lengths."""
    pr = res.get("per_route", [])
    if not pr:
        return ""
    lim = res.get("min_samples_for_median", 30)
    secs = [r["hours"] * 3600 for r in pr]
    thin = sorted(r["hours"] * 3600 for r in pr if (r.get("n_30s") or 0) < lim)
    n_un = len(res.get("unevaluated_routes", []))

    parts = [
        f"Route lengths in the trusted pool span {int(min(secs))} s to "
        f"{max(secs) / 3600:.1f} h, a factor of "
        f"{max(secs) / max(min(secs), 1):.0f}.",
    ]
    if thin:
        lens = " and ".join(f"{int(x)} s" for x in thin)
        parts.append(
            f"Leave-one-route-out would hand some folds a test set of only "
            f"{lens} — far too few blackout start points for a median to mean "
            f"anything.")
    else:
        parts.append(
            "Leave-one-route-out would hand the shortest folds too few blackout "
            "start points for a median to mean anything.")
    if n_un:
        routes = ", ".join(res["unevaluated_routes"])
        parts.append(
            f"One route ({routes}) is already too short to place even a single "
            f"blackout.")
    parts.append(
        "Folds are instead duration-balanced groups of whole routes: sorted "
        "longest first and dealt snake-wise, which keeps each fold near 1/k of "
        "the total time while never splitting a route.")
    return " ".join(parts)


def write_markdown(res: dict) -> None:
    af = res["across_folds"]
    SHORT_ROUTE_NOTE = _short_route_note(res)
    L = [
        "# Route-wise cross-validation",
        "",
        f"**{res['k']} folds** over the **{res['n_trusted_routes']} trusted "
        f"routes** ({res['trusted_hours']} h). Every route is held out exactly "
        f"once, so the pooled figures below cover the whole trusted pool rather "
        f"than one held-out road.",
        "",
        f"Each fold trains a fresh model through the same `train.fit` the shipped "
        f"model uses ({res['epochs']} epochs, "
        f"rotate_aug={res['rotate_aug']}, {res['context_s']} s context, "
        f"{res['output_interval_s']} s output interval). Routes never straddle "
        f"train and test; val is a disjoint fold.",
        "",
        "## Pooled — the headline",
        "",
        "| Blackout | n | model median | p90 | drift | baseline | oracle |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in res["pooled"]:
        L.append(f"| **{int(p['duration_s'])} s** | {p['n']} | "
                 f"**{p['model_med_m']} m** | {p['model_p90_m']} m | "
                 f"{p['model_drift_med_pct']} % | {p['baseline_med_m']} m | "
                 f"{p['oracle_med_m']} m |")

    L += [
        "",
        "## Spread across folds",
        "",
        "| metric | mean | std | min | max |",
        "|---|---|---|---|---|",
    ]
    labels = {"speed_mae_ms": "speed MAE (m/s)", "dpsi_mae_deg": "Δψ MAE (°)",
              "med_10s": "10 s median (m)", "med_30s": "30 s median (m)",
              "med_60s": "60 s median (m)", "med_120s": "120 s median (m)"}
    for k, lab in labels.items():
        a = af[k]
        L.append(f"| {lab} | **{a['mean']}** | {a['std']} | {a['min']} | "
                 f"{a['max']} |")

    L += [
        "",
        "The spread is the point of this table. A single-split number cannot show "
        "it, and the fold-to-fold range is what a reader should have in mind when "
        "reading any one figure.",
        "",
        "## Per fold",
        "",
        "| fold | test hours | test routes | speed MAE | Δψ MAE | 30 s med | "
        "baseline 30 s |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in res["folds"]:
        L.append(f"| {f['fold']} | {f['test_hours']} | "
                 f"{', '.join(f['test_routes'])} | {f['speed_mae_ms']} | "
                 f"{f['dpsi_mae_deg']}° | {f['med_30s']} m | "
                 f"{f['base_med_30s']} m |")

    L += [
        "",
        "## Per route — out-of-sample, one fold each",
        "",
        "| route | h | fold | 10 s | 30 s | 60 s | 120 s | n at 30 s | "
        "baseline 30 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lim = res.get("min_samples_for_median", 30)

    def _m(v):
        return "—" if v is None else f"{v} m"

    ordered = sorted(res.get("per_route", []),
                     key=lambda r: (r["med_30s"] is None, r["med_30s"]))
    for r in ordered:
        thin = (r.get("n_30s") or 0) < lim
        flag = " ⚠" if thin else ""
        L.append(f"| {r['route']}{flag} | {r['hours']} | {r['fold']} | "
                 f"{_m(r['med_10s'])} | **{_m(r['med_30s'])}** | "
                 f"{_m(r['med_60s'])} | {_m(r['med_120s'])} | "
                 f"{r.get('n_30s', 0)} | {_m(r['base_med_30s'])} |")

    thin_routes = [r["route"] for r in ordered if (r.get("n_30s") or 0) < lim]
    solid = [r for r in ordered if (r.get("n_30s") or 0) >= lim]
    L += [
        "",
        "Sorted easiest first. `—` means the route is shorter than the blackout, "
        f"so no sample exists. **⚠ marks fewer than {lim} blackout samples at "
        f"30 s** — those medians are indicative only.",
        "",
    ]
    if solid:
        lo, hi = solid[0], solid[-1]
        L.append(
            f"Across the {len(solid)} routes with enough samples, the 30 s "
            f"median spans **{lo['med_30s']} m ({lo['route']}) to "
            f"{hi['med_30s']} m ({hi['route']})** — a factor of "
            f"{hi['med_30s'] / max(lo['med_30s'], 1e-9):.1f}. Road difficulty "
            f"varies far more than the pooled median suggests, which is exactly "
            f"why one held-out road is not enough evidence.")
        L.append("")
    if thin_routes:
        L.append(f"Thin rows: {', '.join(thin_routes)}.")
        L.append("")
    if res.get("unevaluated_routes"):
        L.append(
            f"**Not evaluated:** {', '.join(res['unevaluated_routes'])}. Held "
            f"out by its fold as intended, but its longest continuous valid span "
            f"is shorter than the minimum needed to place even one blackout, so "
            f"it contributes training and validation data only. "
            f"{len(res['per_route'])} of {res['n_trusted_routes']} trusted "
            f"routes therefore appear in the pooled figures.")
        L.append("")

    L += [
        "## Why not leave-one-route-out",
        "",
        SHORT_ROUTE_NOTE,
        "",
        "## Caveats",
        "",
        "- Fold models are trained on ~3/5 of the trusted pool, so each sees "
        "*less* data than the shipped model. These figures are therefore a "
        "slightly pessimistic estimate of the shipped model, not a measurement "
        "of it.",
        "- The trusted pool is 12 routes from 4 drivers, one handset, one "
        "vehicle. Cross-validation quantifies variation *within* that pool; it "
        "says nothing about a different phone, city or car.",
        "- Weakly-coupled routes are excluded throughout, matching the shipped "
        "configuration.",
    ]
    (METRIC_DIR / "crossval.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
