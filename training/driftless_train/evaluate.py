"""Dead-reckoning evaluation: the numbers and plots Round 1 is screened on.

The question a judge will ask is "how far off are you after GPS dies, and how do
you know?". So the metric is not window-level regression error -- it is position
error after a GNSS blackout of a stated duration, measured over many blackout
start points across held-out routes, against three references:

  oracle   integrate the TRUE speed and yaw rate -- the floor imposed by the
           dead-reckoning model itself (0.2%-ish, not zero)
  baseline no ML: hold the last known speed, integrate the phone gyro for
           heading -- what you get without the learned component
  model    the TCN's predicted speed and heading change

Run:  python -m driftless_train.evaluate
Out:  artifacts/metrics/eval_*.csv, artifacts/plots/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import DT, PROC_DIR, load_index, make_splits, split_runs
from .model import SpeedHeadingTCN
from .paths import METRIC_DIR, MODEL_DIR, PLOT_DIR
from .preprocess import FEATURE_CHANNELS

BLACKOUT_S = (10.0, 30.0, 60.0, 120.0)

# Complementary-filter gain blending propagated speed (v += dv) against the
# absolute-speed head. Measured on val, the optimum is NOT a constant:
#
#   alpha   10 s med   120 s med
#   0.00      7.94 m     291.2 m     pure dv integration: best early, drifts late
#   1.00     13.41 m     147.9 m     pure absolute head: worst early, best late
#
# The reason is physical. A blackout starts from a known speed (the last GNSS
# fix), so integrating dv from it is excellent for the first few seconds and goes
# stale as integration error accumulates. So the gain RAMPS: trust the propagated
# speed initially, hand over to the absolute head as the fix ages. This is
# precisely a Kalman gain responding to growing process covariance, which is the
# argument for role 02 owning this in the EKF rather than a fixed blend.
# Selected on val by --sweep-alpha over a (alpha_max, tau) grid, re-tuned after
# mount-rotation augmentation changed the model. tau=20 s now wins the 30 s
# target outright (33.01 m vs 33.81 m at tau=40) and also 60 s and 120 s, while
# the 10 s gap that previously justified the wider ramp collapsed from 4.5 m to
# 0.22 m. tau=40 keeps a slightly better 30 s p90 (110 vs 115 m); that is the one
# thing given up here.
ALPHA_MAX_DEFAULT = 1.0
ALPHA_TAU_S_DEFAULT = 20.0
GYRO_VERT_IDX = FEATURE_CHANNELS.index("gyro_vert")

# Palette: colour-blind safe, readable in print.
C_GT, C_MODEL, C_BASE, C_ORACLE = "#111111", "#0072B2", "#D55E00", "#009E73"


def load_model(ckpt_path: Path, device: torch.device):
    """Returns (model, context_len, output_interval_len, channel_indices)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpeedHeadingTCN(len(ck["channels"]), width=ck["width"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    idx = ck.get("channel_indices")
    return (model, ck["win"], ck.get("out_win", ck["win"]),
            None if idx is None else np.asarray(idx))


def longest_valid_span(valid: np.ndarray) -> tuple[int, int]:
    """Largest contiguous all-valid span, as [start, end)."""
    best = (0, 0)
    i = 0
    n = len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


def window_predictions(model, arrays: dict, win: int, out_win: int,
                       span: tuple[int, int], device: torch.device,
                       batch: int = 512, channels=None,
                       rotation: np.ndarray | None = None) -> dict:
    """Tile the span with non-overlapping OUTPUT intervals of out_win samples.

    Each prediction is fed `win` samples of causal context ending at the close of
    its output interval, so consecutive intervals share history but tile the
    timeline exactly once -- integrating them is a Riemann sum over every sample.
    """
    lo, hi = span
    ends = np.arange(lo + win, hi + 1, out_win)
    if ends.size == 0:
        return {}
    starts_out = ends - out_win

    feats = arrays["features"]
    X = np.stack([feats[e - win:e].T for e in ends]).astype(np.float64)
    if rotation is not None:
        # Simulate a differently-mounted phone: a fixed rotation for the run.
        from .augment import rotate_batch
        X = rotate_batch(X, rotation)
    if channels is not None:
        X = X[:, channels, :]
    X = X.astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    P = np.concatenate(preds)                       # (K, 2)

    speed = arrays["speed_ms"].astype(float)
    yaw = arrays["yaw_rate_rads"].astype(float)
    gyro = feats[:, GYRO_VERT_IDX].astype(float)

    return {
        # sample index at which each output interval STARTS
        "starts": starts_out,
        "pred_speed": P[:, 0],
        "pred_dpsi": P[:, 1],
        "pred_dv": P[:, 2] if P.shape[1] > 2 else np.zeros(len(P)),
        "true_speed": np.array([speed[s:s + out_win].mean() for s in starts_out]),
        "true_dpsi": np.array([yaw[s:s + out_win].sum() * DT for s in starts_out]),
        "gyro_dpsi": np.array([gyro[s:s + out_win].sum() * DT for s in starts_out]),
        "win_dt": out_win * DT,
    }


def cf_gain_schedule(n: int, win_dt: float, alpha_max: float,
                     tau_s: float) -> np.ndarray:
    """Ramp the blend from 0 toward alpha_max with time constant tau_s.

    tau_s <= 0 means a constant gain of alpha_max.
    """
    if tau_s <= 0:
        return np.full(n, alpha_max)
    t = (np.arange(n) + 1) * win_dt
    return alpha_max * (1.0 - np.exp(-t / tau_s))


def speed_from_cf(pred_abs: np.ndarray, pred_dv: np.ndarray, v0: float,
                  alpha) -> np.ndarray:
    """Propagate speed from a known starting value, corrected toward the absolute head.

    A blackout begins at a known speed -- whatever the last GNSS fix reported --
    so the well-conditioned move is to integrate the predicted speed CHANGE from
    there. Pure integration would drift without bound, so each step is pulled
    gently toward the absolute-speed head. Returns the MEAN speed over each
    output interval, which is what displacement needs.
    """
    n = len(pred_dv)
    a = np.full(n, float(alpha)) if np.isscalar(alpha) else np.asarray(alpha)[:n]
    v = np.empty(n + 1)
    v[0] = v0
    for i in range(n):
        v[i + 1] = max((1.0 - a[i]) * (v[i] + pred_dv[i])
                       + a[i] * pred_abs[i], 0.0)
    return 0.5 * (v[:-1] + v[1:])


def integrate(speed: np.ndarray, dpsi: np.ndarray, win_dt: float,
              e0: float, n0: float, psi0: float) -> tuple[np.ndarray, np.ndarray]:
    """Tiled dead reckoning with mid-point heading (compass: east=sin, north=cos)."""
    d = speed * win_dt
    psi = psi0 + np.cumsum(dpsi) - dpsi          # heading at each window START
    psi_mid = psi + dpsi / 2.0
    e = e0 + np.cumsum(d * np.sin(psi_mid))
    n = n0 + np.cumsum(d * np.cos(psi_mid))
    return e, n


def blackout_errors(pred: dict, arrays: dict, durations=BLACKOUT_S,
                    start_hop_s: float = 10.0,
                    alpha_max: float = ALPHA_MAX_DEFAULT,
                    tau_s: float = ALPHA_TAU_S_DEFAULT) -> list[dict]:
    """Position error after a blackout of each duration, over many start points."""
    starts, win_dt = pred["starts"], pred["win_dt"]
    e_gt, n_gt = arrays["east_m"], arrays["north_m"]
    head = arrays["heading_rad"].astype(float)

    hop = max(int(round(start_hop_s / win_dt)), 1)
    rows = []
    for T in durations:
        k = int(round(T / win_dt))
        if k < 1 or len(starts) <= k:
            continue
        for k0 in range(0, len(starts) - k, hop):
            i0 = starts[k0]
            iend = starts[k0 + k]
            psi0, e0, n0 = head[i0], e_gt[i0], n_gt[i0]
            true_d = float(np.hypot(e_gt[iend] - e0, n_gt[iend] - n0))
            travelled = float(np.hypot(np.diff(e_gt[i0:iend]),
                                       np.diff(n_gt[i0:iend])).sum())
            if travelled < 1.0:
                continue        # parked: a drift percentage would be meaningless

            sl = slice(k0, k0 + k)
            out = {"duration_s": T, "k0": k0, "travelled_m": travelled,
                   "true_disp_m": true_d}
            # Speed at the moment GNSS was lost -- available in service from the
            # last fix, so the complementary filter is entitled to it.
            v0 = float(pred["true_speed"][k0])
            gain = cf_gain_schedule(k, win_dt, alpha_max, tau_s)
            cf_speed = speed_from_cf(pred["pred_speed"][sl], pred["pred_dv"][sl],
                                     v0, gain)
            for name, sp, dp in (
                ("model", cf_speed, pred["pred_dpsi"][sl]),
                ("model_abs", pred["pred_speed"][sl], pred["pred_dpsi"][sl]),
                ("oracle", pred["true_speed"][sl], pred["true_dpsi"][sl]),
                ("baseline", np.full(k, v0), pred["gyro_dpsi"][sl]),
            ):
                e, n = integrate(sp, dp, win_dt, e0, n0, psi0)
                err = float(np.hypot(e[-1] - e_gt[iend], n[-1] - n_gt[iend]))
                out[f"{name}_err_m"] = err
                out[f"{name}_drift_pct"] = 100.0 * err / travelled
            rows.append(out)
    return rows


def plot_trajectory(pred: dict, arrays: dict, run_id: str, out: Path) -> None:
    """Full-span dead reckoning from a single GNSS fix -- the headline figure."""
    starts, win_dt = pred["starts"], pred["win_dt"]
    e_gt, n_gt = arrays["east_m"], arrays["north_m"]
    head = arrays["heading_rad"].astype(float)
    i0 = starts[0]
    idx = np.r_[starts, starts[-1] + int(win_dt / DT)]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot(e_gt[idx[0]:idx[-1]], n_gt[idx[0]:idx[-1]], color=C_GT, lw=2.2,
            label="Ground truth (vehicle GNSS)", zorder=3)
    cf = speed_from_cf(pred["pred_speed"], pred["pred_dv"],
                       float(pred["true_speed"][0]),
                       cf_gain_schedule(len(pred["pred_dv"]), pred["win_dt"],
                                        ALPHA_MAX_DEFAULT, ALPHA_TAU_S_DEFAULT))
    for name, sp, dp, c in (
        ("Oracle (true speed + yaw)", pred["true_speed"], pred["true_dpsi"], C_ORACLE),
        ("Driftless model (IMU only)", cf, pred["pred_dpsi"], C_MODEL),
        ("Baseline (held speed + gyro)",
         np.full(len(starts), pred["true_speed"][0]), pred["gyro_dpsi"], C_BASE),
    ):
        e, n = integrate(sp, dp, win_dt, e_gt[i0], n_gt[i0], head[i0])
        ax.plot(e, n, color=c, lw=1.6, alpha=0.9, label=name)

    ax.scatter([e_gt[i0]], [n_gt[i0]], s=70, color=C_GT, marker="o",
               zorder=4, label="Start (last GNSS fix)")
    # Clip to the ground-truth extent plus a margin. Unclipped, an 86-minute
    # baseline track runs so far off that it rescales the axes and hides
    # everything else.
    span_e = e_gt[idx[0]:idx[-1]]
    span_n = n_gt[idx[0]:idx[-1]]
    pad = 0.45 * max(np.ptp(span_e), np.ptp(span_n))
    ax.set_xlim(span_e.min() - pad, span_e.max() + pad)
    ax.set_ylim(span_n.min() - pad, span_n.max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    dur = len(starts) * win_dt
    ax.set_title(f"{run_id} — EXTREME STRESS CASE: GNSS off for the whole "
                 f"{dur/60:.0f} min run\nNot the product claim (see the 60 s "
                 f"blackout figure). Tracks may leave the frame.", fontsize=10)
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_blackout_segments(pred: dict, arrays: dict, run_id: str, out: Path,
                           duration_s: float = 60.0, n: int = 6,
                           min_travel_m: float = 150.0) -> None:
    """Small multiples of realistic blackouts -- the figure that shows the claim.

    A whole-run plot is the wrong headline: nobody dead-reckons for 86 minutes,
    and over that span even the oracle wanders kilometres because heading is
    integrated from yaw rate. What matters is a tunnel- or car-park-length
    blackout, so we draw several of those, spread along the route, each starting
    from a real GNSS fix and annotated with its own final error.
    """
    starts, win_dt = pred["starts"], pred["win_dt"]
    k = int(round(duration_s / win_dt))
    e_gt, n_gt = arrays["east_m"], arrays["north_m"]
    head = arrays["heading_rad"].astype(float)
    if k < 1 or len(starts) <= k + 1:
        return

    # Candidate start points spread across the run that actually cover ground.
    cands = []
    for k0 in np.linspace(0, len(starts) - k - 1, 4 * n).astype(int):
        i0, iend = starts[k0], starts[k0 + k]
        travelled = float(np.hypot(np.diff(e_gt[i0:iend]),
                                   np.diff(n_gt[i0:iend])).sum())
        if travelled >= min_travel_m:
            cands.append((k0, travelled))
    if not cands:
        return
    picks = [c[0] for c in cands[::max(len(cands) // n, 1)]][:n]

    cols = 3
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 3.8 * rows_n))
    axes = np.atleast_1d(axes).ravel()

    # axes is padded to a full grid, so it is deliberately longer than picks.
    for ax, k0 in zip(axes, picks, strict=False):
        i0, iend = starts[k0], starts[k0 + k]
        psi0, e0, n0 = head[i0], e_gt[i0], n_gt[i0]
        sl = slice(k0, k0 + k)
        v0 = float(pred["true_speed"][k0])
        gain = cf_gain_schedule(k, win_dt, ALPHA_MAX_DEFAULT, ALPHA_TAU_S_DEFAULT)
        cf = speed_from_cf(pred["pred_speed"][sl], pred["pred_dv"][sl], v0, gain)

        ax.plot(e_gt[i0:iend + 1] - e0, n_gt[i0:iend + 1] - n0,
                color=C_GT, lw=2.6, label="Ground truth", zorder=3)
        errs = {}
        for name, sp, dp, c, lbl in (
            ("model", cf, pred["pred_dpsi"][sl], C_MODEL, "Driftless (IMU only)"),
            ("baseline", np.full(k, v0), pred["gyro_dpsi"][sl], C_BASE,
             "Baseline (held speed + gyro)"),
        ):
            e, n = integrate(sp, dp, win_dt, e0, n0, psi0)
            ax.plot(e - e0, n - n0, color=c, lw=1.7, label=lbl)
            errs[name] = float(np.hypot(e[-1] - e_gt[iend], n[-1] - n_gt[iend]))

        t_min = i0 * DT / 60.0
        ax.set_title(f"t = {t_min:.0f} min   ·   error {errs['model']:.0f} m "
                     f"(baseline {errs['baseline']:.0f} m)", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25, ls=":")
        ax.tick_params(labelsize=7)

    for ax in axes[len(picks):]:
        ax.axis("off")
    axes[0].legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle(f"{run_id} — {duration_s:.0f} s GNSS blackouts at six points "
                 f"along a held-out route\n"
                 f"each starts from a real GNSS fix; axes in metres, "
                 f"relative to that fix", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_blackout_curve(rows: list[dict], out: Path) -> None:
    """Median and p90 position error against blackout duration."""
    if not rows:
        return
    dur = sorted({r["duration_s"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, c in (("model", C_MODEL), ("baseline", C_BASE), ("oracle", C_ORACLE),
                    ("model_abs", "#999999")):
        med = [np.median([r[f"{name}_err_m"] for r in rows if r["duration_s"] == d])
               for d in dur]
        p90 = [np.percentile([r[f"{name}_err_m"] for r in rows
                              if r["duration_s"] == d], 90)
               for d in dur]
        axes[0].plot(dur, med, "o-", color=c, label=f"{name} (median)")
        axes[0].fill_between(dur, med, p90, color=c, alpha=0.15)
        dmed = [np.median([r[f"{name}_drift_pct"] for r in rows
                           if r["duration_s"] == d])
                for d in dur]
        axes[1].plot(dur, dmed, "o-", color=c, label=f"{name} (median)")

    axes[0].axhline(10.0, color="#888", ls="--", lw=1)
    axes[0].annotate("10 m target", (dur[0], 10.5), fontsize=8, color="#666")
    axes[0].set_xlabel("GNSS blackout duration (s)")
    axes[0].set_ylabel("Final position error (m)")
    axes[0].set_title("Position error vs blackout duration\n(band = median to p90)",
                      fontsize=10)
    axes[1].axhline(10.0, color="#888", ls="--", lw=1)
    axes[1].set_xlabel("GNSS blackout duration (s)")
    axes[1].set_ylabel("Drift (% of distance travelled)")
    axes[1].set_title("Drift as a fraction of distance travelled", fontsize=10)
    for a in axes:
        a.grid(alpha=0.25, ls=":")
        a.legend(fontsize=8)
        a.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_speed_trace(pred: dict, run_id: str, out: Path, max_s: float = 300.0) -> None:
    win_dt = pred["win_dt"]
    k = min(int(max_s / win_dt), len(pred["starts"]))
    t = np.arange(k) * win_dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    axes[0].plot(t, pred["true_speed"][:k], color=C_GT, lw=1.8, label="True speed")
    axes[0].plot(t, pred["pred_speed"][:k], color=C_MODEL, lw=1.4,
                 label="Predicted speed")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, np.rad2deg(pred["true_dpsi"][:k]), color=C_GT, lw=1.8,
                 label="True heading change")
    axes[1].plot(t, np.rad2deg(pred["pred_dpsi"][:k]), color=C_MODEL, lw=1.4,
                 label="Predicted heading change")
    axes[1].set_ylabel(f"Δψ per {win_dt:.1f} s window (deg)")
    axes[1].set_xlabel("Time into run (s)")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25, ls=":")
    axes[0].set_title(f"{run_id} — regressor output vs vehicle ground truth",
                      fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=MODEL_DIR / "tcn_best.pt")
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-plots", type=int, default=4)
    ap.add_argument("--alpha-max", type=float, default=ALPHA_MAX_DEFAULT)
    ap.add_argument("--tau", type=float, default=ALPHA_TAU_S_DEFAULT,
                    help="gain ramp time constant, s; <=0 for a constant gain")
    ap.add_argument("--rotate-eval", type=int, default=None,
                    help="apply a simulated mount rotation (seed) at eval, to "
                         "measure how much the model leans on raw body axes")
    ap.add_argument("--max-tilt-deg", type=float, default=60.0)
    ap.add_argument("--sweep-alpha", action="store_true",
                    help="sweep alpha on this split and exit (use --split val)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, win, out_win, chan_idx = load_model(args.ckpt, device)
    rotation = None
    if args.rotate_eval is not None:
        from .augment import tilt_rotation
        rotation = tilt_rotation(np.random.default_rng(args.rotate_eval),
                                 args.max_tilt_deg)
        print(f"SIMULATED MOUNT ROTATION seed={args.rotate_eval} "
              f"(tilt up to {args.max_tilt_deg:.0f} deg)")

    runs, _ = load_index(args.proc_dir)
    if args.split == "all":
        sel = runs
    else:
        sel = split_runs(runs, make_splits(runs))[args.split]
    if not sel:
        print(f"no runs in split '{args.split}'")
        return 1
    print(f"evaluating {len(sel)} runs from split '{args.split}' "
          f"(context {win*DT:.1f}s, output interval {out_win*DT:.1f}s)")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep_alpha:
        # Grid over (alpha_max, tau). tau <= 0 reproduces a constant gain, so the
        # constant-blend baselines are included in the same table.
        grid = [(a, 0.0) for a in (0.0, 0.05, 0.2, 0.5, 1.0)]
        grid += [(a, t) for a in (0.5, 1.0) for t in (5.0, 10.0, 20.0, 40.0, 80.0)]

        cache = []
        for run in sel:
            arrays = run.load()
            span = longest_valid_span(arrays["valid"])
            if span[1] - span[0] < win + out_win * 20:
                continue
            pr = window_predictions(model, arrays, win, out_win, span, device,
                                    channels=chan_idx, rotation=rotation)
            if pr:
                cache.append((pr, arrays))
        if not cache:
            print("nothing to sweep")
            return 1

        print(f"\ngain sweep on split '{args.split}'"
              f"{'' if args.split == 'val' else '  WARNING: tune on val, not test'}")
        print(f"{'a_max':>6} {'tau':>6} " +
              " ".join(f"{int(T)}s med".rjust(9) for T in BLACKOUT_S) +
              f" {'30s p90':>9}")
        # Selection objective: the median error at 30 s, which is the roadmap's
        # stated target (<10 m at 30 s). Averaging across durations lets the
        # 120 s numbers dominate and picks a gain that is bad where it matters.
        best = None
        for a_max, tau in grid:
            rows = []
            for pr, arrays in cache:
                rows += blackout_errors(pr, arrays, alpha_max=a_max, tau_s=tau)
            meds = [float(np.median([r["model_err_m"] for r in rows
                                     if r["duration_s"] == T])) for T in BLACKOUT_S]
            p90_30 = float(np.percentile([r["model_err_m"] for r in rows
                                          if r["duration_s"] == 30.0], 90))
            score = meds[BLACKOUT_S.index(30.0)]
            mark = ""
            if best is None or score < best[2]:
                best, mark = (a_max, tau, score), "   <-- best @30s"
            print(f"{a_max:6.2f} {tau:6.1f} " +
                  " ".join(f"{m:9.2f}" for m in meds) + f" {p90_30:9.2f}{mark}")

        print(f"\nbest: alpha_max {best[0]}, tau {best[1]} s "
              f"-> {best[2]:.2f} m median at 30 s")
        print("Set ALPHA_MAX_DEFAULT / ALPHA_TAU_S_DEFAULT, or pass --alpha-max/--tau.")
        return 0

    all_rows, per_run, n_plots = [], [], 0
    for run in sel:
        arrays = run.load()
        span = longest_valid_span(arrays["valid"])
        if span[1] - span[0] < win + out_win * 20:
            print(f"  skip {run.run_id}: no usable span")
            continue
        pred = window_predictions(model, arrays, win, out_win, span, device,
                                  channels=chan_idx, rotation=rotation)
        if not pred:
            continue

        rows = blackout_errors(pred, arrays, alpha_max=args.alpha_max,
                               tau_s=args.tau)
        for r in rows:
            r["run_id"] = run.run_id
            r["route"] = run.route
        all_rows += rows

        sp_mae = float(np.abs(pred["pred_speed"] - pred["true_speed"]).mean())
        dp_mae = float(np.abs(pred["pred_dpsi"] - pred["true_dpsi"]).mean())
        gy_mae = float(np.abs(pred["gyro_dpsi"] - pred["true_dpsi"]).mean())
        per_run.append({
            "run_id": run.run_id, "route": run.route,
            "span_s": round((span[1] - span[0]) * DT, 1),
            "n_windows": len(pred["starts"]),
            "speed_mae_ms": round(sp_mae, 4),
            "dpsi_mae_deg": round(float(np.rad2deg(dp_mae)), 4),
            "gyro_dpsi_mae_deg": round(float(np.rad2deg(gy_mae)), 4),
        })
        print(f"  {run.run_id:14} {per_run[-1]['span_s']:7.0f}s  "
              f"speed MAE {sp_mae:.3f} m/s  dpsi MAE {np.rad2deg(dp_mae):.2f} deg "
              f"(gyro-only {np.rad2deg(gy_mae):.2f} deg)", flush=True)

        if n_plots < args.max_plots:
            tag = run.run_id.replace("#", "_")
            plot_blackout_segments(pred, arrays, run.run_id,
                                   PLOT_DIR / f"blackouts60_{tag}.png",
                                   duration_s=60.0)
            plot_blackout_segments(pred, arrays, run.run_id,
                                   PLOT_DIR / f"blackouts30_{tag}.png",
                                   duration_s=30.0)
            plot_trajectory(pred, arrays, run.run_id,
                            PLOT_DIR / f"traj_{run.run_id.replace('#','_')}.png")
            plot_speed_trace(pred, run.run_id,
                             PLOT_DIR / f"speed_{run.run_id.replace('#','_')}.png")
            n_plots += 1

    if not all_rows:
        print("no blackout windows evaluated")
        return 1

    import csv
    with (METRIC_DIR / "eval_per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)
    with (METRIC_DIR / "eval_blackouts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    plot_blackout_curve(all_rows, PLOT_DIR / "blackout_error.png")

    print(f"\ngain: alpha_max={args.alpha_max} tau={args.tau}s")
    print(f"{'dur':>6} {'n':>5} | "
          f"{'model med':>10} {'model p90':>10} {'model %':>8} | "
          f"{'abs-only':>9} | {'base med':>9} {'base %':>7} | {'oracle med':>10}")
    summary = []
    for T in BLACKOUT_S:
        sub = [r for r in all_rows if r["duration_s"] == T]
        if not sub:
            continue
        # `sub` is bound as a default rather than captured: closing over a
        # loop variable is a bug waiting for someone to defer the call.
        def g(k, f=np.median, sub=sub):
            return float(f([r[k] for r in sub]))
        row = {
            "duration_s": T, "n": len(sub),
            "model_err_med_m": round(g("model_err_m"), 2),
            "model_err_p90_m": round(
                g("model_err_m", lambda a: np.percentile(a, 90)), 2),
            "model_drift_med_pct": round(g("model_drift_pct"), 3),
            "model_abs_err_med_m": round(g("model_abs_err_m"), 2),
            "baseline_err_med_m": round(g("baseline_err_m"), 2),
            "baseline_drift_med_pct": round(g("baseline_drift_pct"), 3),
            "oracle_err_med_m": round(g("oracle_err_m"), 2),
            "oracle_drift_med_pct": round(g("oracle_drift_pct"), 3),
        }
        summary.append(row)
        print(f"{T:6.0f} {len(sub):5d} | {row['model_err_med_m']:10.2f} "
              f"{row['model_err_p90_m']:10.2f} {row['model_drift_med_pct']:7.2f}% | "
              f"{row['model_abs_err_med_m']:9.2f} | "
              f"{row['baseline_err_med_m']:9.2f} "
              f"{row['baseline_drift_med_pct']:6.2f}% | "
              f"{row['oracle_err_med_m']:10.2f}")

    (METRIC_DIR / "eval_summary.json").write_text(json.dumps(
        {"split": args.split, "win": win, "out_win": out_win,
         "alpha_max": args.alpha_max, "tau_s": args.tau,
         "n_runs": len(per_run),
         "blackout_summary": summary,
         "speed_mae_ms_mean": round(
             float(np.mean([r["speed_mae_ms"] for r in per_run])), 4),
         "dpsi_mae_deg_mean": round(
             float(np.mean([r["dpsi_mae_deg"] for r in per_run])), 4),
         "gyro_dpsi_mae_deg_mean": round(
             float(np.mean([r["gyro_dpsi_mae_deg"] for r in per_run])), 4),
         }, indent=2))
    print(f"\n-> {METRIC_DIR}/eval_*.csv, {PLOT_DIR}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
