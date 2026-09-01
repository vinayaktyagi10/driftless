"""Allan-deviation fit of IMU noise densities from real IO-VNBD logs.

Why this exists
---------------
`edge-engine/include/driftless/imu_noise.h` carries datasheet-class placeholders
and says so:

    // TODO: fit from Allan deviation once IO-VNBD / FOG logs are in hand.

and notes that Q "is the single biggest lever on how fast the covariance inflates
during a blackout". We hold the handset logs, so the handset half of that TODO is
ours to close. This produces measured values for
`ImuNoiseParams::consumerMems()`.

Method
------
For a signal sampled at 1/dt, the overlapping Allan deviation sigma(tau) of the
*rate* signal has two regimes we care about:

  * White noise (angle/velocity random walk) falls as tau^-1/2. Reading the
    curve at tau = 1 s gives the noise density directly:
        N = sigma(1 s)                 [rad/s/sqrt(Hz)] or [m/s^2/sqrt(Hz)]
  * Bias instability (rate random walk) rises as tau^+1/2, with
        K = sigma(tau) * sqrt(3/tau)   [rad/s^2/sqrt(Hz)] or [m/s^3/sqrt(Hz)]

We fit each slope over a decade where it dominates and report both, plus the
bias-instability floor for context.

Honest caveats, which belong in the report rather than buried here:
  * These logs are 10 Hz, so the white-noise region is only observable for
    tau >= 0.2 s. Anything faster than that is aliased and this fit cannot see
    it -- a 100 Hz capture would pin N better.
  * The vehicle is MOVING. Allan deviation assumes a stationary sensor; real
    dynamics inflate the estimate. We therefore fit only over spans the vehicle
    reports as stopped, which is what makes this usable at all.
  * This gives the HANDSET (consumerMems) parameters only. The FOG unit for the
    edge engine needs its own logs; nothing here speaks to it.

Run:  python -m driftless_train.allan
Out:  artifacts/metrics/allan_imu_noise.json  + .md
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .paths import METRIC_DIR, PROC_DIR
from .preprocess import FEATURE_CHANNELS

# Channel triples to characterise, in physical XYZ order.
ACC_CH = ("acc_x", "acc_y", "acc_z")
GYR_CH = ("gyro_yaw", "gyro_roll", "gyro_pitch")   # physical X, Y, Z

MIN_STATIONARY_S = 20.0     # shorter spans cannot resolve the tau range
STATIONARY_SPEED = 0.3      # m/s; below this the vehicle is stopped


def overlapping_allan(x: np.ndarray, dt: float,
                      taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal.

    Returns (tau_used, sigma). Implemented on the integrated signal, which is the
    standard formulation and avoids re-averaging the rate series per tau.
    """
    theta = np.cumsum(x) * dt          # integrate rate -> angle/velocity
    n = len(theta)
    out_t, out_s = [], []
    for tau in taus:
        m = int(round(tau / dt))
        if m < 1 or n < 3 * m:
            continue
        # sigma^2 = 1/(2 tau^2 (n-2m)) * sum (theta[i+2m] - 2 theta[i+m] + theta[i])^2
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        var = np.sum(d * d) / (2.0 * tau * tau * len(d))
        if var > 0:
            out_t.append(tau)
            out_s.append(np.sqrt(var))
    return np.asarray(out_t), np.asarray(out_s)


# How far the measured log-log slope may sit from the model's slope before the
# fit is refused. A fixed-slope fit to data of the wrong slope returns a number
# with no meaning, which is worse than returning nothing.
SLOPE_TOLERANCE = 0.25


def _fit_power(taus: np.ndarray, sigma: np.ndarray, slope: float,
               lo: float, hi: float) -> dict:
    """Fixed-slope fit in log-log space, WITH a validity check on the slope.

    The level is fitted at the model's slope (with ~one usable decade of tau
    there is not enough leverage to identify slope and level at once, and the
    physics fixes the slope). But the observed slope is measured independently
    and the fit is REJECTED when the two disagree -- otherwise this happily
    reports a "bias random walk" from a region whose slope is -1/2, which is
    exactly the mistake this function used to make.

    Returns {value, observed_slope, n_points, identifiable, reason}.
    """
    m = (taus >= lo) & (taus <= hi)
    n = int(m.sum())
    if n < 3:
        return {"value": None, "observed_slope": None, "n_points": n,
                "identifiable": False,
                "reason": f"only {n} tau points in [{lo}, {hi}] s"}

    lt, ls = np.log(taus[m]), np.log(sigma[m])
    observed = float(np.polyfit(lt, ls, 1)[0])

    if abs(observed - slope) > SLOPE_TOLERANCE:
        return {"value": None, "observed_slope": round(observed, 3),
                "n_points": n, "identifiable": False,
                "reason": (f"observed log-log slope {observed:+.2f} is "
                           f"incompatible with the {slope:+.2f} this parameter "
                           f"assumes; the regime is not present in this data")}

    log_c = float(np.mean(ls - slope * lt))
    return {"value": float(np.exp(log_c)), "observed_slope": round(observed, 3),
            "n_points": n, "identifiable": True, "reason": ""}


def stationary_spans(speed: np.ndarray, valid: np.ndarray, dt: float,
                     min_s: float = MIN_STATIONARY_S) -> list[tuple[int, int]]:
    """Contiguous spans where the vehicle is stopped and the data is valid."""
    still = (speed < STATIONARY_SPEED) & valid
    spans, i, n = [], 0, len(still)
    need = int(min_s / dt)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        if j - i >= need:
            spans.append((i, j))
        i = j
    return spans


def characterise(dt: float = 0.1) -> dict:
    idx = json.loads((PROC_DIR / "index.json").read_text())
    ch = {c: i for i, c in enumerate(FEATURE_CHANNELS)}

    segments: list[np.ndarray] = []
    total_s = 0.0
    n_runs = 0
    for r in idx["runs"]:
        f = PROC_DIR / f"{r['run_id']}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            feats, speed, valid = z["features"], z["speed_ms"], z["valid"]
        spans = stationary_spans(speed.astype(float), valid, dt)
        if spans:
            n_runs += 1
        for a, b in spans:
            segments.append(feats[a:b])
            total_s += (b - a) * dt

    if not segments:
        raise SystemExit("no stationary spans found")

    taus = np.unique(np.round(np.logspace(np.log10(2 * dt), np.log10(60.0), 40)
                              / dt).astype(int)) * dt

    result = {
        "source": "IO-VNBD smartphone (S-*) runs, stationary spans only",
        "sample_rate_hz": round(1.0 / dt, 3),
        "n_runs_contributing": n_runs,
        "n_segments": len(segments),
        "stationary_seconds": round(total_s, 1),
        "min_segment_s": MIN_STATIONARY_S,
        "axes": {},
    }

    for label, names, unit_n, unit_k in (
        ("accel", ACC_CH, "m/s^2/sqrt(Hz)", "m/s^3/sqrt(Hz)"),
        ("gyro", GYR_CH, "rad/s/sqrt(Hz)", "rad/s^2/sqrt(Hz)"),
    ):
        per_axis_n, per_axis_k = [], []
        curves = {}
        for axis, name in zip("xyz", names, strict=True):
            # Average the Allan variance across segments at each tau, weighting
            # by segment length -- concatenating the segments would inject a
            # step discontinuity at every join.
            acc_num: dict[float, list[float]] = {}
            for seg in segments:
                x = seg[:, ch[name]].astype(float)
                x = x - x.mean()            # a constant bias is not noise
                t, s = overlapping_allan(x, dt, taus)
                for ti, si in zip(t, s, strict=True):
                    acc_num.setdefault(float(ti), []).append(si * si)
            if not acc_num:
                continue
            t_all = np.array(sorted(acc_num))
            s_all = np.array([np.sqrt(np.mean(acc_num[t])) for t in t_all])

            nfit = _fit_power(t_all, s_all, -0.5, 0.2, 2.0)
            kfit = _fit_power(t_all, s_all, +0.5, 10.0, 60.0)
            N = nfit["value"]
            K = None if kfit["value"] is None else kfit["value"] * np.sqrt(3.0)

            if N is not None:
                per_axis_n.append(N)
            if K is not None:
                per_axis_k.append(K)
            curves[axis] = {
                "tau_s": [round(float(v), 4) for v in t_all],
                "sigma": [float(f"{v:.6g}") for v in s_all],
                "noise_density": None if N is None else float(f"{N:.4g}"),
                "noise_density_fit": nfit,
                "bias_random_walk": None if K is None else float(f"{K:.4g}"),
                "bias_random_walk_fit": kfit,
                "min_sigma": float(f"{s_all.min():.4g}"),
                "tau_at_min_s": float(t_all[int(np.argmin(s_all))]),
            }

        result["axes"][label] = {
            "unit_noise_density": unit_n,
            "unit_bias_random_walk": unit_k,
            "per_axis": curves,
            "noise_density_mean": (float(f"{np.mean(per_axis_n):.4g}")
                                   if per_axis_n else None),
            "bias_random_walk_mean": (float(f"{np.mean(per_axis_k):.4g}")
                                      if per_axis_k else None),
        }
    return result


def write_markdown(res: dict, path) -> None:
    a = res["axes"]["accel"]
    g = res["axes"]["gyro"]

    def line(blk, key, name, unit):
        v = blk[f"{key}_mean"]
        if v is None:
            return f"| `{name}` | **not identifiable** | {unit} |"
        return f"| `{name}` | {v} | {unit} |"

    L = [
        "# IMU noise characterisation from IO-VNBD (handset)",
        "",
        f"Measured over **{res['stationary_seconds']:.0f} s** of stationary data "
        f"in **{res['n_segments']} spans** across {res['n_runs_contributing']} "
        f"runs, at {res['sample_rate_hz']} Hz. Overlapping Allan deviation.",
        "",
        "Addresses the handset half of the TODO in "
        "`edge-engine/include/driftless/imu_noise.h`. **Read the caveats before "
        "using any of these** -- two of the four parameters turned out not to be "
        "identifiable from this data at all, and reporting a number for them "
        "would have been worse than reporting nothing.",
        "",
        "## What this data can and cannot pin down",
        "",
        "| parameter | fitted value | unit |",
        "|---|---|---|",
        line(a, "noise_density", "accel_noise_density", a["unit_noise_density"]),
        line(g, "noise_density", "gyro_noise_density", g["unit_noise_density"]),
        line(a, "bias_random_walk", "accel_bias_random_walk",
             a["unit_bias_random_walk"]),
        line(g, "bias_random_walk", "gyro_bias_random_walk",
             g["unit_bias_random_walk"]),
        "",
        "### Why parameters are rejected rather than reported",
        "",
        "A fixed-slope fit returns a number whatever the data looks like, so the "
        "observed log-log slope is measured independently and the fit is refused "
        "when the two disagree (tolerance "
        f"{SLOPE_TOLERANCE}). Observed slopes:",
        "",
        "| sensor | axis | white-noise region (expect -0.50) | bias-instability "
        "region (expect +0.50) |",
        "|---|---|---|---|",
    ]
    for label, blk in (("accel", a), ("gyro", g)):
        for axis, c in blk["per_axis"].items():
            ns = c["noise_density_fit"]["observed_slope"]
            ks = c["bias_random_walk_fit"]["observed_slope"]
            L.append(f"| {label} | {axis} | {ns} | {ks} |")

    L += [
        "",
        "**Bias random walk is not observable here.** In the 10-60 s window "
        "where bias instability should make the curve RISE at +1/2, it is still "
        "falling at about -0.5 for every axis. The bias-instability floor simply "
        "has not been reached within the longest stationary spans available "
        "(minimum span "
        f"{res['min_segment_s']:.0f} s). Estimating it needs stationary records "
        "of ~10 minutes or more.",
        "",
        "**The gyro white-noise fit is contaminated.** Its Allan deviation is "
        "flat-to-rising at short tau rather than falling at -1/2, which is the "
        "signature of a correlated/periodic disturbance, not white noise. The "
        "vehicle is stopped but the engine is running, so the handset is sitting "
        "in idle vibration.",
        "",
        "## Caveats that must travel with these numbers",
        "",
        "- **The engine is running.** 'Stationary' here means the vehicle is not "
        "moving; it does not mean the sensor is at rest. Idle vibration inflates "
        "every figure, so treat these as UPPER BOUNDS on sensor noise -- though "
        "arguably closer to what the filter meets in service than a datasheet "
        "value is.",
        "- **10 Hz logs.** The white-noise region is only observable for "
        "tau >= 0.2 s, and higher-frequency vibration energy aliases down into "
        "that band, inflating it further.",
        "- **Handset only.** Nothing here speaks to the FOG unit; `fogGrade()` "
        "still needs its own logs.",
        "- One handset model, one vehicle.",
        "",
        "## The 10-minute fix",
        "",
        "Everything above is limited by the data, not the method. A proper "
        "characterisation needs one capture that nobody has taken yet:",
        "",
        "1. Phone flat on a desk, **engine off**, no one touching it.",
        "2. Log accelerometer and gyroscope at the highest rate the handset "
        "offers (`SENSOR_DELAY_FASTEST`, typically 100-500 Hz).",
        "3. **10 minutes minimum** -- long enough to reach the bias-instability "
        "floor.",
        "",
        "Then rerun `python -m driftless_train.allan` against that log and all "
        "four parameters become identifiable. This is a role 01 task with the "
        "capture app and takes longer to read about than to do.",
    ]
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args(argv)

    res = characterise(args.dt)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "allan_imu_noise.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, METRIC_DIR / "allan_imu_noise.md")

    a, g = res["axes"]["accel"], res["axes"]["gyro"]
    print(f"stationary data: {res['stationary_seconds']:.0f} s in "
          f"{res['n_segments']} spans")
    for label, blk, key in (("accel noise density", a, "noise_density"),
                            ("gyro  noise density", g, "noise_density"),
                            ("accel bias rand walk", a, "bias_random_walk"),
                            ("gyro  bias rand walk", g, "bias_random_walk")):
        v = blk[f"{key}_mean"]
        unit = blk["unit_noise_density" if key == "noise_density"
                   else "unit_bias_random_walk"]
        if v is None:
            axis0 = next(iter(blk["per_axis"].values()))
            print(f"  {label:22} NOT IDENTIFIABLE - "
                  f"{axis0[f'{key}_fit']['reason']}")
        else:
            print(f"  {label:22} {v} {unit}")
    print(f"-> {METRIC_DIR/'allan_imu_noise.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
