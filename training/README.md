# Driftless — training (role 03: data & model)

SIH 2026 · PS #26168 (ISRO) · GNSS-denied smartphone navigation

This is the data and model half of Driftless: it turns the public **IO-VNBD**
driving dataset into a trained **speed & heading-change regressor**, exports it
for both runtimes (TFLite for the Android app, ONNX for the C++ edge engine),
and produces the **dead-reckoning position plots** that Round 1 is screened on.

```
download → audit → pair (S+V) → prepare → train → evaluate → export
```

## Quickstart

```bash
cd training
uv venv --python 3.12 .venv          # or any Python 3.10-3.12 venv
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=.

python -m driftless_train.download    # both S and V sides, LFS-aware (~425 MB)
python -m driftless_train.audit       # per-run inventory + usability gates
python -m driftless_train.prepare     # pair S with V, materialise arrays
python -m driftless_train.train --epochs 40
python -m driftless_train.evaluate --split val --sweep-alpha   # tune the blend
python -m driftless_train.evaluate --split test                # report numbers
python -m driftless_train.export      # ONNX + TFLite, with parity checks
python -m driftless_train.crossval --k 5   # route-wise CV: the honest headline
python -m driftless_train.sensor_tier      # does the model transfer to a cleaner IMU?
python -m driftless_train.noise_params     # measurement noise for role 02's UKF
python -m driftless_train.allan            # IMU noise characterisation
python -m driftless_train.report      # assemble ROUND1_EVIDENCE.md
pytest tests/ -q                      # pins every dataset trap below
```

**You do not need to run any of this to use the model.** The verified exports are
committed under `artifacts/models/` — see *Handover to the other roles* below.
Everything above is only needed to retrain or to reproduce the numbers.

## What IO-VNBD actually is (measured, not assumed)

The dataset ships as 564 Git-LFS CSVs. `raw.githubusercontent.com` serves
130-byte LFS pointers, not data; real bytes come from
`media.githubusercontent.com/media/...`. The same 72 drives appear **twice**,
under `Categorised` (with driver/route folders) and `Uncategorised` (flat) — we
keep the categorised copy because only it identifies the driver and route
family, which is what route-wise splitting needs.

Each drive has two sides, and in the *Synchronised* set they share a clock:

| | **S-\*.csv** (smartphone) | **V-\*.csv** (vehicle) |
|---|---|---|
| Columns | 24 | 29 |
| Rate | 10 Hz | 10 Hz |
| Position | GNSS, **held for 9 s**, 4–6 dp | survey GNSS, **every sample**, 7 dp |
| Speed | GNSS Doppler, held 9 s | CAN velocity + 4 wheel speeds |
| Rotation | accel / gyro / mag / gravity | yaw rate, steering angle |
| Role here | **model input** | **ground truth** |

### Six traps found in this data, and what each would have cost

1. **The gyroscope axis labels do not mean what they say.** This was the most
   expensive one. Correlated against the paired vehicle CAN yaw rate on every
   available run:

   | column | corr with vehicle yaw rate |
   |---|---|
   | `GYROSCOPE Yaw` | −0.09 … +0.69 (inconsistent) |
   | **`GYROSCOPE Pitch`** | **+0.977 … +0.987 (every run)** |
   | `GYROSCOPE Roll` | −0.79 … +0.59 (sign flips between runs) |

   The column headed *Pitch* is the vertical-axis (yaw) rate. The accelerometer
   *is* in (X, Y, Z) order with Z vertical, so projecting the gyro triple onto
   gravity in naive (Yaw, Pitch, Roll) order silently picks up *Roll* — noise
   with no stable sign relationship to the vehicle's turning. Consequences before
   this was found: the single most informative channel in the dataset was fed to
   the model as garbage, heading error was 3–5× worse, and the phone/vehicle
   consistency check rejected perfectly good runs as "badly mounted". With the
   correct axis every run scores **+0.85 to +0.999**. Relatedly, the `GRAVITY`
   columns are near-constant at (0, 0, 9.8066) — as few as 18 distinct values in
   a whole run — so attitude is taken from a low-passed accelerometer instead.

2. **The S file's speed header lies about its unit.** The column reads
   `GPS SPEED (Kmh)`; the values are **m/s**. Verified against
   position-differentiated speed on S-M/S-S1/S-S2 — median ratio **1.000**
   (a real km/h column would give 3.6). Trusting the header scales every speed
   label by 3.6×. Meanwhile the V file's `Velocity (km/hr)` genuinely *is* km/h
   (ratio 0.2778). The dataset is not internally consistent; both are now pinned
   by unit tests.

3. **Files contain multiple runs concatenated**, marked by `TIME SINCE START`
   resetting to ~0 (S-M is 4422 s + 6176 s; S-S2 is 186 s + 9201 s). Sorting a
   file by timestamp instead of splitting it interleaves two physically separate
   drives and fabricates kilometre-scale jumps — it inflated S-M's path length
   to **648,000 km** before this was caught. We cut at resets and never sort
   across one.

4. **The phone's own GNSS is too coarse to label a 2 s window** — it updates
   every 9.00 s, held constant in between. This is why the pipeline pairs S with
   V rather than training on the phone file alone.

5. **GNSS heading is meaningless at standstill.** Across stationary 2 s windows
   its heading difference has std **39°** and reaches **197°**, because
   course-over-ground spins freely when not moving. So the heading-change target
   is the **integral of the vehicle's yaw rate**, which is bounded everywhere and
   agrees with GNSS heading to **1.07° (corr 0.989)** on moving windows. Using
   the naive heading difference tripled the heading error (3.57° → 1.35° MAE).

6. **Vehicle yaw rate is sign-inverted** relative to compass heading (ISO
   convention: positive left, vs bearing increasing clockwise). Measured ratio
   to d(heading)/dt is **−1.005**. The sign is re-measured per run rather than
   hard-coded, so a differently-wired logger cannot silently invert every label.

### The dataset's biggest limitation for our purpose

With the gyro axis corrected, phone/vehicle coupling — how well the phone's own
gyro tracks the car's measured yaw rate — tracks **the driver, not the route**:

| driver | families | routes | runs | runs ≥ 0.70 | coupling range | hours |
|---|---|---|---|---|---|---|
| A | S | 6 | 9 | **9 / 9** | 0.76 – 1.00 | 8.48 |
| B | M | 1 | 2 | **2 / 2** | 0.99 – 1.00 | 2.94 |
| D | Y | 1 | 1 | **1 / 1** | 0.96 | 1.86 |
| E | Vta, Vtb, Vw, Vf | 64 | 39 | **4 / 39** | 0.21 – 0.89 | 14.68 |

It is a strong association, **not a clean partition** — and an earlier version of
this file overstated it as one. Every A/B/D run is well coupled (the weakest is
0.76), and 35 of driver E's 39 runs fall below the 0.70 threshold. But four short
driver-E runs do pass, scoring 0.74–0.89 and totalling 0.38 h, so they sit in the
trusted pool alongside A/B/D. Phone accelerometer noise (|acc| std) is 0.52–0.78
for A/B/D against 0.81–1.07 for driver E.

Driver E's phone is flat like the others (mean accelerometer direction ≈
(0, 0, 1), so no tilt to correct for), and a wide ±300 s lag search finds no
better alignment — so this is not a clock problem. Combined with roughly double
the accelerometer noise, the reading is a phone lying loose and sliding rather
than mounted.

**So only 8 of 72 routes have a phone rigidly enough coupled to learn vehicle
dynamics from** — 13.7 h of the 28 h total. This is the single strongest
argument for the team's own data collection, and role 06 should not quote
"4,400 km of smartphone data" as if it were all usable for this task.

Weakly-coupled runs are kept but confined to **training only** (`TRUSTED_
COUPLING_CORR = 0.70` in `pair.py`): their heading labels are close to
unlearnable, but their speed cues may not be, and speed is where our error lives.
Nothing weakly coupled ever appears in val or test, because our product assumes a
mounted phone and evaluating on a sliding one would measure the wrong thing.
Whether the extra data actually helps is a testable question, not an assumption —
`train.py --min-coupling` runs it both ways. Trained identically and evaluated on
the same rigidly-coupled held-out route:

*Measured on the pre-augmentation model (centred gravity filter, no rotation
augmentation), which is the configuration the comparison was run under. The
absolute numbers are therefore superseded by the results table above; the
comparison between the two rows is what this table is for, and that conclusion
still stands.*

| training data | speed MAE | Δψ MAE | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|---|
| **trusted only (9.7 h)** | **1.543 m/s** | **1.06°** | **7.7 m** | 33.9 m | 69.7 m | **125.6 m** |
| + 14.3 h weakly coupled | 1.666 m/s | 1.19° | 9.0 m | **32.8 m** | **64.6 m** | 126.3 m |

It is close to a wash: trusted-only wins both per-window metrics and the 10 s
blackout, the weak data helps a little at 30–60 s. We default to trusted-only —
better where blackouts are most common, and a simpler thing to defend — but the
extra data is there behind a flag for anyone who wants it.

Note that *validation loss* prefers the all-data model (0.063 vs 0.073) while the
dead-reckoning metric does not. The window-level loss is not the objective;
position error after a blackout is.

One bug this experiment surfaced: weak routes were originally added to train
*and counted toward its duration budget*, which filled train's 70 % share and
pushed the scarce trusted routes out into val/test — leaving only 5.5 h of the
13.7 h of trusted data for training. The 70/15/15 target now applies to trusted
data only, with weak routes extra on top, which nearly doubled trusted training
data to 9.7 h.

### How S and V are aligned

Alignment is measured, never assumed:

1. **Coarse** — S timestamps are local, V is UTC seconds-of-day. The offset is
   recovered and snapped to a quarter-hour (covers UK/France/Nigeria, and India's
   +5:30 for our own captures later). On route S1: exactly **3600 s**.
2. **Fine** — cross-correlate phone gyro-about-vertical against vehicle yaw
   rate. These are the same physical quantity from two independent devices, so
   the peak is the true residual lag. On S1: **+0.10 s at corr −0.74** (negative
   because of the sign convention in trap 5).
3. The peak correlation is kept as a QC score. A wide ±300 s search confirmed
   that failing runs peak at lag 0 — so a low score never meant a clock error;
   with the gyro axis corrected, every run scores 0.85–0.999 and none are
   rejected.

Sanity check that validates the whole target set at once: integrating the *true*
speed and heading reproduces the *true* trajectory to **0.212 % drift over
38.07 km**. That confirms units, the compass convention (east = sin, north =
cos) and target coherence together — and it is the floor any model is measured
against. Independently, wheel-speed × radius agrees with CAN speed to
**0.096 m/s**, and the implied tyre radius is **0.276 m** — physically right for
a real car.

## Model

A small **dilated causal TCN**: 14 input channels × **80 samples (8 s of context
at 10 Hz)** → three outputs. **38,499 parameters**; inference **0.118 ms/window**
(ONNX) and **0.067 ms/window** (TFLite) on laptop CPU. Receptive field 63
samples, covering the context.

**Context length and output interval are deliberately decoupled** — the two
targets want opposite things. Measured on a fixed-budget sweep:

| context | speed MAE | Δψ MAE |
|---|---|---|
| 2 s | 2.68 m/s | 0.86° |
| 4 s | 2.35 m/s | 1.38° |
| **8 s context → 2 s output** | **1.96 m/s** | **0.87°** |

Absolute speed is barely observable from 2 s of accelerometer (it leans on
road/tyre vibration and sustained dynamics), while heading change wants a short
output interval — both for accuracy and because role 02's EKF consumes it as a
per-step increment. Feeding long context but predicting only the final 2 s gets
both.

### The three outputs, and why speed is predicted twice

| output | unit | why |
|---|---|---|
| `speed_ms` | m/s | mean speed over the interval — absolute, but weakly observable |
| `dpsi_rad` | rad | heading change across the interval |
| `dv_ms` | m/s | **change** in speed — directly observable, MAE **0.45 m/s** vs 1.94 m/s for the absolute head |

A blackout begins at a *known* speed — whatever the last GNSS fix reported — so
propagating `v += dv` from it is far better conditioned than predicting absolute
speed from scratch. Pure propagation drifts without bound, so the two are blended
with a **time-varying gain**, ramping from the propagated estimate toward the
absolute head as the fix ages. Measured on val:

| blend | 10 s | 30 s | 120 s |
|---|---|---|---|
| pure `dv` integration | **7.94 m** | 44.2 m | 291.2 m |
| pure absolute head | 13.41 m | **36.0 m** | 147.9 m |
| **ramp (τ = 20 s)** | **8.95 m** | 36.8 m | **144.5 m** |

That the optimum is time-varying rather than constant is the argument for role 02
owning this in the EKF, where a Kalman gain responds to growing process
covariance automatically instead of us hand-tuning τ.

### Mount-rotation augmentation (on by default)

Every phone in IO-VNBD lay flat — mean accelerometer direction ≈ (0, 0, 1) in all
51 runs — while the product puts one in a dashboard mount at an arbitrary angle.
Nine of the fourteen channels are raw body axes, so they go out of distribution
the moment the phone is tilted. Measured on the held-out route, applying a
simulated mount rotation (random azimuth, tilt up to 60°):

| training | unrotated 30 s | rotated 30 s | degradation | 60 s |
|---|---|---|---|---|
| 14 ch, no augmentation | 33.8 m | **186.5 m** | **5.5×** (speed MAE 26.8×) | 70.0 m |
| 5 invariant channels only | 37.5 m | 37.5 m | **1.000× — bit-identical** | 70.3 m |
| **14 ch + rotation augmentation** | **32.8 m** | 33.3 m | **1.01×** | **59.1 m** |

Untreated, the model is worse than the no-ML baseline once the phone is tilted.
Augmentation removes that **and improves accuracy** — 60 s error fell 16 % — so it
is the default (`train.py --rotate-aug`). The 5-channel invariant subset
(`--invariant-only`) is kept as a fallback with a *provable* guarantee rather
than an empirical one: the gravity-projected channels are exactly invariant
because a fixed mount rotation commutes with the (linear) gravity filter. That
identity is asserted on real data in `tests/test_augment.py`.

Three further properties it is built to have:

- **Causal.** A window ending now uses only samples ≤ now, because the phone
  cannot see the future. This was briefly *not* true: gravity was estimated with
  a centred `np.convolve(..., mode="same")`, which averaged ~10 s of future
  samples into the current row. It is now a causal one-pole EMA — what a handset
  could actually run — with the settling transient marked invalid so no window
  trains on it. Two tests pin it, one textual and one behavioural (perturb a
  future sample, assert earlier outputs are unchanged).
- **Attitude-invariant inputs.** A dashboard-mounted phone sits at an arbitrary
  angle, so alongside the 9 raw channels we feed 5 derived ones computed by
  projecting onto the measured gravity direction: `acc_norm`, `acc_vert`,
  `acc_horiz`, **`gyro_vert`** (yaw rate about *true* vertical — the heading
  driver), `gyro_horiz`. On IO-VNBD this degenerates to axis selection because
  the phone is flat in every run, but it is what makes the same code correct for
  our own tilted-mount captures.
- **SI units at the boundary.** Input normalisation and output de-normalisation
  are constant buffers *inside* the graph, so the phone and the C++ engine have
  nothing to reimplement and cannot disagree with training about scaling.

## Results

Median position error after a GNSS blackout. **These are the cross-validated
numbers** — 5 folds over all 12 trusted routes,
each route held out in turn, 4659 blackout start points at 30 s:

| Blackout | Driftless | p90 | Drift | Baseline (no ML) | Oracle floor |
|---|---|---|---|---|---|
| **10 s** | **11.18 m** | 29.3 m | 13.2 % | 28.66 m | 1.5 m |
| **30 s** | **42.09 m** | 118.41 m | 18.3 % | 163.46 m | 6.21 m |
| **60 s** | **88.17 m** | 256.01 m | 18.9 % | 365.7 m | 17.85 m |
| **120 s** | **191.74 m** | 559.22 m | 19.7 % | 727.02 m | 50.45 m |

Per-window across folds: speed MAE **2.242
± 0.349 m/s**, heading MAE
**1.624 ±
0.71°**.

At 30 s the model is **3.9×**
better than dead reckoning without ML, and at 120 s
**3.8×**. The oracle row is
the floor: what perfect speed and heading would still cost, given that position
comes from integration.

Regenerate with `python -m driftless_train.crossval --k 5` (or
`--rebuild-docs` to re-aggregate saved samples); numbers come from
`artifacts/metrics/crossval.json`.

### On the single held-out split

The shipped checkpoint is trained on the frozen 70/15/15 split and evaluated on
its one test route, which is what `evaluate.py` reports:

| Blackout | Driftless | p90 | Drift | Baseline (no ML) | Oracle floor |
|---|---|---|---|---|---|
| **10 s** | **8.6 m** | 21.2 m | 12.0 % | 30.2 m | 1.4 m |
| **30 s** | **31.4 m** | 70.4 m | 15.2 % | 163.5 m | 4.6 m |
| **60 s** | **57.6 m** | 146.4 m | 15.1 % | 354.7 m | 11.9 m |
| **120 s** | **117.6 m** | 331.0 m | 13.9 % | 574.9 m | 29.0 m |

Per-window: speed MAE **1.544 m/s**, heading MAE **1.07°** — against **15.71°**
for raw gyro integration, i.e. **14.7× better**.

These are the numbers for the artefact that actually ships, so they are kept
here — but that route turns out to be the easiest of the eleven, so they are not
the ones to quote. Regenerate with
`python -m driftless_train.evaluate --split test`; source
`artifacts/metrics/eval_summary.json`.

### How the cross-validation was built, and what it changed

The single-split table above holds out one route. To find out whether that road
was representative, `crossval.py` deals the trusted routes into
duration-balanced folds and trains one model per fold through the *same*
`train.fit` the shipped model uses, so the estimate is of this model rather than
a lookalike. 11 of 12 routes yield blackout samples; the twelfth is shorter than
the shortest blackout and contributes training data only.

Fold-to-fold spread at 30 s: **42.244 ± 5.243 m** (range 35.7–49.25 m). Speed
MAE **2.242 ± 0.349 m/s**, Δψ MAE **1.624 ± 0.71°**.

**The honest headline is the cross-validated one.** Pooled 30 s error is
**42.09 m** against **31.39 m** on the single test route — **1.34× worse**.
Cross-validated on its own, **S/S1** is the easiest of the 11 evaluated routes
at 30 s (32.45 m, against a per-route median of 43.94 m), so the single-split
number is the optimistic end of this model's range rather than its centre. Two
effects are mixed in and 12 routes cannot fully separate them: each fold model
trains on ~3/5 of the trusted pool, which pushes its error *up*, while the test
route is genuinely easier, which pushes the single-split figure *down*. Both
numbers are quoted everywhere; the cross-validated one leads.

Per route, each tested out-of-sample by the one fold that held it out:

| route | h | 10 s | 30 s | 60 s | 120 s | n at 30 s |
|---|---|---|---|---|---|---|
| S/S1 | 1.437 | 8.75 m | **32.45 m** | 61.18 m | 113.47 m | 507 |
| S/S3b | 0.189 | 7.88 m | **34.22 m** | 45.77 m | 81.01 m | 55 |
| S/S4 | 2.527 | 10.0 m | **35.14 m** | 68.98 m | 139.92 m | 870 |
| Vw/Vw05 ⚠ | 0.028 | 12.14 m | **40.89 m** | 140.66 m | — | 4 |
| S/S3a | 0.684 | 11.02 m | **41.35 m** | 82.07 m | 173.06 m | 240 |
| S/S2 | 2.608 | 11.32 m | **43.94 m** | 95.65 m | 232.96 m | 911 |
| M/M (Driver B) | 2.944 | 10.98 m | **46.78 m** | 106.0 m | 239.75 m | 954 |
| S/S3c | 1.033 | 13.24 m | **47.66 m** | 95.68 m | 248.12 m | 365 |
| Y/Y1 | 1.859 | 13.78 m | **51.52 m** | 106.05 m | 211.23 m | 644 |
| Vta/Vta24 ⚠ | 0.033 | 18.29 m | **58.41 m** | 97.15 m | — | 5 |
| Vta/Vta02 | 0.305 | 17.38 m | **59.72 m** | 152.68 m | 420.69 m | 104 |

Sorted easiest first. `—` means the route is shorter than the blackout. **⚠
marks fewer than 30 samples at 30 s** — indicative only. Across the 9 routes
with enough samples the 30 s median spans **32.45 m (S/S1) to 59.72 m
(Vta/Vta02)**, a factor of 1.8 — precisely why one held-out road is not enough.
Rebuild the tables from saved samples with `python -m driftless_train.crossval
--rebuild-docs`; full detail in `artifacts/metrics/crossval.md`.

## The speed head reads vibration — and that is a real limitation

Role 04/05 asked whether the edge engine can reuse this model on FOG-grade
inertial input. Answering it turned up something that matters more for our own
model than for the edge tier.

There is **no FOG-grade data in IO-VNBD** to test against, so the shift is
simulated instead: low-pass the held-out phone channels and re-derive the
attitude-invariant features. That removes the high-frequency content a cleaner
sensor would not have, and simultaneously simulates the anti-alias filtering
that decimating a 200 Hz FOG stream to 10 Hz applies. Ratios are paired per
route against that route's own native result, so route difficulty cancels.

| tier | acc HF removed | speed MAE | Δψ MAE | 30 s blackout |
|---|---|---|---|---|
| native phone | — | 1.00× | 1.00× | 1.00× (32.67 m) |
| low-pass 4 Hz | 11 % | 1.108× | 0.987× | **1.024×** (33.58 m) |
| low-pass 2 Hz | 33 % | 2.288× | 1.015× | **2.107×** (69.81 m) |
| low-pass 1 Hz | 45 % | 3.446× | 1.038× | **3.496×** (114.99 m) |
| low-pass 0.5 Hz | 50 % | 4.035× | 1.05× | **4.32×** (142.02 m) |

**Heading is untouched; speed collapses.** Δψ error stays within 1.05× at every
cutoff, while speed MAE degrades up to 4.0×. The reason is that the band being
removed is not noise — it is a speed cue. Across 3 held-out runs (7081 windows)
high-frequency energy correlates with true speed at **+0.640** (range +0.494 to
+0.722 by route, which is road surface):

| speed band | n | HF accel RMS (m/s²) |
|---|---|---|
| 0–2 m/s | 1275 | 0.2845 |
| 2–5 m/s | 662 | 0.5527 |
| 5–10 m/s | 2212 | 0.7794 |
| 10–15 m/s | 1999 | 1.007 |
| 15–20 m/s | 551 | 0.9102 |
| 20–30 m/s | 379 | 0.8929 |

Road, tyre and engine vibration grows with speed, so the model has learned to
use vibration amplitude as a partial speedometer. Note it **saturates above ~10
m/s** — it separates 0 from 10 m/s well and 20 from 30 m/s barely. Yaw rate
needs no such cue, because heading dynamics live below 2 Hz.

Two consequences, and the second is the important one:

1. **The edge tier needs its own model and its own data.** Reusing this one on
   decimated FOG input removes precisely the band its speed head depends on.
2. **A different vehicle or handset is an untested shift for the shipped model.**
   The cross-validation varied route and driver but held vehicle, handset and
   mount fixed, so it is blind to this dependence by construction. The reported
   numbers do not cover a different car, tyres, mount or phone.

### Training the shortcut away — and what that measures

The fix is the same shape as the one that fixed mount sensitivity: train on
low-passed copies of every run (`--lowpass-aug 4 2 1`) so the model cannot lean
on any single frequency band. Measured on the same held-out runs, same routes,
so route difficulty cancels — 30 s blackout error relative to each model's own
native result:

| 30 s blackout, relative to native | native | 4 Hz | 2 Hz | 1 Hz | 0.5 Hz |
|---|---|---|---|---|---|
| shipped (`tcn_best.pt`) | 32.67 m | 1.02× | 2.11× | 3.50× | 4.32× |
| `--lowpass-aug 4 2 1` | 35.23 m | **0.99×** | **0.99×** | **1.01×** | **1.48×** |

**The dependence is gone.** And 0.5 Hz was never trained on — the cutoffs were
4/2/1 — so the model learned not to rely on high-frequency content in general
rather than memorising the three tiers it saw. Heading improves slightly under
low-pass (0.88× at 2 Hz).

The price is **8% on the native phone tier** (32.67 → 35.23 m). That figure is
the useful part: it is the size of the vibration shortcut, i.e. how much of the
shipped model's accuracy comes from a cue specific to this vehicle, mount and
road surface. Read the other way, **the headline numbers above are ~8%
optimistic for any vehicle that is not the IO-VNBD one.**

**The shipped checkpoint is still `tcn_best.pt`**, because the phone path *is*
the native tier and it is better there (30 s on test: 31.39 m vs 33.26 m). The
augmented recipe is the one to fine-tune from for a different vehicle, handset
or a cleaner IMU — which makes the edge tier's data collection plausibly a
fine-tune rather than a from-scratch dataset. Checkpoints are gitignored, so
that is a retrain, not a file to copy.

Caveat unchanged: there is still no FOG data to validate the simulation
against, so this bounds the architecture's response to losing high-frequency
content. It does not certify FOG performance.

`python -m driftless_train.sensor_tier`, and
`--ckpt <augmented> --tag lpaug` for the second row; detail in
`artifacts/metrics/sensor_tier.md` and `sensor_tier_lpaug.md`.

## Evaluation: the question a judge will ask

Not window-level regression error — **position error after a GNSS blackout of a
stated duration**, over many blackout start points on held-out routes, against
three references:

| Reference | What it is |
|---|---|
| **oracle** | integrate the *true* speed and yaw — the dead-reckoning floor |
| **baseline** | no ML: hold last known speed, integrate phone gyro |
| **model** | the TCN's predicted speed and heading change |

Splits are **route-wise, frozen to `configs/splits.json`** on first generation.
Overlapping windows from one drive in both train and test would let the model
memorise a road instead of learning vehicle dynamics.

## Export: two runtimes, one set of weights

| | Size | Parity vs PyTorch | Latency | Input |
|---|---|---|---|---|
| **ONNX** (C++ edge engine) | 218.3 KB (one self-contained file) | 2.3e-06 rel | 0.108 ms/window | `[1, 14, 80]` **NCW** |
| **TFLite** (Android app) | 182.3 KB | 1.5e-06 rel | 0.066 ms/window | `[1, 80, 14]` **NWC** |

Note the layouts differ — TFLite is channels-last (time-major), which is the
natural layout for an Android ring buffer anyway. Parity is asserted numerically
on real recorded windows before either export is accepted.

**The TFLite model is built by rebuilding the network in Keras and porting the
trained weights, not by converting the ONNX.** `onnx2tf` mistranslates this
architecture: bisecting the graph showed the **residual connection** is the
culprit — one block without `x + y` converts, and with it the TFLite graph fails
to prepare (`num_input_elements != num_output_elements (3936 != 3)`, where
3936 = 48×82 is a padded intermediate). More dangerously, a variant that *did*
convert disagreed with PyTorch by 5.4e-2 while reporting success. The Keras route
matches to ~1e-6.

Two further traps on that path, both worth knowing before anyone repeats it:

- **Keras `GroupNormalization` defaults to `epsilon=1e-3`; PyTorch uses `1e-5`.**
  That difference alone put the ported model 5e-3 out. It must be set explicitly.
- **`tf.lite.Optimize.DEFAULT` silently applies int8 dynamic-range weight
  quantisation**, costing 2–8 % output error. float16 was far worse (>100 %
  relative, because the de-normalisation constants span too wide a range). At
  182 KB the model does not need either, so we ship unquantised float32.

## Layout

```
training/driftless_train/
  schema.py      canonical columns for both sides; the unit corrections live here
  geo.py         local-tangent-plane ENU, angle wrapping
  preprocess.py  run splitting, attitude-invariant features, validity flags
  pair.py        S<->V clock alignment and dense target construction
  prepare.py     materialise per-run arrays  -> data/processed/
  dataset.py     causal windowing, route-wise splits, normalisation stats
  model.py       SpeedHeadingTCN + a non-learned baseline to beat
  train.py       training loop
  evaluate.py    blackout metrics + the Round-1 plots
  export.py      ONNX + TFLite with numerical parity gates
  audit.py       dataset inventory -> artifacts/metrics/dataset_audit.md
  paths.py       every path the pipeline reads or writes, in one place
training/tests/  pins every trap above
training/export/to_tflite.py   thin wrapper on the documented entry point
training/data/   raw + prepared data (gitignored; download.py fetches it)
training/artifacts/
  models/        the exported ONNX + TFLite (tracked) and checkpoints (not)
  metrics/       dataset audit, eval CSVs, eval_summary.json
  plots/         the Round-1 figures
  ROUND1_EVIDENCE.md   the paper-round document, generated by report.py
```

## Handover to the other roles

- **Role 01 (app):** load `training/artifacts/models/tcn_speed_heading.tflite`
  (committed — no training run or dataset download needed). Feed a
  `(1, 80, 14)` float32 array — **80 timesteps × 14 channels, time-major** — of
  the channels named in `artifacts/models/stats.json`, in that order, in SI
  units. That is 8 s of history at 10 Hz; keep a ring buffer and infer every
  2 s. Read `[speed_ms, dpsi_rad, dv_ms]`. Do **not** normalise — the graph does
  it. Do **not** enable int8/float16 quantisation (see above).
- **Role 02 (fusion):** you get three measurements per 2 s interval: mean speed,
  heading change, and speed change. Displacement = `speed × 2.0 s`. Use `dv_ms`
  to propagate speed from the last GNSS fix and `speed_ms` as the bounding
  measurement — our fixed-gain blend is a stand-in for what your Kalman gain
  should do properly, and the sweep in §Model shows the optimum is genuinely
  time-varying. `artifacts/metrics/eval_summary.json` has the error
  distributions to set measurement noise from.
- **Roles 04–05 (edge engine):** same weights via
  `training/artifacts/models/tcn_speed_heading.onnx` (committed, with its
  `.onnx.data` sidecar — keep the two together), **NCW** layout `(1, 14, 80)`,
  verified to match PyTorch to 9.7e-7 relative. Latency headroom is large
  (0.118 ms/window on laptop CPU).
- **Role 06 (writeup):** start from `training/artifacts/ROUND1_EVIDENCE.md`,
  which is generated from the artifacts and has every number traceable to a
  file. `artifacts/metrics/dataset_audit.md` for the data section,
  `artifacts/plots/*.png` for figures, `eval_summary.json` for the metrics
  table. The five traps above are the substance of a credible
  "we understood our data" section.

## Known scope limits

- Trained on IO-VNBD, recorded in the UK/France/Nigeria with **one phone** and
  one vehicle. Indian roads and other handsets are the fine-tuning step, exactly
  as the roadmap sequences it (pre-train public → fine-tune own captures).
- Ground truth is the vehicle's CAN + survey GNSS, so labels inherit its
  ~0.2 % dead-reckoning self-consistency floor.
- The IO-VNBD phone data is 10 Hz; our own captures target 100 Hz. The window is
  defined in seconds, not samples, so the model retrains at the higher rate
  without redesign — but it does need retraining.

## Running the exports to where the other builds expect them

`training/export/to_tflite.py` is kept as the entry point its original docstring
promised, and can place the files for the consumers:

```bash
cd training
python export/to_tflite.py --copy-to-consumers
# tcn_speed_heading.tflite -> android/app/src/main/assets/models/
# tcn_speed_heading.onnx    -> edge-engine/models/
```

## Handover artefacts for role 02 (fusion)

`artifacts/metrics/measurement_noise.md` derives the filter's measurement noise
from held-out residuals, in the terms the UKF reasons about:

- Forward-speed measurement: **σ = 2.20 m/s**, with a **+0.49 m/s systematic
  bias** on the held-out route, and a per-speed-band table because the error
  scales with speed.
- Heading change: **σ = 1.45°** per 2 s.
- **The correlation warning.** Consecutive predictions share most of their 8 s
  context, so the residuals are not independent: speed lag-1 autocorrelation
  **0.74**, decorrelation time **8 s** — exactly the context length. Feeding one
  measurement every 2 s as if independent over-informs the filter by about
  **2×** in σ. This is the same trap `ukf_fusion_engine.h` already documents for
  the non-holonomic constraint, where an over-tight σ at high rate collapsed the
  attitude covariance and made the filter reject 21 honest GNSS fixes after a
  blackout.

The measurement is the **longitudinal** body-velocity component — the one axis
the non-holonomic constraint deliberately leaves free — so it fits the existing
`updateUnscented` path with no new filter code. The file contains the snippet.

## IMU noise characterisation

`artifacts/metrics/allan_imu_noise.md` addresses the handset half of the TODO in
`edge-engine/include/driftless/imu_noise.h` (*"fit from Allan deviation once
IO-VNBD / FOG logs are in hand"*), using overlapping Allan deviation over
**4,864 s** of stationary data in 106 spans.

Two of the four parameters turned out **not to be identifiable** from this data,
and the module refuses to report them rather than returning a number: in the
10–60 s window where bias instability should make the curve rise at +½, it is
still falling at ≈ −½ on every axis, so the bias-instability floor is never
reached within the available spans. The gyro white-noise fit is also contaminated
— its Allan deviation is flat-to-rising at short τ, the signature of a periodic
disturbance, because the vehicle is stopped but the engine is idling.

The fix is a capture nobody has taken yet: phone flat on a desk, **engine off**,
10 minutes at `SENSOR_DELAY_FASTEST`. Then all four parameters become
identifiable. That is a ten-minute role 01 task.
