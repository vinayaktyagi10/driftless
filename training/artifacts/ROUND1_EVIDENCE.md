# Driftless — Round 1 evidence

SIH 2026 · PS #26168 (ISRO) · role 03 (data + model training)
Generated 2026-09-01 from the artifacts in this repo.

## 1. What was built

A speed & heading-change regressor trained on the public **IO-VNBD** dataset, evaluated by **dead reckoning through simulated GNSS blackouts** on held-out routes, and exported for both target runtimes.

## 2. Data

- **51 runs**, **27.96 h**, **1268.8 km** of paired phone-IMU + vehicle-ground-truth driving.
- Route families: M, S, Vf, Vta, Vtb, Vw, Y.
- Feature channels (14): `acc_x`, `acc_y`, `acc_z`, `gyro_yaw`, `gyro_pitch`, `gyro_roll`, `grav_x`, `grav_y`, `grav_z`, `acc_norm`, `acc_vert`, `acc_horiz`, `gyro_vert`, `gyro_horiz`.
- Ground truth is the paired vehicle file (survey GNSS at 7 decimal places, CAN velocity, wheel speeds, yaw rate) at a genuine 10 Hz. The phone's own GNSS updates only every 9 s and is unusable as a per-window label.

Sanity check on the ground truth itself: integrating the **true** speed and heading reproduces the **true** trajectory to **0.212 % drift over 38.07 km**. That validates units, the compass convention and target coherence together, and sets the floor any model is measured against.

Full per-run inventory: `artifacts/metrics/dataset_audit.md`.

## 3. Split discipline

Splits are **route-wise and duration-balanced**, frozen to `configs/splits.json`. Windows from one drive never straddle train and test — otherwise the model can memorise a road instead of learning vehicle dynamics.

Routes are additionally separated by **phone/vehicle coupling**. Only rigidly-coupled routes may appear in val or test; weakly-coupled ones are train-only and excluded from the reported model (see the coupling section of the README). The hours below are trusted data; the 70/15/15 target applies to it.

| Split | Trusted routes | Trusted hours | Route IDs |
|---|---|---|---|
| train | 10 | 9.70 | M/M (Driver B), S/S2, S/S3a, S/S3b, S/S3c, Vta/Vta02, Vta/Vta24, Vta/Vta25, Vw/Vw05, Y/Y1 |
| val | 1 | 2.53 | S/S4 |
| test | 1 | 1.44 | S/S1 |

Plus 35 weakly-coupled train-only routes (14.29 h), not used by the reported model.

## 4. Results — position error after a GNSS blackout

Held-out **test** split, 1 run(s). Context 8.0 s, output interval 2.0 s. Blackout start points every 10 s across each run; each row aggregates all of them.

- **model** — the regressor, with speed propagated from the last known fix and blended toward the absolute head (ramp τ = 20.0 s)
- **abs-only** — the absolute speed head alone, no propagation
- **baseline** — no ML: hold the last known speed, integrate the phone gyro for heading
- **oracle** — integrate the *true* speed and yaw: the dead-reckoning floor

| Blackout | n | model median (m) | model p90 (m) | model drift | abs-only (m) | baseline (m) | baseline drift | oracle (m) |
|---|---|---|---|---|---|---|---|---|
| 10 s | 489 | **8.6** | 21.2 | 12.0 % | 10.6 | 30.1 | 51.0 % | 1.4 |
| 30 s | 507 | **31.4** | 70.4 | 15.2 % | 31.8 | 163.5 | 87.9 % | 4.6 |
| 60 s | 508 | **57.5** | 146.4 | 15.1 % | 56.7 | 354.7 | 88.3 % | 11.9 |
| 120 s | 502 | **117.6** | 331.0 | 13.9 % | 115.7 | 574.9 | 75.5 % | 29.0 |

### Per-window regression accuracy

- Speed MAE **1.544 m/s**
- Heading-change MAE **1.070°** per 2.0 s window
- Same heading by raw gyro integration alone: **15.708°** — the learned head is 14.7× better

Raw per-blackout records: `artifacts/metrics/eval_blackouts.csv`.

## 5. Figures

### `plots/blackout_error.png`

![blackout_error](plots/blackout_error.png)

*Position error and drift against blackout duration, model vs baseline vs oracle.*

### `plots/blackouts30_S-S1_0.png`

![blackouts30_S-S1_0](plots/blackouts30_S-S1_0.png)

### `plots/blackouts60_S-S1_0.png`

![blackouts60_S-S1_0](plots/blackouts60_S-S1_0.png)

### `plots/speed_S-S1_0.png`

![speed_S-S1_0](plots/speed_S-S1_0.png)

*Regressor output vs vehicle ground truth on S-S1#0.*

### `plots/traj_S-S1_0.png`

![traj_S-S1_0](plots/traj_S-S1_0.png)

*Dead reckoning on S-S1#0 with GNSS off for the entire run — no position updates after the start point.*

## 6. Deployment artefacts

- **38,499 parameters**, input `[1, 14, 80]`, outputs `speed_ms, dpsi_rad, dv_ms` in SI units.
- **ONNX** (C++ edge engine, roles 04–05): 123.4 KB, matches PyTorch to **2.27e-06** relative on real windows, **0.1183 ms/window** on CPU.
- **TFLite** (Android app, role 01): 182.3 KB, matches PyTorch to **1.51e-06** relative.

Normalisation is baked into both graphs, so the phone and the C++ engine cannot disagree with training about scaling.

## 7. Honest limitations

- **The regressor alone does not reach the <10 m at 30 s target.** It reaches roughly that at a 10 s blackout; at 30 s the residual is dominated by absolute-speed error, which is the fundamentally hard part of inertial-only odometry. Closing the rest is what the road-network constraint (map matching) and the EKF in role 02 are for — a vehicle on a known road cannot be anywhere the map does not allow.
- Trained on IO-VNBD: UK/France/Nigeria, one phone, one vehicle. Indian roads and other handsets are the fine-tuning step the roadmap already sequences (pre-train public → fine-tune own captures).
- IO-VNBD phone data is 10 Hz; our own captures target 100 Hz. The window is defined in seconds, so the design carries over — but it needs retraining, not just reuse.
- Ground truth is the vehicle's own CAN + survey GNSS, so labels inherit its ~0.2 % self-consistency floor.

## 8. Reproducing these numbers

```bash
cd training
export PYTHONPATH=.
python -m driftless_train.download    # LFS-aware, both S and V sides
python -m driftless_train.audit       # per-run inventory
python -m driftless_train.prepare     # pair S with V, materialise arrays
python -m driftless_train.train --epochs 40
python -m driftless_train.evaluate --split val --sweep-alpha  # tune blend
python -m driftless_train.evaluate --split test               # numbers
python -m driftless_train.export
python -m driftless_train.report
pytest tests/ -q                      # pins every dataset trap
```
