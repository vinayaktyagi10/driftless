# Sensor-tier transfer: can the phone model serve a cleaner IMU?

Measured on 3 trusted held-out run(s) (S/S1, S/S4) with the shipped checkpoint, unmodified. A cleaner sensor is simulated by low-passing the raw accelerometer and gyroscope axes and re-deriving the attitude-invariant channels — which also simulates the anti-alias filtering that decimating a 200 Hz FOG stream to 10 Hz would apply.

Ratios are **paired per route** against that route's own native result, so route difficulty cancels; the cross-validation showed per-route 30 s medians spanning a factor of 1.8, so unpaired numbers would be noise.

| tier | acc HF removed | gyro HF removed | speed MAE | Δψ MAE | 30 s | 60 s |
|---|---|---|---|---|---|---|
| **native phone** | — | — | 1.00× | 1.00× | 1.00× (32.67 m) | 1.00× |
| low-pass 4 Hz | 10.9 % | 8.7 % | 1.108× | 0.987× | **1.024×** (33.58 m) | 1.04× |
| low-pass 2 Hz | 33.2 % | 32.0 % | 2.288× | 1.015× | **2.107×** (69.81 m) | 2.204× |
| low-pass 1 Hz | 45.0 % | 43.2 % | 3.446× | 1.038× | **3.496×** (114.99 m) | 3.714× |
| low-pass 0.5 Hz | 50.0 % | 48.2 % | 4.035× | 1.05× | **4.32×** (142.02 m) | 4.495× |

## Why the speed head breaks and the heading head does not

The band the low-pass removes is not noise — it is a speed cue. Across 3 held-out runs (7081 windows), high-frequency energy above 2 Hz correlates with true speed at **+0.640** (accelerometer, range +0.494 to +0.722 by route) and **+0.566** (gyroscope):

| speed band | n | HF accel RMS (m/s²) | HF gyro RMS (rad/s) |
|---|---|---|---|
| 0–2 m/s | 1275 | 0.2845 | 0.02849 |
| 2–5 m/s | 662 | 0.5527 | 0.05727 |
| 5–10 m/s | 2212 | 0.7794 | 0.08389 |
| 10–15 m/s | 1999 | 1.007 | 0.10582 |
| 15–20 m/s | 551 | 0.9102 | 0.10746 |
| 20–30 m/s | 379 | 0.8929 | 0.11225 |

Road, tyre and engine vibration grows with speed, so the model reads vibration amplitude as a partial speedometer. Note the cue **saturates above ~10 m/s** — it separates 0 from 10 m/s well and 30 from 20 m/s barely — and its strength varies by route, which is road surface. It is a real cue, and it is specific to this sensor, mount, vehicle and surface. Yaw rate needs no such cue: heading dynamics live below 2 Hz, which is why Δψ survives every cutoff tested.

**This is a limitation of the shipped model, not only an argument about the edge tier.** The cross-validation varied route and driver but held vehicle, handset and mount fixed, so it cannot see this dependence. A different car or phone is a shift the reported numbers do not cover.

## What this does and does not establish

- It bounds how this architecture responds to losing high-frequency content, which is the dominant difference between a phone MEMS stream and a decimated FOG stream.
- It does **not** certify FOG performance. There is no FOG-grade data in IO-VNBD to validate the simulation against.
- The vehicle CAN channels are not a substitute: 3 inertial signals at 10 Hz, quantised to 0.1 °/s and 0.0092 g, which measurement puts below one LSB of their own noise — so their noise density cannot be measured from this dataset at all.
- `filtfilt` is zero-phase on purpose: a real cleaner sensor has no group delay, so a causal filter would penalise the model for a timing shift instead of the noise change under test. It is an offline test input generator, not an online filter.