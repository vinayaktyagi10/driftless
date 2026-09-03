# IMU noise characterisation from IO-VNBD (handset)

Measured over **4864 s** of stationary data in **106 spans** across 28 runs, at 10.0 Hz. Overlapping Allan deviation.

Addresses the handset half of the TODO in `edge-engine/include/driftless/imu_noise.h`. **Read the caveats before using any of these** -- two of the four parameters turned out not to be identifiable from this data at all, and reporting a number for them would have been worse than reporting nothing.

## What this data can and cannot pin down

| parameter | fitted value | unit |
|---|---|---|
| `accel_noise_density` | 0.04728 | m/s^2/sqrt(Hz) |
| `gyro_noise_density` | 0.009238 | rad/s/sqrt(Hz) |
| `accel_bias_random_walk` | **not identifiable** | m/s^3/sqrt(Hz) |
| `gyro_bias_random_walk` | **not identifiable** | rad/s^2/sqrt(Hz) |

### Why parameters are rejected rather than reported

A fixed-slope fit returns a number whatever the data looks like, so the observed log-log slope is measured independently and the fit is refused when the two disagree (tolerance 0.25). Observed slopes:

| sensor | axis | white-noise region (expect -0.50) | bias-instability region (expect +0.50) |
|---|---|---|---|
| accel | x | -0.486 | -1.417 |
| accel | y | -0.483 | -1.449 |
| accel | z | -0.591 | -0.634 |
| gyro | x | -0.555 | -0.415 |
| gyro | y | -0.421 | -0.414 |
| gyro | z | -0.291 | -0.367 |

**Bias random walk is not observable here.** In the 10-60 s window where bias instability should make the curve RISE at +1/2, it is still falling at about -0.5 for every axis. The bias-instability floor simply has not been reached within the longest stationary spans available (minimum span 20 s). Estimating it needs stationary records of ~10 minutes or more.

**The gyro white-noise fit is contaminated.** Its Allan deviation is flat-to-rising at short tau rather than falling at -1/2, which is the signature of a correlated/periodic disturbance, not white noise. The vehicle is stopped but the engine is running, so the handset is sitting in idle vibration.

## Caveats that must travel with these numbers

- **The engine is running.** 'Stationary' here means the vehicle is not moving; it does not mean the sensor is at rest. Idle vibration inflates every figure, so treat these as UPPER BOUNDS on sensor noise -- though arguably closer to what the filter meets in service than a datasheet value is.
- **10 Hz logs.** The white-noise region is only observable for tau >= 0.2 s, and higher-frequency vibration energy aliases down into that band, inflating it further.
- **Handset only.** Nothing here speaks to the FOG unit; `fogGrade()` still needs its own logs.
- One handset model, one vehicle.

## The 10-minute fix

Everything above is limited by the data, not the method. A proper characterisation needs one capture that nobody has taken yet:

1. Phone flat on a desk, **engine off**, no one touching it.
2. Log accelerometer and gyroscope at the highest rate the handset offers (`SENSOR_DELAY_FASTEST`, typically 100-500 Hz).
3. **10 minutes minimum** -- long enough to reach the bias-instability floor.

Then rerun `python -m driftless_train.allan` against that log and all four parameters become identifiable. This is a role 01 task with the capture app and takes longer to read about than to do.