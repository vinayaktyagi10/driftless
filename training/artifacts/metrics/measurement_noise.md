# Measurement noise for the fusion filter

From residuals pooled over the held-out **test** and **val** splits (7081 predictions, 2.0 s apart, 8.0 s context).

## Forward-speed measurement

> **Do not hard-code the bias.** These values are pooled over every held-out route. Bias is not a stable property of this model: it is a property of each route's speed distribution interacting with a shrinkage estimator, and it changes sign between held-out routes (see the per-split table below). Subtracting one route's bias measurably hurts on another. Use **0.0** unless you have calibrated on the actual deployment route.

- sigma **2.893 m/s**, bias -0.1767 m/s, MAE 1.8721 m/s, p95 |error| 6.2476 m/s

Error scales with speed, so a single sigma is wrong across the range:

### Per held-out split — why one route is not enough

| split | routes | n | speed sigma (m/s) | speed bias (m/s) |
|---|---|---|---|---|
| test | S/S1 | 2569 | **2.2046** | +0.4853 |
| val | S/S4 | 4512 | **3.0745** | -0.5536 |

Sigma differs by ~40 % between held-out routes and the bias changes sign. Use the pooled sigma above; an over-tight sigma is the failure mode that collapses the filter's covariance and makes it reject honest GNSS fixes.

> **This table is NOT a calibration curve.** It bins by *true* speed, which the filter cannot observe, and a minimum-MSE estimator necessarily looks biased when conditioned on the truth -- it shrinks toward the mean, measured here at std(pred)/std(true) = 0.84 with a pred-on-true slope of 0.74. Correcting that de-shrinks the estimate and *raises* MSE. Cross-fitting a linear correction between the two held-out routes made speed MAE worse in both directions (1.54 -> 1.69 and 2.06 -> 2.25 m/s). This table shows where the error lives; it is not meant to be inverted.

| speed band | n | sigma (m/s) | bias (m/s) | sigma / mean speed |
|---|---|---|---|---|
| 0-5 m/s | 1928 | 1.5529 | 0.6336 | 102.6 % |
| 5-10 m/s | 2225 | 1.8577 | 0.7773 | 24.3 % |
| 10-15 m/s | 2000 | 2.3832 | -0.1312 | 19.5 % |
| 15-25 m/s | 803 | 3.8771 | -3.6253 | 20.6 % |
| 25-40 m/s | 125 | 4.5265 | -8.2272 | 16.8 % |

## Speed-change (`dv`) measurement

- sigma **0.7968 m/s**, bias -0.022 m/s, MAE 0.5461 m/s over 2.0 s
- decorrelation time 2.0 s (lag-1 0.3006)

Definition: `v[end-1] - v[start] within the output interval, matching dataset.window_targets`. This is the better-conditioned of the two speed outputs and is what a blackout should propagate from, since it starts from a known speed.

## Heading-change measurement

- sigma **1.415°** (0.024696 rad) per 2.0 s, bias 0.0362°

## The correlation warning

Consecutive predictions share most of their input context, so their errors are **not independent**:

- speed residual lag-1 autocorrelation **0.7662**, decorrelation time **8.0 s**
- heading residual lag-1 autocorrelation **0.1798**, decorrelation time **2.0 s**

Feeding every prediction as an independent measurement at 2.0 s spacing therefore over-informs the filter. If you do that, inflate sigma by about **2.0x** for speed (sqrt(tau / update interval)). The alternative -- and the better fix -- is to apply the update only once per decorrelation time, or to model the correlation.

This is the same failure mode already documented for the non-holonomic constraint in `ukf_fusion_engine.h`: an over-tight sigma applied at high rate collapsed the attitude covariance and made the filter reject honest GNSS fixes after the blackout. Same trap, different measurement.

## Suggested wiring

The measurement is the longitudinal component of body velocity -- the one axis the non-holonomic constraint deliberately leaves free. It fits the existing unscented path with no new filter code:

```cpp
// h(x) = forward component of body velocity
auto h = [](const NavState& s) {
    Eigen::VectorXd z(1);
    z(0) = UkfFusionEngine::bodyVelocity(s).x();
    return z;
};
Eigen::MatrixXd sqrt_R(1, 1);
sqrt_R(0, 0) = 2.893 * 2.0;   // pooled held-out sigma x correlation inflation
engine.updateUnscented(h, z_from_model, sqrt_R, 0.99);
```

Speed-dependent sigma from the table above is better than the single value, and the NIS gate at 0.99 will reject the occasional bad prediction rather than letting it into the state.

## Per-run breakdown

| run | n | speed sigma | speed bias | dpsi sigma | dpsi bias |
|---|---|---|---|---|---|
| S-S1#0 | 2569 | 2.2046 | 0.4853 | 1.4537° | 0.1965° |
| S-S4#0 | 1741 | 2.0391 | 0.3596 | 1.3474° | 0.0529° |
| S-S4#1 | 2771 | 3.5748 | -1.1273 | 1.4025° | -0.1228° |