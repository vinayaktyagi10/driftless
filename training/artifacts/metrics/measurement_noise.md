# Measurement noise for the fusion filter

From residuals on the held-out **test** split (2569 predictions, 2.0 s apart, 8.0 s context).

## Forward-speed measurement

- sigma **2.2046 m/s**, bias 0.4853 m/s, MAE 1.5444 m/s, p95 |error| 4.5311 m/s

Error scales with speed, so a single sigma is wrong across the range:

| speed band | n | sigma (m/s) | bias (m/s) | sigma / mean speed |
|---|---|---|---|---|
| 0-5 m/s | 771 | 2.0749 | 0.8847 | 108.2 % |
| 5-10 m/s | 1020 | 1.7048 | 0.8516 | 22.6 % |
| 10-15 m/s | 700 | 2.5277 | -0.154 | 20.7 % |
| 15-25 m/s | 78 | 2.4831 | -2.5161 | 15.2 % |

## Heading-change measurement

- sigma **1.4537°** (0.025373 rad) per 2.0 s, bias 0.1965°

## The correlation warning

Consecutive predictions share most of their input context, so their errors are **not independent**:

- speed residual lag-1 autocorrelation **0.7367**, decorrelation time **8.0 s**
- heading residual lag-1 autocorrelation **0.1903**, decorrelation time **2.0 s**

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
sqrt_R(0, 0) = 2.2046 * 2.0;   // sigma x correlation inflation
engine.updateUnscented(h, z_from_model, sqrt_R, 0.99);
```

Speed-dependent sigma from the table above is better than the single value, and the NIS gate at 0.99 will reject the occasional bad prediction rather than letting it into the state.

## Per-run breakdown

| run | n | speed sigma | speed bias | dpsi sigma | dpsi bias |
|---|---|---|---|---|---|
| S-S1#0 | 2569 | 2.2046 | 0.4853 | 1.4537° | 0.1965° |