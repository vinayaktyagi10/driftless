# Route-wise cross-validation

**5 folds** over the **12 trusted routes** (13.664 h). Every route is held out exactly once, so the pooled figures below cover the whole trusted pool rather than one held-out road.

Each fold trains a fresh model through the same `train.fit` the shipped model uses (40 epochs, rotate_aug=True, 8.0 s context, 2.0 s output interval). Routes never straddle train and test; val is a disjoint fold.

## Pooled — the headline

| Blackout | n | model median | p90 | drift | baseline | oracle |
|---|---|---|---|---|---|---|
| **10 s** | 4471 | **11.18 m** | 29.3 m | 13.229 % | 28.66 m | 1.5 m |
| **30 s** | 4659 | **42.09 m** | 118.41 m | 18.315 % | 163.46 m | 6.21 m |
| **60 s** | 4665 | **88.17 m** | 256.01 m | 18.855 % | 365.7 m | 17.85 m |
| **120 s** | 4587 | **191.74 m** | 559.22 m | 19.675 % | 727.02 m | 50.45 m |

## Spread across folds

| metric | mean | std | min | max |
|---|---|---|---|---|
| speed MAE (m/s) | **2.242** | 0.349 | 1.619 | 2.674 |
| Δψ MAE (°) | **1.624** | 0.71 | 0.95 | 2.93 |
| 10 s median (m) | **11.232** | 0.824 | 10.44 | 12.8 |
| 30 s median (m) | **42.244** | 5.243 | 35.7 | 49.25 |
| 60 s median (m) | **89.154** | 13.492 | 73.36 | 105.97 |
| 120 s median (m) | **194.704** | 35.256 | 152.13 | 239.75 |

The spread is the point of this table. A single-split number cannot show it, and the fold-to-fold range is what a reader should have in mind when reading any one figure.

## Per fold

| fold | test hours | test routes | speed MAE | Δψ MAE | 30 s med | baseline 30 s |
|---|---|---|---|---|---|---|
| 0 | 3.005 | M/M (Driver B), Vta/Vta24, Vw/Vw05 | 2.2267 | 2.9296° | 46.69 m | 162.09 m |
| 1 | 2.815 | S/S2, S/S3b, Vta/Vta25 | 1.619 | 1.4609° | 42.41 m | 153.03 m |
| 2 | 2.832 | S/S4, Vta/Vta02 | 2.2641 | 1.7242° | 37.17 m | 160.98 m |
| 3 | 2.542 | Y/Y1, S/S3a | 2.4267 | 1.0562° | 49.25 m | 169.56 m |
| 4 | 2.47 | S/S1, S/S3c | 2.6737 | 0.9503° | 35.7 m | 171.25 m |

## Per route — out-of-sample, one fold each

| route | h | fold | 10 s | 30 s | 60 s | 120 s | n at 30 s | baseline 30 s |
|---|---|---|---|---|---|---|---|---|
| S/S1 | 1.437 | 4 | 8.75 m | **32.45 m** | 61.18 m | 113.47 m | 507 | 163.46 m |
| S/S3b | 0.189 | 1 | 7.88 m | **34.22 m** | 45.77 m | 81.01 m | 55 | 164.93 m |
| S/S4 | 2.527 | 2 | 10.0 m | **35.14 m** | 68.98 m | 139.92 m | 870 | 162.19 m |
| Vw/Vw05 ⚠ | 0.028 | 0 | 12.14 m | **40.89 m** | 140.66 m | — | 4 | 206.18 m |
| S/S3a | 0.684 | 3 | 11.02 m | **41.35 m** | 82.07 m | 173.06 m | 240 | 153.74 m |
| S/S2 | 2.608 | 1 | 11.32 m | **43.94 m** | 95.65 m | 232.96 m | 911 | 150.25 m |
| M/M (Driver B) | 2.944 | 0 | 10.98 m | **46.78 m** | 106.0 m | 239.75 m | 954 | 160.98 m |
| S/S3c | 1.033 | 4 | 13.24 m | **47.66 m** | 95.68 m | 248.12 m | 365 | 186.89 m |
| Y/Y1 | 1.859 | 3 | 13.78 m | **51.52 m** | 106.05 m | 211.23 m | 644 | 176.99 m |
| Vta/Vta24 ⚠ | 0.033 | 0 | 18.29 m | **58.41 m** | 97.15 m | — | 5 | 149.99 m |
| Vta/Vta02 | 0.305 | 2 | 17.38 m | **59.72 m** | 152.68 m | 420.69 m | 104 | 148.95 m |

Sorted easiest first. `—` means the route is shorter than the blackout, so no sample exists. **⚠ marks fewer than 30 blackout samples at 30 s** — those medians are indicative only.

Across the 9 routes with enough samples, the 30 s median spans **32.45 m (S/S1) to 59.72 m (Vta/Vta02)** — a factor of 1.8. Road difficulty varies far more than the pooled median suggests, which is exactly why one held-out road is not enough evidence.

Thin rows: Vw/Vw05, Vta/Vta24.

**Not evaluated:** Vta/Vta25. Held out by its fold as intended, but its longest continuous valid span is shorter than the minimum needed to place even one blackout, so it contributes training and validation data only. 11 of 12 trusted routes therefore appear in the pooled figures.

## Why not leave-one-route-out

Route lengths in the trusted pool span 100 s to 2.9 h, a factor of 105. Leave-one-route-out would hand some folds a test set of only 100 s and 118 s — far too few blackout start points for a median to mean anything. One route (Vta/Vta25) is already too short to place even a single blackout. Folds are instead duration-balanced groups of whole routes: sorted longest first and dealt snake-wise, which keeps each fold near 1/k of the total time while never splitting a route.

## Caveats

- Fold models are trained on ~3/5 of the trusted pool, so each sees *less* data than the shipped model. These figures are therefore a slightly pessimistic estimate of the shipped model, not a measurement of it.
- The trusted pool is 12 routes from 4 drivers, one handset, one vehicle. Cross-validation quantifies variation *within* that pool; it says nothing about a different phone, city or car.
- Weakly-coupled routes are excluded throughout, matching the shipped configuration.