# Role 03 decision log — data & model

Non-obvious algorithmic choices in `training/`, each recorded with the named
alternative it was chosen over and the evidence that decided it. Scoped to this
directory so it cannot collide with a repo-level log for the edge engine.

Entries are append-only. If a decision is reversed, add a new entry saying so
rather than editing the old one — the point of the log is the reasoning trail,
including the parts that turned out wrong.

---

## D1 — Bounded-tilt mount augmentation, not uniform SO(3)

**Chosen:** random azimuth plus a tilt drawn uniformly up to `--max-tilt-deg`
(60° default), with a random spin. `augment.tilt_rotation`.

**Over:** uniform-on-SO(3) rotations.

**Why:** uniform SO(3) includes upside-down and edge-on phones, which no
dashboard or vent mount produces. Training against orientations that cannot
occur spends capacity on an impossible input distribution. The bounded family is
both the realistic one and a harder test to pass honestly, because the model
cannot score well by learning "gravity is roughly −Z".

**Evidence:** without augmentation the model degrades 5.5× at a 60 s blackout
under tilt — worse than the no-ML baseline. With it, 1.01×, *and* absolute
accuracy improved (60 s error fell 16 %), so it is the default rather than a
robustness/accuracy trade.

**Kept as a fallback:** `--invariant-only` trains on the 5 gravity-projected
channels, which are *provably* mount-invariant (a fixed rotation commutes with
the linear gravity filter — asserted on real data in `tests/test_augment.py`).
Augmentation is empirical; the subset is a guarantee. We ship the augmented
14-channel model because it is more accurate, and keep the proof as the answer
to "what if the augmentation distribution is wrong".

## D2 — Attitude from a causal one-pole EMA of the accelerometer

**Chosen:** `preprocess._causal_gravity`, a one-pole IIR low-pass via
`scipy.signal.lfilter`.

**Over:** (a) the dataset's own `GRAVITY` columns, and (b) a centred moving
average (`np.convolve(..., mode="same")`), which is what this originally used.

**Why (a):** on IO-VNBD the `GRAVITY` columns are near-constant at
(0, 0, 9.8066) — per-axis std ~0.02, as few as **18 distinct values** in a whole
run. They carry no attitude information, so projecting onto them is projecting
onto a constant.

**Why (b):** a centred filter averages ~10 s of **future** samples into the
current row. That breaks the causality guarantee the windowing depends on and
cannot be reproduced on a phone in real time — the model would have been trained
on information it will never have at inference.

**Pinned by:** `test_gravity_comes_from_a_causal_lowpass_of_the_accelerometer`
(textual, over the whole `imu_derived` chain) and
`test_causal_gravity_uses_no_future_samples` (behavioural: perturbing a future
sample must not change an earlier output).

**Amended (review of #3):** a single non-finite input sample used to replace the
*entire* run's estimate with the crude fallback. Because the filter is
recursive, one bad sample poisons everything after it — but rows before it are
converged and valid, and were being discarded. Now repaired row-wise. Latent on
IO-VNBD (0 of 51 runs contain a non-finite feature) but not on our own captures,
where dropouts are expected and a tilted mount makes the `GRAVITY` fallback
genuinely wrong. See `test_non_finite_sample_only_repairs_the_rows_it_poisons`.

## D3 — Duration-balanced fold and split assignment, not route counts

**Chosen:** routes are allocated by **total duration**, longest first, each going
to whichever split is furthest below its target share. Cross-validation folds
use the same idea via snake dealing (`crossval.build_folds`).

**Over:** every-k-th-route assignment, or equal route counts per fold.

**Why:** route lengths on this dataset span orders of magnitude, so counting
routes does not balance data. An every-5th-route rule handed validation a single
681 s slow city segment (max 12 m/s) while testing on a motorway route (p95
30 m/s) — early stopping then optimised a regime the test set did not contain,
and the speed head was extrapolating far outside anything it had seen.

**Consequence accepted:** folds are balanced in time, not in difficulty. Per-route
30 s error still spans 32.45 → 59.72 m (a factor of 1.8), which is why the
report leads with the pooled cross-validated figure and publishes the per-route
table rather than a single number.

## D4 — Allan fits are fixed-slope but slope-validated

**Chosen:** fit the *level* at the slope physics dictates (−1/2 for white noise,
+1/2 for rate random walk), measure the observed slope independently, and
**refuse the fit** when the two differ by more than `SLOPE_TOLERANCE = 0.25`.

**Over:** free two-parameter (slope + level) fits, or fixed-slope fits with no
validity check.

**Why:** with roughly one usable decade of τ there is not enough leverage to
identify slope and level simultaneously, and the physics already fixes the
slope. But an unchecked fixed-slope fit reports a confident number from a region
of the wrong slope — it happily produced a "bias random walk" from a τ^−1/2
region, which is exactly the mistake this function used to make. Returning
`identifiable: false` with a reason is more useful than a meaningless number.

**Downstream effect:** bias instability comes back unidentifiable from current
data. That is what made dropping the online learned bias corrector the right
call (roles 04/05, PR #5) — there is nothing to train or validate against — and
why an engine-off desk capture is still an open ask.

## D5 — Three output heads, and a complementary filter over them

**Chosen:** predict `[speed_ms, dpsi_rad, dv_ms]` and blend the propagated speed
(`v += dv`) against the absolute `speed_ms` head with a complementary filter
(`--alpha-max`, `--tau`).

**Over:** predicting absolute speed alone.

**Why:** absolute speed is only weakly observable from an accelerometer, while
`dv` is directly observable (the integral of longitudinal acceleration). A real
blackout starts from a known speed — the last GNSS fix — so propagating from it
is far better conditioned than predicting speed from scratch. The absolute head
exists to stop the propagated estimate drifting without bound.

## D6 — Decoupled context length and output interval

**Chosen:** 8 s of context (80 samples), predicting over only the final 2 s
(20 samples).

**Over:** equal context and output windows.

**Why:** the two targets want opposite things. Measured on a 25-epoch sweep at
fixed output granularity: a 2 s window gives speed MAE 2.68 m/s / dpsi 0.86°,
and 4 s gives 2.35 / 1.38. Speed wants long context (vibration and sustained
dynamics); heading change wants a short output interval, both for accuracy and
because role 02's filter consumes it as a per-step increment.

## D7 — Shipping the un-augmented checkpoint despite the vibration shortcut

**Chosen:** `tcn_best.pt` (rotation augmentation only) remains the shipped model.
`--lowpass-aug 4 2 1` is documented as the recipe to fine-tune from for a
different vehicle, handset or a cleaner IMU.

**Over:** promoting the low-pass-augmented checkpoint.

**Why:** the augmented model removes the vibration dependence almost entirely
(2 Hz low-pass costs 0.99× versus 2.11×, and 0.5 Hz — never trained on — costs
1.48× versus 4.32×), but costs 8 % on the **native** phone tier, which is the
tier the Android app actually runs on. On test at 30 s: 31.39 m shipped versus
33.26 m augmented.

**What the 8 % measures:** the size of the vibration shortcut, i.e. the portion
of the shipped model's accuracy that comes from a cue specific to this vehicle,
mount and road surface. Read the other way, the headline numbers are ~8 %
optimistic for any vehicle that is not the IO-VNBD one. That is now stated in
the limitations of both the README and the evidence doc.

**Revisit when:** our own multi-vehicle captures exist. At that point the
native tier is no longer a single vehicle and this trade should be re-measured
rather than re-argued.

## D8 — `n_ch` follows the checkpoint, and parity inputs must follow it too

**Chosen:** `sample_inputs` takes the checkpoint's channel subset and applies it
to the windows it draws from disk, with a hard shape assertion.

**Why:** recorded because it was a review finding, not a design choice. The
`.npz` files always hold all 14 channels; an `--invariant-only` checkpoint
expects 5. The export path reduced `n_ch` correctly from `ck["channels"]` but
fed 14-channel windows to the parity check, so exporting that variant crashed
instead of exporting. Anything that manufactures model input must derive its
channel set from the checkpoint, never from the on-disk layout.
