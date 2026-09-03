"""Windowing, route-wise splitting and normalisation.

Two rules this module enforces, because breaking either produces a number that
looks good and means nothing:

* Windows are **causal**: a window ending at sample i contains only samples
  <= i. The phone cannot see the future, so neither can training.
* Splits are **by route**, never by window. Overlapping windows from one drive
  landing in both train and test would let the model memorise a road instead of
  learning vehicle dynamics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import PROC_DIR, SPLITS_PATH

# Context length and output interval are DECOUPLED, because the two targets want
# different things. Measured on a 25-epoch sweep at fixed output granularity:
#
#   window   speed MAE   dpsi MAE
#   2.0 s    2.68 m/s    0.86 deg
#   4.0 s    2.35 m/s    1.38 deg
#
# Absolute speed is barely observable from 2 s of accelerometer -- it needs long
# context (road/tyre vibration and sustained dynamics). Heading change wants a
# SHORT output interval, both for accuracy and because role 02's EKF consumes it
# as a per-step increment. So we feed a long context window and predict over only
# its final OUT_WIN samples.
WIN = 80          # 8.0 s of context at 10 Hz
OUT_WIN = 20      # predict over the last 2.0 s -- the EKF's update interval
STRIDE_TRAIN = 5  # 0.5 s hop
DT = 0.1


@dataclass
class Run:
    run_id: str
    route: str
    family: str
    path: Path

    def load(self) -> dict[str, np.ndarray]:
        with np.load(self.path) as z:
            return {k: z[k] for k in z.files}


def load_index(proc_dir: Path = PROC_DIR) -> tuple[list[Run], dict]:
    idx = json.loads((proc_dir / "index.json").read_text())
    runs = [Run(r["run_id"], r["route"], r["family"], proc_dir / f"{r['run_id']}.npz")
            for r in idx["runs"] if (proc_dir / f"{r['run_id']}.npz").exists()]
    return runs, idx


def make_splits(runs: list[Run], out: Path = SPLITS_PATH,
                targets: tuple[float, float, float] = (0.70, 0.15, 0.15),
                proc_dir: Path = PROC_DIR) -> dict[str, list[str]]:
    """Deterministic, duration-balanced, route-wise split.

    Routes stay intact -- windows from one drive must never straddle train and
    test -- but they are allocated by DURATION, not by count. Counting routes
    fails badly on this dataset: an every-5th-route rule handed validation a
    single 681 s slow city segment (max 12 m/s) while testing on a motorway route
    (p95 30 m/s), so early stopping optimised the wrong regime and the speed head
    was extrapolating far outside anything it had seen.

    Longest routes are placed first, each going to whichever split is currently
    furthest below its target share. Stratified within route family so every
    family is represented in train.

    Written to disk on first use and reused verbatim afterwards, so the split can
    never drift between the numbers in the report and the numbers in the repo.
    """
    if out.exists():
        return json.loads(out.read_text())["splits"]

    from .pair import TRUSTED_COUPLING_CORR

    idx_path = proc_dir / "index.json"
    dur: dict[str, float] = {}
    fam: dict[str, str] = {}
    coupling: dict[str, float] = {}
    if idx_path.exists():
        for r in json.loads(idx_path.read_text())["runs"]:
            dur[r["route"]] = dur.get(r["route"], 0.0) + r["duration_s"]
            fam[r["route"]] = r["family"]
            coupling[r["route"]] = max(coupling.get(r["route"], 0.0),
                                       abs(r.get("align_corr", 0.0)))

    # Routes where the phone is not rigidly coupled to the vehicle are TRAIN-ONLY.
    # Evaluating on them would measure how well we predict a sliding phone, not a
    # mounted one, and our product assumes a mount.
    weak = {r for r, c in coupling.items() if c < TRUSTED_COUPLING_CORR}
    for r in runs:
        dur.setdefault(r.route, 1.0)
        fam.setdefault(r.route, r.family)

    names = ("train", "val", "test")
    splits: dict[str, list[str]] = {k: [] for k in names}
    got = {k: 0.0 for k in names}

    # Weakly-coupled routes go to train and are EXCLUDED from the duration
    # budget. Counting them was a real bug: they filled train's 70% share, which
    # pushed the scarce trusted routes into val/test and left only 5.5 h of the
    # 13.7 h of trusted data available for training. The 70/15/15 target applies
    # to trusted data; weak routes are extra on top, for whoever wants them.
    by_family: dict[str, list[str]] = {}
    for route in {r.route for r in runs}:
        if route in weak:
            splits["train"].append(route)
            continue
        by_family.setdefault(fam.get(route, "?"), []).append(route)

    for family in sorted(by_family):
        routes = sorted(by_family[family], key=lambda r: (-dur[r], r))
        if len(routes) < 3:
            # Too few routes to hold any out without losing the family entirely.
            for route in routes:
                splits["train"].append(route)
                got["train"] += dur[route]
            continue
        for route in routes:
            total = sum(got.values()) + dur[route]
            deficit = {k: targets[i] - (got[k] / total if total else 0.0)
                       for i, k in enumerate(names)}
            pick = max(names, key=lambda k: deficit[k])
            splits[pick].append(route)
            got[pick] += dur[route]

    total = sum(got.values()) or 1.0
    weak_h = sum(dur.get(r, 0.0) for r in weak) / 3600
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "policy": "route-wise, duration-balanced, stratified within route "
                  "family; longest route first to whichever split is furthest "
                  "below its target share. Families with <3 routes stay in "
                  "train. Routes whose phone/vehicle coupling is below "
                  f"{TRUSTED_COUPLING_CORR} are TRAIN-ONLY -- never evaluated on. "
                  "Frozen on first generation.",
        "weak_coupling_train_only": sorted(weak),
        "targets": dict(zip(names, targets, strict=True)),
        "achieved_duration_frac_of_trusted": {k: round(got[k] / total, 4)
                                              for k in names},
        "trusted_duration_s": {k: round(got[k], 1) for k in names},
        "weak_train_only_duration_h": round(weak_h, 2),
        "splits": {k: sorted(v) for k, v in splits.items()},
    }, indent=2))
    return {k: sorted(v) for k, v in splits.items()}


# Target vector layout, shared by the dataset, the model heads and the exports.
TARGETS: tuple[str, ...] = ("speed_ms", "dpsi_rad", "dv_ms")


class WindowDataset(Dataset):
    """Causal IMU context -> (mean speed, heading change, speed change).

    Three targets, because dead reckoning needs speed two different ways:

      speed_ms : mean speed over the output interval. Absolute, but only weakly
                 observable from an accelerometer -- it leans on vibration and
                 sustained-dynamics cues.
      dpsi_rad : heading change across the interval.
      dv_ms    : CHANGE in speed across the interval. Directly observable, since
                 it is the integral of longitudinal acceleration.

    The third one matters because a real blackout starts from a known speed --
    the last GNSS fix -- so propagating `v += dv` is far better conditioned than
    predicting absolute speed from scratch, while the absolute head remains
    available to stop the propagated estimate drifting without bound. That is
    exactly the pair of measurements role 02's EKF wants.
    """

    def __init__(self, runs: list[Run], win: int = WIN, stride: int = STRIDE_TRAIN,
                 out_win: int = OUT_WIN, moving_only: bool = False,
                 rotate_aug: bool = False, max_tilt_deg: float = 60.0,
                 channels: np.ndarray | None = None, seed: int = 0,
                 lowpass_aug: tuple[float, ...] = ()):
        self.win, self.stride, self.out_win = win, stride, out_win
        # Mount-rotation augmentation. Every IO-VNBD phone lay flat, so without
        # this the raw body-frame channels have never seen a tilted handset --
        # which is what a dashboard mount is. See augment.py for why this is
        # exact rather than approximate.
        self.rotate_aug = rotate_aug
        self.max_tilt_deg = max_tilt_deg
        # Sensor-tier augmentation. `sensor_tier.py` showed the speed head leans
        # on high-frequency vibration, which is specific to this phone, mount,
        # vehicle and road surface: low-passing held-out input at 2 Hz costs
        # 2.11x at a 30 s blackout while heading is untouched. Training on a mix
        # of native and low-passed copies removes the option of relying on any
        # single band. Filtering is done PER RUN, once, up front -- the same way
        # sensor_tier filters at evaluation time, so train and eval see the same
        # transform. Filtering an 80-sample window instead would restart the
        # causal gravity estimator inside every window and not match eval.
        self.lowpass_aug = tuple(lowpass_aug)
        self.channels = None if channels is None else np.asarray(channels)
        # One INDEPENDENT stream per augmentation, spawned from the same
        # seed. A single shared generator made the draws order-dependent:
        # enabling --lowpass-aug consumed integers ahead of the rotation draw,
        # so the same seed produced different rotations depending on which other
        # flags were set, and no two flag combinations were comparable.
        self._rng_lowpass, self._rng_rotate = (
            np.random.default_rng(seed).spawn(2))
        if out_win > win:
            raise ValueError("out_win cannot exceed the context length")
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.items: list[tuple[str, int]] = []      # (run_id, context end, exclusive)

        self.tiers: dict[str, list[np.ndarray]] = {}
        for run in runs:
            a = run.load()
            self.arrays[run.run_id] = a
            if self.lowpass_aug:
                from .sensor_tier import simulate_tier
                self.tiers[run.run_id] = [
                    simulate_tier(a["features"], DT, hz).astype(np.float32)
                    for hz in self.lowpass_aug]
            valid = a["valid"]
            n = len(valid)
            # The context [end-win, end) must be entirely valid; the output
            # interval is its final out_win samples.
            csum = np.r_[0, np.cumsum(valid.astype(np.int64))]
            for end in range(win, n + 1, stride):
                if csum[end] - csum[end - win] != win:
                    continue
                if moving_only and not a["moving"][end - out_win:end].any():
                    continue
                self.items.append((run.run_id, end))

    def __len__(self) -> int:
        return len(self.items)

    def window_targets(self, a: dict[str, np.ndarray],
                       end: int) -> tuple[float, float, float]:
        """Mean speed and heading change over the OUTPUT interval.

        Heading change is the integral of the vehicle's yaw rate, NOT the
        difference of GNSS heading. Measured on route S1: GNSS heading is
        meaningless below ~2 m/s -- across stationary windows its 2 s difference
        has std 39 deg and reaches 197 deg, because course-over-ground spins
        freely at standstill. The yaw-rate integral is bounded everywhere and
        agrees with GNSS heading to 1.07 deg (corr 0.989) on moving windows,
        where GNSS heading IS trustworthy. So we take the signal that is right
        everywhere and validate it against the one that is right sometimes.

        Both quantities span out_win*dt, so tiling non-overlapping output
        intervals reproduces a Riemann sum over every sample -- which is exactly
        what the dead-reckoning evaluation does.
        """
        lo = end - self.out_win
        v = a["speed_ms"]
        speed = float(v[lo:end].mean())
        dpsi = float(a["yaw_rate_rads"][lo:end].sum() * DT)
        dv = float(v[end - 1] - v[lo])
        return speed, dpsi, dv

    def __getitem__(self, i: int):
        run_id, end = self.items[i]
        a = self.arrays[run_id]
        feats = a["features"]
        if self.lowpass_aug:
            # Uniform over {native} U cutoffs, so the native tier keeps a real
            # share of the batch rather than being crowded out.
            k = int(self._rng_lowpass.integers(0, len(self.lowpass_aug) + 1))
            if k > 0:
                feats = self.tiers[run_id][k - 1]
        x = feats[end - self.win:end].T   # (C, W) causal context
        if self.rotate_aug:
            from .augment import rotate_window, tilt_rotation
            x = rotate_window(x.astype(np.float64),
                              tilt_rotation(self._rng_rotate,
                                            self.max_tilt_deg))
        if self.channels is not None:
            x = x[self.channels]
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.tensor(self.window_targets(a, end), dtype=torch.float32))


def compute_stats(ds: WindowDataset, max_windows: int = 20000) -> dict:
    """Per-channel input stats and target stats, from the TRAIN split only."""
    idx = np.linspace(0, len(ds) - 1, min(max_windows, len(ds))).astype(int)
    xs, ys = [], []
    for i in idx:
        x, y = ds[int(i)]
        xs.append(x.numpy())
        ys.append(y.numpy())
    X = np.stack(xs)                    # (N, C, W)
    Y = np.stack(ys)                    # (N, n_targets)
    return {
        "x_mean": X.mean(axis=(0, 2)).tolist(),
        "x_std": (X.std(axis=(0, 2)) + 1e-6).tolist(),
        "y_mean": Y.mean(axis=0).tolist(),
        "y_std": (Y.std(axis=0) + 1e-6).tolist(),
        "n_windows_sampled": int(len(idx)),
    }


def split_runs(runs: list[Run], splits: dict[str, list[Run]],
               proc_dir: Path = PROC_DIR,
               min_coupling: float = 0.0) -> dict[str, list[Run]]:
    """Materialise the split. `min_coupling` drops weakly-coupled TRAIN routes."""
    want = {k: set(v) for k, v in splits.items()}
    out = {k: [r for r in runs if r.route in want[k]]
           for k in ("train", "val", "test")}

    if min_coupling > 0.0:
        idx_path = proc_dir / "index.json"
        if idx_path.exists():
            corr = {r["run_id"]: abs(r.get("align_corr", 0.0))
                    for r in json.loads(idx_path.read_text())["runs"]}
            out["train"] = [r for r in out["train"]
                            if corr.get(r.run_id, 0.0) >= min_coupling]
    return out
