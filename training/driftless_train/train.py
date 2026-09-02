"""Train the speed & heading-change regressor.

Run:  python -m driftless_train.train --epochs 30
Out:  artifacts/models/tcn_best.pt, artifacts/models/stats.json,
      artifacts/metrics/train_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    OUT_WIN,
    PROC_DIR,
    STRIDE_TRAIN,
    WIN,
    WindowDataset,
    compute_stats,
    load_index,
    make_splits,
    split_runs,
)
from .model import SpeedHeadingTCN
from .paths import METRIC_DIR, MODEL_DIR


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, device, y_std, opt=None, sched=None):
    """One pass. Loss is Huber on std-normalised residuals so the two heads --
    metres per second and radians -- contribute comparably."""
    train = opt is not None
    model.train(train)
    huber = nn.HuberLoss(delta=1.0, reduction="mean")
    tot, n = 0.0, 0
    abs_err = None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = huber((pred - y) / y_std, torch.zeros_like(y))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        bs = x.shape[0]
        tot += float(loss.detach()) * bs
        n += bs
        e = (pred - y).abs().sum(dim=0).detach().cpu().numpy()
        abs_err = e if abs_err is None else abs_err + e

    if abs_err is None:
        abs_err = np.zeros(3)
    return tot / max(n, 1), abs_err / max(n, 1)


@dataclass
class FitConfig:
    """Everything that defines a training run.

    Exists so cross-validation trains *identically* to the canonical path -- a
    duplicated loop would let the two drift, and then the CV estimate would not
    be an estimate of the shipped model.
    """

    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    width: int = 48
    device: str = "auto"


def fit(ds_tr, ds_va, n_channels: int, cfg: FitConfig,
        on_epoch=None) -> dict:
    """Train one model. Returns {model, stats, best_val, history}.

    `on_epoch(entry)` is called after every epoch with that epoch's history
    record, so callers can log however they like without this function knowing
    anything about files.
    """
    stats = compute_stats(ds_tr)
    device = pick_device(cfg.device)
    model = SpeedHeadingTCN(n_channels, width=cfg.width,
                            n_out=len(stats["y_mean"])).to(device)
    model.set_stats(stats)

    y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(len(dl_tr), 1),
        pct_start=0.25)

    best, best_state, history = float("inf"), None, []
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, dl_tr, device, y_std, opt, sched)
        va_loss, va_mae = run_epoch(model, dl_va, device, y_std)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_mae": va_mae.tolist(),
                        "lr": opt.param_groups[0]["lr"],
                        "secs": round(time.time() - t0, 1)})
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().clone() for k, v in
                          model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(history[-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "stats": stats, "best_val": best,
            "history": history, "device": device}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--win", type=int, default=WIN, help="context length, samples")
    ap.add_argument("--out-win", type=int, default=OUT_WIN,
                    help="output interval, samples")
    ap.add_argument("--stride", type=int, default=STRIDE_TRAIN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--min-coupling", type=float, default=0.0,
                    help="drop TRAIN runs whose phone/vehicle coupling is below "
                         "this; val/test are already restricted to trusted runs")
    ap.add_argument("--tag", default="", help="suffix for checkpoint filenames")
    ap.add_argument("--rotate-aug", action="store_true",
                    help="augment training windows with random mount rotations")
    ap.add_argument("--max-tilt-deg", type=float, default=60.0)
    ap.add_argument("--lowpass-aug", type=float, nargs="*", default=[],
                    metavar="HZ",
                    help="sensor-tier augmentation: also train on copies of "
                         "each run low-passed at these cutoffs (Hz), so the "
                         "model cannot rely on high-frequency vibration alone. "
                         "Suggested: --lowpass-aug 4 2 1")
    ap.add_argument("--invariant-only", action="store_true",
                    help="train on the 5 gravity-projected, mount-invariant "
                         "channels only")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for plumbing checks; ignores the split policy")
    args = ap.parse_args(argv)

    tag = f"_{args.tag}" if args.tag else ""
    runs, idx = load_index(args.proc_dir)
    channels = idx["channels"]
    print(f"{len(runs)} runs, {len(channels)} channels")

    if args.smoke:
        parts = {"train": runs, "val": runs, "test": runs}
        print("SMOKE MODE: train == val, numbers are meaningless")
    else:
        splits = make_splits(runs)
        parts = split_runs(runs, splits, min_coupling=args.min_coupling)
        for k, v in parts.items():
            print(f"  {k:5} {len(v):3d} runs")
        if not parts["train"] or not parts["val"]:
            print("\nnot enough routes for a real split yet -- use --smoke, or "
                  "prepare more routes first")
            return 1

    from .augment import invariant_channel_indices
    chan_idx = invariant_channel_indices() if args.invariant_only else None
    if chan_idx is not None:
        channels = [channels[i] for i in chan_idx]
        print(f"mount-invariant subset: {channels}")

    ds_tr = WindowDataset(parts["train"], win=args.win, stride=args.stride,
                          out_win=args.out_win, rotate_aug=args.rotate_aug,
                          max_tilt_deg=args.max_tilt_deg, channels=chan_idx,
                          lowpass_aug=tuple(args.lowpass_aug))
    # Validation is never augmented: we want a stable yardstick across epochs.
    ds_va = WindowDataset(parts["val"], win=args.win, stride=args.out_win,
                          out_win=args.out_win, channels=chan_idx)
    print(f"windows: train {len(ds_tr)}  val {len(ds_va)}")
    if not len(ds_tr) or not len(ds_va):
        print("empty window set")
        return 1

    stats = compute_stats(ds_tr)   # same computation fit() performs internally
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"stats{tag}.json").write_text(json.dumps(
        {**stats, "channels": channels, "win": args.win,
         "out_win": args.out_win, "fs_hz": idx["fs_hz"]},
        indent=2))
    from .dataset import TARGETS
    print("targets", TARGETS, "\n  mean", np.round(stats["y_mean"], 4).tolist(),
          "\n  std ", np.round(stats["y_std"], 4).tolist())

    cfg = FitConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, width=args.width,
                    device=args.device)

    log_path = METRIC_DIR / f"train_log{tag}.csv"
    f = log_path.open("w", newline="")
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss", "val_speed_mae_ms",
                "val_dpsi_mae_rad", "val_dpsi_mae_deg", "val_dv_mae_ms",
                "lr", "secs"])
    state = {"best": float("inf")}

    def on_epoch(e):
        ep, va_mae = e["epoch"], e["val_mae"]
        w.writerow([ep, f"{e['train_loss']:.6f}", f"{e['val_loss']:.6f}",
                    f"{va_mae[0]:.4f}", f"{va_mae[1]:.6f}",
                    f"{np.rad2deg(va_mae[1]):.4f}",
                    f"{va_mae[2]:.4f}" if len(va_mae) > 2 else "",
                    f"{e['lr']:.2e}", f"{e['secs']:.1f}"])
        f.flush()
        flag = ""
        if e["val_loss"] < state["best"]:
            state["best"] = e["val_loss"]
            flag = " *"
        print(f"ep {ep:3d}/{args.epochs}  train {e['train_loss']:.4f}  "
              f"val {e['val_loss']:.4f}"
              f"  speed MAE {va_mae[0]:.3f} m/s"
              f"  dpsi MAE {np.rad2deg(va_mae[1]):.3f} deg"
              f"  dv MAE {va_mae[2]:.3f} m/s  ({e['secs']:.0f}s){flag}",
              flush=True)

    out = fit(ds_tr, ds_va, len(channels), cfg, on_epoch=on_epoch)
    f.close()

    print(f"device {out['device']}  params {out['model'].n_params}")
    torch.save({"state_dict": out["model"].state_dict(),
                "stats": out["stats"], "channels": channels, "win": args.win,
                "out_win": args.out_win,
                "channel_indices": None if chan_idx is None else chan_idx.tolist(),
                "rotate_aug": args.rotate_aug, "width": args.width,
                "lowpass_aug": list(args.lowpass_aug),
                "epoch": min(out["history"],
                             key=lambda h: h["val_loss"])["epoch"]},
               MODEL_DIR / f"tcn_best{tag}.pt")
    print(f"\nbest val {out['best_val']:.4f} -> {MODEL_DIR/f'tcn_best{tag}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
