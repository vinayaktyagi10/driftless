"""Export the trained regressor for both runtimes, and prove the exports agree.

Two consumers, one brain:
  ONNX   -> the C++ edge engine (roles 04-05) via ONNX Runtime
  TFLite -> the Android app (role 01) via LiteRT

Normalisation lives inside the graph, so neither consumer has to reimplement it.
Every export is checked numerically against PyTorch before it is accepted -- a
silently wrong export is worse than a missing one, because it fails on stage.

Run:  python -m driftless.export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_model

from .paths import MODEL_DIR

# Parity tolerance is expressed as a fraction of each output's own standard
# deviation, not as a raw absolute number. The outputs are in SI units (speed in
# m/s with std ~4.5, heading change in rad with std ~0.25), so one absolute
# threshold cannot be meaningful for both. 0.1% of a target's std is far below
# the model's own error and far above float32 kernel noise.
TOL_ONNX_FRAC = 1e-3
TOL_TFLITE_FRAC = 2e-2


def sample_inputs(n_ch: int, win: int, n: int = 64, seed: int = 0) -> np.ndarray:
    """Windows for parity testing: real recorded ones if we have them.

    Synthetic Gaussian noise is a poor parity test -- it lands far outside the
    input distribution, where the graphs can differ in ways that never occur in
    service. Real windows test the region the model actually operates in.
    """
    from .dataset import PROC_DIR
    proc = sorted(PROC_DIR.glob("*.npz"))
    if proc:
        xs = []
        for f in proc:
            with np.load(f) as z:
                feats, valid = z["features"], z["valid"]
            step = max((len(feats) - win) // 32, 1)
            for start in range(0, len(feats) - win, step):
                if valid[start:start + win].all():
                    xs.append(feats[start:start + win].T)
                if len(xs) >= n:
                    break
            if len(xs) >= n:
                break
        if len(xs) >= 8:
            return np.stack(xs[:n]).astype(np.float32)

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, n_ch, win)).astype(np.float32)
    x[:, 2, :] += 9.81      # acc_z sits near gravity
    return x


def _parity(y_ref: np.ndarray, y_new: np.ndarray, frac: float) -> dict:
    """Per-output parity, scaled by each output's spread across the test batch."""
    scale = np.maximum(y_ref.std(axis=0), 1e-6)
    diff = np.abs(y_ref - y_new).max(axis=0)
    rel = diff / scale
    return {
        "max_abs_diff_per_output": [round(float(d), 8) for d in diff],
        "output_std": [round(float(s), 6) for s in scale],
        "max_rel_diff": round(float(rel.max()), 8),
        "tolerance_rel": frac,
        "passed": bool(rel.max() < frac),
    }


def export_onnx(model, n_ch: int, win: int, out: Path) -> dict:
    dummy = torch.zeros(1, n_ch, win, dtype=torch.float32)
    torch.onnx.export(
        model, (dummy,), str(out),
        input_names=["imu_window"], output_names=["speed_dpsi"],
        dynamic_axes={"imu_window": {0: "batch"}, "speed_dpsi": {0: "batch"}},
        opset_version=18, do_constant_folding=True,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    X = sample_inputs(n_ch, win)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(X)).numpy()
    y_onnx = sess.run(["speed_dpsi"], {"imu_window": X})[0]

    result = {"path": str(out), "size_kb": round(out.stat().st_size / 1024, 1),
              "n_test_windows": int(len(X)),
              **_parity(y_torch, y_onnx, TOL_ONNX_FRAC)}

    # Latency, single window -- the number role 01 and role 02 actually care about.
    x1 = X[:1]
    import time
    for _ in range(20):
        sess.run(["speed_dpsi"], {"imu_window": x1})
    t0 = time.perf_counter()
    N = 500
    for _ in range(N):
        sess.run(["speed_dpsi"], {"imu_window": x1})
    result["latency_ms_per_window"] = round((time.perf_counter() - t0) / N * 1000, 4)
    return result


def export_tflite(out_dir: Path, n_ch: int, win: int, width: int, n_out: int,
                  dilations: tuple[int, ...], model) -> dict:
    """Export TFLite by rebuilding the network in Keras and porting the weights.

    Not by converting the ONNX: onnx2tf mistranslates the residual connection in
    this architecture (the TFLite graph fails to prepare, and a variant that did
    convert was 5.4e-2 off PyTorch -- silently wrong). See keras_port for the
    bisection. The Keras route matches PyTorch to ~1e-6.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        return {"passed": False, "skipped": True,
                "reason": f"tensorflow not installed: {type(e).__name__}",
                "how_to_enable": "uv pip install tensorflow"}

    from .keras_port import keras_from_checkpoint, run_tflite, to_tflite

    km = keras_from_checkpoint(model, n_ch, win, width, n_out, dilations)

    X = sample_inputs(n_ch, win, n=48)            # (N, C, W)
    X_nwc = np.transpose(X, (0, 2, 1))            # Keras/TFLite are channels-last
    with torch.no_grad():
        y_torch = model(torch.from_numpy(X)).numpy()

    y_keras = km.predict(X_nwc, verbose=0)
    keras_parity = _parity(y_torch, y_keras, TOL_TFLITE_FRAC)

    blob = to_tflite(km, quantise="none")
    out = out_dir / "tcn_speed_heading.tflite"
    out.write_bytes(blob)
    y_tfl = run_tflite(blob, X_nwc)

    import time
    t0 = time.perf_counter()
    N = 200
    run_tflite(blob, X_nwc[:1].repeat(N, axis=0))
    latency = (time.perf_counter() - t0) / N * 1000

    return {
        "path": str(out),
        "size_kb": round(out.stat().st_size / 1024, 1),
        "skipped": False,
        "via": "keras weight port (not onnx2tf)",
        "input_shape": [1, win, n_ch],
        "input_layout": "NWC / channels-last (time-major) -- note this differs "
                        "from the ONNX model's NCW",
        "quantisation": "none (float32). int8 dynamic-range costs 2-8% error, "
                        "float16 is unusable here (>100%); the model is small "
                        "enough that neither is worth it.",
        "keras_vs_torch_max_rel": keras_parity["max_rel_diff"],
        "latency_ms_per_window": round(latency, 4),
        **_parity(y_torch, y_tfl, TOL_TFLITE_FRAC),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=MODEL_DIR / "tcn_best.pt")
    ap.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--skip-tflite", action="store_true")
    args = ap.parse_args(argv)

    model, win, out_win = load_model(args.ckpt, torch.device("cpu"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_ch = len(ck["channels"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from .dataset import TARGETS
    report = {"channels": ck["channels"], "win": win, "out_win": out_win,
              "n_params": model.n_params,
              "input_shape": [1, n_ch, win],
              "context_s": round(win * 0.1, 2),
              "output_interval_s": round(out_win * 0.1, 2),
              "output": list(TARGETS),
              "notes": "Input normalisation and output de-normalisation are baked "
                       "into the graph; feed raw SI-unit IMU features and read "
                       "SI-unit outputs."}

    onnx_path = args.out_dir / "tcn_speed_heading.onnx"
    report["onnx"] = export_onnx(model, n_ch, win, onnx_path)
    r = report["onnx"]
    print(f"ONNX   {'PASS' if r['passed'] else 'FAIL'}  {r['size_kb']} KB  "
          f"max rel Δ {r['max_rel_diff']:.2e} (tol {r['tolerance_rel']:.0e})  "
          f"{r['latency_ms_per_window']:.3f} ms/window  "
          f"on {r['n_test_windows']} real windows")

    if args.skip_tflite:
        report["tflite"] = {"skipped": True, "reason": "--skip-tflite"}
    else:
        report["tflite"] = export_tflite(
            args.out_dir, n_ch, win, ck["width"], model.n_out,
            tuple(1 << i for i in range(len(model.blocks))), model)
    t = report["tflite"]
    if t.get("skipped"):
        print(f"TFLite SKIP  {t.get('reason','')}")
        if "how_to_enable" in t:
            print(f"       enable: {t['how_to_enable']}")
    else:
        print(f"TFLite {'PASS' if t['passed'] else 'FAIL'}  {t['size_kb']} KB  "
              f"max rel Δ {t['max_rel_diff']:.2e} (tol {t['tolerance_rel']:.0e})  "
              f"{t['latency_ms_per_window']:.3f} ms/window  "
              f"input {t['input_shape']} {t['input_layout'].split(' --')[0]}")

    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"-> {args.out_dir/'export_report.json'}")
    ok = report["onnx"]["passed"] and (report["tflite"].get("skipped")
                                       or report["tflite"]["passed"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
