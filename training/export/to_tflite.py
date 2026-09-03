"""Export a trained checkpoint for both runtimes.

  TFLite -> the Android app  (android/app/src/main/assets/models/)
  ONNX   -> the C++ edge engine (edge-engine/)

IMPORTANT correction to the original plan in this file, which said
"PyTorch -> ONNX -> TFLite, via onnx2tf":

**onnx2tf does not work for this architecture.** Bisecting the graph showed the
residual connection is the culprit -- a block without `x + y` converts, and with
it the TFLite graph fails to prepare ("num_input_elements != num_output_elements
(3936 != 3)", where 3936 = 48x82 is a padded intermediate). Worse, a variant that
DID convert disagreed with PyTorch by 5.4e-2 while reporting success, which is the
dangerous kind of failure: it would have shipped a silently wrong model.

So TFLite is produced by rebuilding the network with Keras layers and porting the
trained weights across, then using the first-party Keras -> TFLite converter.
That matches PyTorch to ~1e-6. See driftless_train/keras_port.py for the
bisection and two further traps (Keras GroupNormalization defaults to
epsilon=1e-3 vs PyTorch's 1e-5; tf.lite.Optimize.DEFAULT silently applies int8
weight quantisation worth 2-8% error).

This file is a thin wrapper so the documented entry point keeps working:

    python -m driftless_train.export          # equivalent, more options
    python export/to_tflite.py --copy-to-consumers
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftless_train.export import main as export_main  # noqa: E402
from driftless_train.paths import (ANDROID_ASSETS_DIR, EDGE_MODELS_DIR,  # noqa: E402
                                   MODEL_DIR)


def copy_to_consumers() -> None:
    """Place the exports where the Android and edge-engine builds look for them."""
    for src, dest_dir in ((MODEL_DIR / "tcn_speed_heading.tflite", ANDROID_ASSETS_DIR),
                          (MODEL_DIR / "tcn_speed_heading.onnx", EDGE_MODELS_DIR),
                          (MODEL_DIR / "tcn_speed_heading.onnx.data", EDGE_MODELS_DIR)):
        if not src.exists():
            print(f"missing {src.name}; run the export first")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)
        print(f"{src.name} -> {dest_dir}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--copy-to-consumers"]
    rc = export_main(argv)
    if "--copy-to-consumers" in sys.argv[1:]:
        copy_to_consumers()
    raise SystemExit(rc)
