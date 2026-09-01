"""Single source of truth for every path the pipeline reads or writes.

Each module used to derive its own root from `__file__`, which broke the moment
the package moved. One definition here instead.

Layout, relative to the repo root:

    training/data/io-vnbd/     raw IO-VNBD CSVs (gitignored; see download.py)
    training/data/processed/   per-run arrays from prepare.py (gitignored)
    training/configs/          frozen manifest + train/val/test split
    training/artifacts/        models, metrics, plots, ROUND1_EVIDENCE.md
"""

from __future__ import annotations

from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]      # training/
REPO_ROOT = TRAIN_ROOT.parent                          # the repo itself

DATA_DIR = TRAIN_ROOT / "data"
RAW_DIR = DATA_DIR / "io-vnbd"
PROC_DIR = DATA_DIR / "processed"

CONFIG_DIR = TRAIN_ROOT / "configs"
MANIFEST_PATH = CONFIG_DIR / "iovnbd_manifest.json"
SPLITS_PATH = CONFIG_DIR / "splits.json"

ARTIFACT_DIR = TRAIN_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
METRIC_DIR = ARTIFACT_DIR / "metrics"
PLOT_DIR = ARTIFACT_DIR / "plots"
EVIDENCE_PATH = ARTIFACT_DIR / "ROUND1_EVIDENCE.md"

# Where the other roles' builds expect the exported models. to_tflite.py can copy
# there; the files themselves are produced by driftless_train.export.
ANDROID_ASSETS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
EDGE_MODELS_DIR = REPO_ROOT / "edge-engine" / "models"
