"""Shared paths and thresholds for the local app and the Docker image."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
RUNTIME_DIR = Path(os.environ.get("RUNTIME_DIR", ROOT / "runtime"))
UI_DIR = Path(os.environ.get("UI_DIR", ROOT / "ui"))

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
INCOMING_DIR = RUNTIME_DIR / "incoming"
ACCEPTED_DIR = RUNTIME_DIR / "accepted"
DB_PATH = RUNTIME_DIR / "pipeline.db"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODEL_DIR / "bean_leaf_model.keras"))
PRETRAINED_PATH = MODEL_DIR / "bean_leaf_pretrained.keras"
H5_PATH = MODEL_DIR / "bean_leaf_model.h5"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_PATH = MODEL_DIR / "feature_summary.json"
CLASS_PATH = MODEL_DIR / "class_names.json"

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "15"))
MIN_RETRAIN_SAMPLES = int(os.environ.get("MIN_RETRAIN_SAMPLES", "6"))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "1") == "1"
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "6"))
RETRAIN_EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", "4"))
