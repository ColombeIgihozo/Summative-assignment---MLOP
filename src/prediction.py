"""Load the saved network and score one leaf photograph."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import MODEL_PATH
from src.preprocessing import CLASS_NAMES, batch_from_images, extract_features, load_rgb

_MODEL = None
_MODEL_MTIME = None
_MODEL_FILE = None


def load_model(model_path: Path | None = None, force: bool = False):
    """Cache the active network and reload it after a retrain writes a new file."""
    global _MODEL, _MODEL_MTIME, _MODEL_FILE
    import tensorflow as tf

    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one with: python -m src.model train"
        )
    mtime = path.stat().st_mtime
    if force or _MODEL is None or _MODEL_FILE != path or _MODEL_MTIME != mtime:
        _MODEL = tf.keras.models.load_model(path)
        _MODEL_MTIME = mtime
        _MODEL_FILE = path
    return _MODEL


def predict_image(source, model_path: Path | None = None) -> dict:
    """Return the class, confidence, and the colour/texture features of one image."""
    import time

    started = time.perf_counter()
    image = load_rgb(source)
    features = extract_features(image)
    batch = batch_from_images([image])
    model = load_model(model_path)
    probabilities = model.predict(batch, verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    index = int(np.argmax(probabilities))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "label": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "features": features,
        "latency_ms": round(elapsed_ms, 2),
        "model_path": str(Path(model_path or MODEL_PATH)),
    }


def warmup(model_path: Path | None = None) -> None:
    """Run one dummy leaf through the network so the first real request is faster."""
    from PIL import Image

    blank = Image.new("RGB", (224, 224), (70, 120, 60))
    predict_image(blank, model_path=model_path)
