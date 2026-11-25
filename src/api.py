"""HTTP API and web UI for diagnosis, monitoring, and retraining."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import (
    AUTO_RETRAIN,
    FEATURE_PATH,
    METRICS_PATH,
    MIN_RETRAIN_SAMPLES,
    MODEL_PATH,
    RETRAIN_THRESHOLD,
    ROOT,
    TEST_DIR,
    TRAIN_DIR,
    UI_DIR,
)
from src.database import (
    active_run,
    create_run,
    get_run,
    init_db,
    insert_upload,
    latest_run,
    list_runs,
    list_uploads,
    log_prediction,
    pending_by_class,
    pending_count,
    prediction_stats,
    set_run_pid,
)
from src.prediction import predict_image, warmup
from src.preprocessing import (
    CLASS_NAMES,
    feature_stories,
    is_image_file,
    load_rgb,
    normalize_label,
)

STARTED = time.time()
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_ZIP_BYTES = 64 * 1024 * 1024


def _ensure_dirs() -> None:
    from src.config import ACCEPTED_DIR, INCOMING_DIR, RUNTIME_DIR

    for folder in (RUNTIME_DIR, INCOMING_DIR, ACCEPTED_DIR, UI_DIR, TEST_DIR, TRAIN_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for name in CLASS_NAMES:
        (INCOMING_DIR / name).mkdir(parents=True, exist_ok=True)
        (ACCEPTED_DIR / name).mkdir(parents=True, exist_ok=True)


def _warmup() -> None:
    if MODEL_PATH.exists():
        try:
            warmup(MODEL_PATH)
        except Exception as exc:  # the UI should still open if warmup fails
            print(f"Model warmup skipped: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_dirs()
    init_db()
    _warmup()
    yield


app = FastAPI(title="BeanLeaf Doctor", version="1.0.0", lifespan=lifespan)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _public_metrics() -> dict:
    raw = _read_json(METRICS_PATH)
    keys = (
        "accuracy",
        "loss",
        "precision",
        "recall",
        "f1",
        "auc",
        "confusion_matrix",
        "support",
        "learning_rate",
        "epochs_ran",
        "optimization",
        "uploaded_images",
        "replay_images",
        "classification_report",
    )
    return {key: raw[key] for key in keys if key in raw}


def _save_leaf(payload: bytes, label: str, original_name: str, source: str) -> dict:
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"{original_name} is larger than 8 MB.")
    try:
        image = load_rgb(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {original_name}: {exc}") from exc
    label = normalize_label(label)
    from src.config import INCOMING_DIR
    import uuid

    folder = INCOMING_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{label}_{uuid.uuid4().hex[:10]}.jpg"
    path = folder / filename
    image.save(path, format="JPEG", quality=90)
    stored = str(path.resolve())
    upload_id = insert_upload(original_name, stored, label, source, path.stat().st_size)
    return {"id": upload_id, "filename": filename, "label": label, "bytes": path.stat().st_size}


def _ingest_zip(payload: bytes) -> list[dict]:
    if len(payload) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=400, detail="Zip files must be 64 MB or smaller.")
    saved = []
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="That file is not a valid zip archive.") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.name.startswith(".") or not is_image_file(name) or len(name.parts) < 2:
                continue
            try:
                label = normalize_label(name.parts[-2])
            except ValueError:
                continue
            saved.append(_save_leaf(archive.read(info), label, name.name, "zip"))
    if not saved:
        raise HTTPException(
            status_code=400,
            detail=(
                "No labelled images found. Put images in folders named "
                "angular_leaf_spot, bean_rust, and healthy."
            ),
        )
    return saved


def _maybe_auto_retrain() -> dict | None:
    if not AUTO_RETRAIN or pending_count() < RETRAIN_THRESHOLD or active_run() is not None:
        return None
    return start_retrain("threshold")


def start_retrain(trigger: str) -> dict:
    if active_run() is not None:
        raise HTTPException(status_code=409, detail="A retraining run is already in progress.")
    waiting = pending_count()
    if waiting < MIN_RETRAIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at least {MIN_RETRAIN_SAMPLES} labelled images. {waiting} are waiting.",
        )
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="The initial model has not been trained yet.")
    run_id = create_run(trigger)
    process = subprocess.Popen(
        [sys.executable, "-m", "src.model", "retrain", "--run-id", str(run_id)],
        cwd=str(ROOT),
    )
    set_run_pid(run_id, process.pid)
    record = get_run(run_id)
    return record or {"id": run_id, "status": "running", "trigger_name": trigger}


@app.get("/api/health")
def health() -> dict:
    uptime = time.time() - STARTED
    return {
        "status": "up" if MODEL_PATH.exists() else "degraded",
        "model_loaded": MODEL_PATH.exists(),
        "uptime_seconds": round(uptime, 1),
        "started_at": datetime.fromtimestamp(STARTED, timezone.utc).isoformat(),
        "classes": list(CLASS_NAMES),
    }


@app.get("/api/dashboard")
def dashboard() -> dict:
    return {
        "health": health(),
        "metrics": _public_metrics(),
        "predictions": prediction_stats(),
        "retrain": {
            "pending": pending_count(),
            "threshold": RETRAIN_THRESHOLD,
            "minimum": MIN_RETRAIN_SAMPLES,
            "auto": AUTO_RETRAIN,
            "run": latest_run(),
        },
    }


@app.get("/api/insights")
def insights() -> dict:
    summary = _read_json(FEATURE_PATH)
    if not summary:
        from src.preprocessing import save_feature_summary

        if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
            raise HTTPException(status_code=404, detail="Feature summary is not available yet.")
        summary = save_feature_summary(TRAIN_DIR, FEATURE_PATH)
    summary["stories"] = feature_stories(summary)
    return summary


@app.get("/api/samples")
def samples() -> list[dict]:
    catalog = []
    for label in CLASS_NAMES:
        folder = TEST_DIR / label
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg"))[:2]:
            catalog.append(
                {
                    "label": label,
                    "filename": path.name,
                    "url": f"/api/samples/{label}/{path.name}",
                }
            )
    return catalog


@app.get("/api/samples/{label}/{filename}")
def sample_file(label: str, filename: str):
    try:
        class_name = normalize_label(label)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = (TEST_DIR / class_name / filename).resolve()
    if path.parent != (TEST_DIR / class_name).resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Choose an image first.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Images must be 8 MB or smaller.")
    if active_run() is not None:
        # Prediction stays available. The loaded weights are the last completed model.
        pass
    try:
        result = predict_image(payload, MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score that image: {exc}") from exc
    log_prediction(file.filename or "upload.jpg", result)
    return result


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] | None = File(default=None),
    label: str = Form(default=""),
    archive: UploadFile | None = File(default=None),
) -> dict:
    saved: list[dict] = []
    if archive is not None and archive.filename:
        saved.extend(_ingest_zip(await archive.read()))
    if files:
        if not label:
            raise HTTPException(status_code=400, detail="Choose a class for the uploaded images.")
        for item in files:
            name = item.filename or "leaf.jpg"
            if not is_image_file(Path(name)):
                raise HTTPException(status_code=400, detail=f"{name} is not a supported image.")
            saved.append(_save_leaf(await item.read(), label, name, "files"))
    if not saved:
        raise HTTPException(status_code=400, detail="Upload images or a zip of class folders.")
    triggered = _maybe_auto_retrain()
    return {
        "saved": saved,
        "pending": pending_count(),
        "pending_by_class": pending_by_class(),
        "threshold": RETRAIN_THRESHOLD,
        "auto_started": triggered,
    }


@app.post("/api/retrain")
def retrain() -> dict:
    return start_retrain("manual")


@app.get("/api/retrain/status")
def retrain_status() -> dict:
    return {
        "pending": pending_count(),
        "pending_by_class": pending_by_class(),
        "threshold": RETRAIN_THRESHOLD,
        "minimum": MIN_RETRAIN_SAMPLES,
        "auto": AUTO_RETRAIN,
        "run": latest_run(),
        "history": list_runs(),
        "uploads": list_uploads(),
    }


@app.get("/api/evaluate")
def evaluate(refresh: bool = False) -> dict:
    """Return held-out test metrics. Pass refresh=true to score the test set again."""
    if not refresh and METRICS_PATH.exists():
        return _public_metrics()
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No model is available to evaluate.")
    from src.model import evaluate_model, make_dataset
    from src.prediction import load_model

    model = load_model(MODEL_PATH)
    dataset = make_dataset(TEST_DIR, shuffle=False, augment=False)
    metrics = evaluate_model(model, dataset)
    return metrics


@app.get("/")
def index():
    page = UI_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="The UI has not been built.")
    return FileResponse(page)


UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")
