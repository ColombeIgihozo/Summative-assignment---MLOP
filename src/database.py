"""SQLite records for uploads, predictions, and retraining runs."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from src.config import DB_PATH, RUNTIME_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect() -> sqlite3.Connection:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                bytes INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY,
                trigger_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                sample_count INTEGER DEFAULT 0,
                metrics_json TEXT,
                model_path TEXT,
                message TEXT,
                pid INTEGER
            );
            """
        )


def insert_upload(filename: str, stored_path: str, label: str, source: str, size: int) -> int:
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO uploads (filename, stored_path, label, source, bytes, uploaded_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (filename, stored_path, label, source, size, utc_now()),
        )
        return int(cursor.lastrowid)


def list_uploads(limit: int = 40) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM uploads ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def pending_count() -> int:
    with connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM uploads WHERE status = 'pending'"
        ).fetchone()
    return int(row["total"])


def pending_by_class() -> dict[str, int]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT label, COUNT(*) AS total
            FROM uploads
            WHERE status = 'pending'
            GROUP BY label
            """
        ).fetchall()
    return {row["label"]: int(row["total"]) for row in rows}


def mark_accepted(paths: list[str]) -> None:
    if not paths:
        return
    with connect() as connection:
        connection.executemany(
            "UPDATE uploads SET status = 'accepted' WHERE stored_path = ?",
            [(path,) for path in paths],
        )


def log_prediction(filename: str, result: dict) -> None:
    try:
        with connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions
                    (filename, predicted_label, confidence, probabilities, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    result["label"],
                    result["confidence"],
                    json.dumps(result["probabilities"]),
                    result["latency_ms"],
                    utc_now(),
                ),
            )
    except sqlite3.OperationalError:
        # A flood of requests can lock SQLite. The prediction itself should still succeed.
        return


def prediction_stats() -> dict:
    with connect() as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(latency_ms) AS avg_latency,
                   AVG(confidence) AS avg_confidence
            FROM predictions
            """
        ).fetchone()
        by_class = connection.execute(
            """
            SELECT predicted_label, COUNT(*) AS total
            FROM predictions
            GROUP BY predicted_label
            """
        ).fetchall()
        recent = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 8"
        ).fetchall()
    return {
        "total": int(totals["total"] or 0),
        "average_latency_ms": round(float(totals["avg_latency"] or 0), 2),
        "average_confidence": round(float(totals["avg_confidence"] or 0), 4),
        "by_class": {row["predicted_label"]: int(row["total"]) for row in by_class},
        "recent": [dict(row) for row in recent],
    }


def create_run(trigger_name: str, pid: int | None = None) -> int:
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO training_runs (trigger_name, status, started_at, message, pid)
            VALUES (?, 'running', ?, 'Retraining has started.', ?)
            """,
            (trigger_name, utc_now(), pid),
        )
        return int(cursor.lastrowid)


def set_run_pid(run_id: int, pid: int) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE training_runs SET pid = ? WHERE id = ?",
            (pid, run_id),
        )


def update_run(run_id: int, message: str, sample_count: int | None = None) -> None:
    with connect() as connection:
        if sample_count is None:
            connection.execute(
                "UPDATE training_runs SET message = ? WHERE id = ?",
                (message, run_id),
            )
        else:
            connection.execute(
                "UPDATE training_runs SET message = ?, sample_count = ? WHERE id = ?",
                (message, sample_count, run_id),
            )


def finish_run(run_id: int, status: str, metrics: dict | None, model_path: str | None, message: str | None = None) -> None:
    metrics = metrics or {}
    if message is None:
        if status == "completed":
            accuracy = metrics.get("accuracy")
            uploaded = metrics.get("uploaded_images", 0)
            replay = metrics.get("replay_images", 0)
            score = f"{accuracy:.3f}" if isinstance(accuracy, float) else "n/a"
            message = (
                f"Preprocessed {uploaded} uploaded images (RGB, 224x224, MobileNet scaling), "
                f"mixed them with {replay} replay images, and fine-tuned the saved custom model. "
                f"Test accuracy is {score}."
            )
        else:
            message = metrics.get("error") or "Retraining failed."
    with connect() as connection:
        connection.execute(
            """
            UPDATE training_runs
            SET status = ?, finished_at = ?, metrics_json = ?, model_path = ?, message = ?,
                sample_count = COALESCE(?, sample_count)
            WHERE id = ?
            """,
            (
                status,
                utc_now(),
                json.dumps(metrics),
                model_path,
                message,
                metrics.get("uploaded_images"),
                run_id,
            ),
        )


def get_run(run_id: int) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM training_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def latest_run() -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 8) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def active_run() -> dict | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT * FROM training_runs
            WHERE status = 'running'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    pid = record.get("pid")
    if pid and not _pid_alive(int(pid)):
        finish_run(record["id"], "failed", {"error": "The retraining process stopped unexpectedly."}, None)
        record["status"] = "failed"
        return None
    return record


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
