"""Flood the diagnosis API with health checks and leaf photographs.

Run against the nginx gateway so the requests are shared across API containers:

    locust -f locustfile.py --headless -u 30 -r 5 -t 45s --host http://localhost:8000 --csv results/locust_c1
"""

from __future__ import annotations

from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent


def _sample_leaf() -> tuple[str, bytes]:
    folder = ROOT / "data" / "test" / "healthy"
    images = sorted(folder.glob("*.jpg"))
    if not images:
        raise FileNotFoundError("No healthy test image is available for the flood test.")
    return images[0].name, images[0].read_bytes()


class DiagnosisUser(HttpUser):
    wait_time = between(0.2, 0.8)

    def on_start(self):
        self.sample_name, self.sample_bytes = _sample_leaf()

    @task(3)
    def predict(self):
        self.client.post(
            "/api/predict",
            files={"file": (self.sample_name, self.sample_bytes, "image/jpeg")},
            name="/api/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/api/health", name="/api/health")
