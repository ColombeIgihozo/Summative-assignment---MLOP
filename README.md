# BeanLeaf Doctor

BeanLeaf Doctor is an end-to-end image classification pipeline for bean leaves photographed in the field. It diagnoses three classes from the Makerere University iBean dataset: angular leaf spot, bean rust, and healthy. The same saved network serves a web page, a Python API, a retraining job, and a Docker deployment that can be flooded with Locust.

The photographs were collected in Uganda by the Makerere AI Lab with the National Crops Resources Research Institute. They are leaf images, not a spreadsheet.

## Video demo

Record this with your camera on and upload it to YouTube. The recording needs to show both a prediction and a retraining run.

1. Open the app and point at the uptime indicator.
2. On **Diagnose a leaf**, click a known test leaf and show that the predicted class matches the button label. Then upload one photo of your own.
3. On **What the photos show**, scroll through the three feature charts.
4. On **Upload and retrain**, add at least six labelled leaf images and press **Retrain on uploaded leaves**. Leave the camera on until the run status changes to completed.
5. Diagnose one of the known test leaves again.

Paste the finished video here:

**YouTube:** https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_VIDEO

## URL

| Where | Address |
| --- | --- |
| Local web app and API | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |
| Cloud | Deploy with the steps below, then replace this cell with the public URL. |

**Public URL:** https://REPLACE_WITH_YOUR_DEPLOYED_URL

## What the project does

A leaf photo is resized to 224 by 224 and scaled to the range MobileNetV2 expects. A custom head, trained on top of ImageNet MobileNetV2 with dropout, L2 regularisation, Adam, early stopping, and a short learning-rate search, returns one class and a confidence. The first trained network is kept as `models/bean_leaf_pretrained.keras`. Later retraining loads the active custom model and fine-tunes it. It does not train a new network from random weights.

Uploaded leaves are written to `runtime/incoming/<class>/` and recorded in SQLite (`runtime/pipeline.db`). Two triggers start retraining:

- **Button:** `POST /api/retrain`, used by **Retrain on uploaded leaves**. It requires at least 6 new images.
- **Threshold:** after an upload, if 15 or more labelled images are still waiting and `AUTO_RETRAIN=1`, the API starts the same job on its own.

The retraining job preprocesses the new files (RGB, 224 by 224, MobileNet scaling), mixes them with a replay sample of the original training leaves, fine-tunes the saved model at a learning rate of `0.0001`, scores the untouched test set, and writes the new weights back to `models/bean_leaf_model.keras`.

## Results from the flood-request simulation

Locust sends health checks and leaf-photo predictions through a gateway that spreads them across the API replicas. Each run used 30 users, spawned 5 at a time, for 40 seconds. A single prediction with nobody else waiting takes about 340 ms.

Docker Desktop is running on this PC and serves the app at http://localhost:8000 through the nginx gateway. The flood numbers below were measured before that engine was available, against 1 and 3 local API processes behind `deploy/process_gateway.py`. That gateway has the same shape as the nginx service in `docker-compose.yml`. The Docker commands in the setup section repeat the flood against real containers.

| API replicas | Predict requests | Failures | Median (ms) | Average (ms) | p95 (ms) | Fastest (ms) | Predictions per second |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 87 | 0 | 5400 | 5667 | 9400 | 992 | 2.68 |
| 3 | 179 | 0 | 3100 | 3599 | 9800 | 361 | 4.62 |

One replica still answered every request, but the 30 users queued and the median rose from about a third of a second to 5.4 seconds. Three replicas almost doubled throughput, from 2.68 to 4.62 predictions per second, and cut the median to 3.1 seconds. The 95th percentile stayed near 10 seconds because the three processes share one CPU, so the slowest requests are contention rather than failed calls. Raw Locust tables are in `results/locust_1_replica_stats.csv` and `results/locust_3_replicas_stats.csv`.

## Repository layout

```
README.md
notebook/bean_leaf_pipeline.ipynb
src/preprocessing.py
src/model.py
src/prediction.py
src/acquire.py
src/api.py
src/database.py
data/train/
data/validation/
data/test/
models/bean_leaf_model.keras
models/bean_leaf_model.h5
models/bean_leaf_pretrained.keras
ui/
locustfile.py
Dockerfile
docker-compose.yml
```

## Setup

Use Python 3.11. TensorFlow 2.17 is pinned to that version.

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m src.acquire
.\.venv\Scripts\python.exe -m ipykernel install --user --name beanleaf --display-name "Python 3.11 (beanleaf)"
```

The iBean zip archives, if you already downloaded them into `data/`, are extracted by `python -m src.acquire`. Otherwise that command downloads the Hugging Face mirror `AI-Lab-Makerere/beans`.

### Train and evaluate

Open `notebook/bean_leaf_pipeline.ipynb` and run every cell, or execute it from the project root:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebook/bean_leaf_pipeline.ipynb --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=beanleaf
```

The notebook saves the model, the metrics, and the feature summary used by the charts. Training uses early stopping, so it may finish before the epoch cap.

### Run the web app

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000.

- **Diagnose a leaf** uploads one image, or sends a labelled test image, and shows the predicted class.
- **What the photos show** charts green share, rust index, and lesion contrast.
- **Upload and retrain** stores a batch of images or a zip whose folders are named `angular_leaf_spot`, `bean_rust`, and `healthy`. The button starts fine-tuning. Crossing 15 new images starts it automatically.
- **Model uptime** shows how long the process has been up, the test-set metrics, and recent requests.

### Docker and the flood test

Build after the model file exists. Stop any local server that is already using port 8000. `docker compose up -d --build` starts the API and the nginx gateway on port 8000.

```powershell
docker compose up -d --build --scale api=1
.\.venv\Scripts\python.exe -m locust -f locustfile.py --headless -u 30 -r 5 -t 45s --host http://localhost:8000 --csv results/locust_1_container
docker compose up -d --scale api=3
.\.venv\Scripts\python.exe -m locust -f locustfile.py --headless -u 30 -r 5 -t 45s --host http://localhost:8000 --csv results/locust_3_containers
docker compose down
```

Locust posts a real test photograph to `/api/predict` and also calls `/api/health`. Read the median, average, and 95th percentile from the printed summary or from the CSV files in `results/`.

### Cloud

The service is a Docker web app. On [Render](https://render.com), create a new Blueprint from `render.yaml` or a new Web Service with the Docker runtime, then set the health check path to `/api/health`. The free and starter instances have 512 MB of RAM, which is too small for TensorFlow. `render.yaml` uses the standard plan, which has 2 GB. After deploy, put the public URL in the URL section above.

The same image runs on any host that has Docker:

```powershell
docker compose up -d --build
```

Port 8000 is the gateway. The API containers are not published directly, so every request, including a flood test, goes through nginx and is shared across the running copies.

## Model metrics

The held-out test set has 128 leaves. The notebook and the uptime page both report:

| Accuracy | Loss | Precision | Recall | F1 | AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.906 | 0.265 | 0.911 | 0.907 | 0.904 | 0.976 |

Angular leaf spot and healthy are rarely confused. Bean rust is the harder class: 33 of 43 test leaves were labelled rust, and most of the mistakes went to angular leaf spot. The learning-rate search kept Adam at `0.0005`. Training ran for 6 epochs with early stopping armed, dropout, L2, and the frozen ImageNet MobileNetV2 backbone.

## Dataset credit

Makerere AI Lab and NaCRRI. iBean leaf images, MIT licence. Mirror used here: [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans). The original Google Cloud zip links currently return HTTP 403.
