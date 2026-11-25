const NAMES = {
  angular_leaf_spot: "Angular leaf spot",
  bean_rust: "Bean rust",
  healthy: "Healthy",
};
const SHORT = {
  angular_leaf_spot: "Spot",
  bean_rust: "Rust",
  healthy: "Healthy",
};
const COLORS = {
  angular_leaf_spot: "#b64024",
  bean_rust: "#c9842a",
  healthy: "#1f6b45",
};
const ADVICE = {
  angular_leaf_spot: "Dark angular lesions fit angular leaf spot. Avoid working the crop while it is wet, remove badly infected leaves, and follow a locally recommended fungicide.",
  bean_rust: "Reddish-brown pustules fit bean rust. Improve airflow, check neighbouring plants, and treat early if the crop is still in the field.",
  healthy: "Nothing in this photo points to the two trained diseases. Keep scouting after rain. A single photo is decision support, not a certified diagnosis.",
};
const FEATURE_LABELS = {
  green_dominance: "Green share",
  rust_index: "Rust index",
  lesion_contrast: "Lesion contrast",
  edge_density: "Edge density",
};

const charts = {};
let startedAt = null;
let pollTimer = null;

function pretty(name) {
  return NAMES[name] || name;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

async function readJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string" ? detail : "The request failed.";
    throw new Error(message);
  }
  return payload;
}

function showPanel(name) {
  document.querySelectorAll("nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.panel === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
  if (name === "insights") loadInsights();
  if (name === "retrain") loadRetrain();
  if (name === "monitor") loadMonitor();
}

document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => showPanel(button.dataset.panel));
});

function tickUptime() {
  if (!startedAt) return;
  const seconds = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const secs = String(seconds % 60).padStart(2, "0");
  document.getElementById("uptime").textContent = `Up for ${hours}:${minutes}:${secs}`;
}

async function refreshHealth() {
  try {
    const health = await readJson("/api/health");
    const dot = document.getElementById("status-dot");
    const label = document.getElementById("status-label");
    dot.classList.toggle("up", health.status === "up");
    dot.classList.toggle("down", health.status !== "up");
    label.textContent = health.status === "up" ? "Model is up" : "Model file missing";
    startedAt = Date.now() - health.uptime_seconds * 1000;
    tickUptime();
  } catch (error) {
    document.getElementById("status-label").textContent = "API unreachable";
    document.getElementById("status-dot").classList.add("down");
  }
}

function showPreview(file) {
  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
}

async function predictFile(file, expectedLabel) {
  showPreview(file);
  const body = new FormData();
  body.append("file", file, file.name || "leaf.jpg");
  const result = await readJson("/api/predict", { method: "POST", body });
  renderPrediction(result, expectedLabel);
}

function renderPrediction(result, expectedLabel) {
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result").hidden = false;
  const expected = document.getElementById("expected");
  if (expectedLabel) {
    const correct = expectedLabel === result.label;
    expected.hidden = false;
    expected.className = correct ? "expected ok" : "expected bad";
    expected.textContent = correct
      ? `Correct. This test leaf is ${pretty(expectedLabel)}.`
      : `This test leaf is ${pretty(expectedLabel)}, but the model said ${pretty(result.label)}.`;
  } else {
    expected.hidden = true;
  }
  document.getElementById("pred-label").textContent = pretty(result.label);
  document.getElementById("pred-confidence").textContent = `${percent(result.confidence)} confidence`;
  document.getElementById("advice").textContent = ADVICE[result.label] || "";
  const bars = document.getElementById("bars");
  bars.innerHTML = Object.entries(result.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => `
      <div class="bar-row">
        <span>${pretty(name)}</span>
        <div class="bar"><span style="width:${(value * 100).toFixed(1)}%;background:${COLORS[name]}"></span></div>
        <strong>${percent(value)}</strong>
      </div>
    `).join("");
  const features = document.getElementById("pred-features");
  features.innerHTML = Object.entries(FEATURE_LABELS).map(([key, label]) => `
    <div><dt>${label}</dt><dd>${result.features[key].toFixed(3)}</dd></div>
  `).join("");
  document.getElementById("pred-meta").textContent = `Response time ${result.latency_ms.toFixed(0)} ms.`;
}

const drop = document.getElementById("drop");
const leafFile = document.getElementById("leaf-file");
leafFile.addEventListener("change", () => {
  const file = leafFile.files[0];
  if (file) predictFile(file).catch(showPredictError);
});
["dragenter", "dragover"].forEach((eventName) => {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.add("hot");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.remove("hot");
  });
});
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) predictFile(file).catch(showPredictError);
});

function showPredictError(error) {
  document.getElementById("result-empty").hidden = false;
  document.getElementById("result").hidden = true;
  document.getElementById("result-empty").textContent = error.message;
}

async function loadSamples() {
  const samples = await readJson("/api/samples");
  const host = document.getElementById("samples");
  host.innerHTML = "";
  samples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${pretty(sample.label)} sample`;
    button.addEventListener("click", async () => {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], sample.filename, { type: blob.type || "image/jpeg" });
      predictFile(file, sample.label).catch(showPredictError);
    });
    host.appendChild(button);
  });
}

function destroyChart(name) {
  if (charts[name]) {
    charts[name].destroy();
    delete charts[name];
  }
}

function barChart(canvasId, feature, summary) {
  destroyChart(canvasId);
  const labels = summary.classes;
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: labels.map((name) => SHORT[name] || pretty(name)),
      datasets: [{
        data: labels.map((name) => summary.features[feature][name].mean),
        backgroundColor: labels.map((name) => COLORS[name]),
        borderRadius: 8,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    },
  });
}

async function loadInsights() {
  const summary = await readJson("/api/insights");
  const host = document.getElementById("stories");
  host.innerHTML = summary.stories.slice(0, 3).map((story, index) => `
    <article class="card story">
      <canvas id="chart-${index}"></canvas>
      <div>
        <h2>${story.title}</h2>
        <p>${story.text}</p>
      </div>
    </article>
  `).join("");
  summary.stories.slice(0, 3).forEach((story, index) => {
    barChart(`chart-${index}`, story.feature, summary);
  });
  destroyChart("scatter");
  charts.scatter = new Chart(document.getElementById("scatter"), {
    type: "scatter",
    data: {
      datasets: summary.classes.map((name) => ({
        label: pretty(name),
        data: summary.features.green_dominance[name].values.map((green, index) => ({
          x: green,
          y: summary.features.rust_index[name].values[index],
        })),
        backgroundColor: COLORS[name],
      })),
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Green share" } },
        y: { title: { display: true, text: "Rust index" } },
      },
    },
  });
  const greenLeader = summary.classes.reduce((best, name) => (
    summary.features.green_dominance[name].mean > summary.features.green_dominance[best].mean ? name : best
  ));
  document.getElementById("scatter-caption").textContent =
    `Healthy tissue should sit toward a higher green share. In this sample, ${pretty(greenLeader)} is the greenest class. Rustier leaves sit higher on the vertical axis.`;
}

function renderUploads(rows) {
  const body = document.getElementById("upload-rows");
  if (!rows.length) {
    body.innerHTML = "<tr><td colspan='5'>No uploads yet.</td></tr>";
    return;
  }
  body.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.filename}</td>
      <td>${pretty(row.label)}</td>
      <td>${row.source}</td>
      <td>${row.status}</td>
      <td>${row.uploaded_at.replace("T", " ").replace("+00:00", " UTC")}</td>
    </tr>
  `).join("");
}

function renderRun(status) {
  const pending = status.pending || 0;
  const threshold = status.threshold || 1;
  const ratio = Math.min(1, pending / threshold);
  document.getElementById("meter-fill").style.width = `${ratio * 100}%`;
  const auto = status.auto
    ? `Automatic retraining starts at ${threshold} new leaves.`
    : "Automatic retraining is off. Use the button.";
  document.getElementById("meter-text").textContent =
    `${pending} labelled image${pending === 1 ? "" : "s"} waiting. At least ${status.minimum} are required. ${auto}`;
  const run = status.run;
  const message = document.getElementById("run-message");
  const button = document.getElementById("retrain-button");
  if (run && run.status === "running") {
    message.textContent = run.message || "Retraining is running. The current model stays available for predictions.";
    button.disabled = true;
    button.textContent = "Retraining…";
  } else {
    button.disabled = false;
    button.textContent = "Retrain on uploaded leaves";
    message.textContent = run ? `${run.trigger_name} run ${run.status}. ${run.message || ""}` : "No retraining run yet.";
  }
  const history = document.getElementById("run-history");
  history.innerHTML = (status.history || []).slice(0, 4).map((item) => `
    <p class="meta">${item.started_at} · ${item.trigger_name} · ${item.status}</p>
  `).join("");
  renderUploads(status.uploads || []);
  if (run && run.status === "running" && !pollTimer) {
    pollTimer = setInterval(loadRetrain, 3000);
  }
  if ((!run || run.status !== "running") && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function loadRetrain() {
  const status = await readJson("/api/retrain/status");
  renderRun(status);
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = document.getElementById("upload-message");
  const body = new FormData();
  const files = document.getElementById("bulk-files").files;
  const archive = document.getElementById("bulk-zip").files[0];
  if (!files.length && !archive) {
    message.textContent = "Choose images or a zip file first.";
    return;
  }
  if (files.length) body.append("label", document.getElementById("class-label").value);
  Array.from(files).forEach((file) => body.append("files", file));
  if (archive) body.append("archive", archive);
  message.textContent = "Saving…";
  try {
    const result = await readJson("/api/upload", { method: "POST", body });
    const started = result.auto_started ? " The automatic trigger started retraining." : "";
    message.textContent = `Saved ${result.saved.length} image${result.saved.length === 1 ? "" : "s"}.${started}`;
    event.target.reset();
    loadRetrain();
  } catch (error) {
    message.textContent = error.message;
  }
});

document.getElementById("retrain-button").addEventListener("click", async () => {
  const message = document.getElementById("run-message");
  try {
    await readJson("/api/retrain", { method: "POST" });
    message.textContent = "Retraining started from the saved custom model.";
    loadRetrain();
  } catch (error) {
    message.textContent = error.message;
  }
});

function renderMatrix(metrics) {
  const matrix = metrics.confusion_matrix;
  const host = document.getElementById("matrix");
  if (!matrix) {
    host.innerHTML = "<p class='meta'>Train the model to see the confusion matrix.</p>";
    return;
  }
  const labels = Object.keys(NAMES);
  const header = `<div><strong></strong>${labels.map((name) => `<strong>${pretty(name)}</strong>`).join("")}</div>`;
  const rows = matrix.map((row, index) => `
    <div><strong>${pretty(labels[index])}</strong>${row.map((value) => `<span class="cell">${value}</span>`).join("")}</div>
  `).join("");
  host.innerHTML = `<div class="matrix">${header}${rows}</div>`;
}

function renderTiles(metrics, predictions) {
  const tiles = [
    ["Accuracy", metrics.accuracy],
    ["Loss", metrics.loss],
    ["Precision", metrics.precision],
    ["Recall", metrics.recall],
    ["F1", metrics.f1],
    ["AUC", metrics.auc],
  ];
  document.getElementById("metric-tiles").innerHTML = tiles.map(([label, value]) => `
    <article class="card tile"><span>${label}</span><strong>${value == null ? "—" : Number(value).toFixed(3)}</strong></article>
  `).join("") + `
    <article class="card tile"><span>Predictions served</span><strong>${predictions.total || 0}</strong></article>
    <article class="card tile"><span>Average latency</span><strong>${predictions.average_latency_ms || 0}<small> ms</small></strong></article>
  `;
}

async function loadMonitor() {
  const dashboard = await readJson("/api/dashboard");
  renderTiles(dashboard.metrics || {}, dashboard.predictions || {});
  renderMatrix(dashboard.metrics || {});
  const recent = document.getElementById("recent");
  const rows = (dashboard.predictions && dashboard.predictions.recent) || [];
  recent.innerHTML = rows.length
    ? rows.map((row) => `
      <div class="recent-item">
        <strong>${pretty(row.predicted_label)}</strong>
        <span class="meta"> ${percent(row.confidence)} · ${Math.round(row.latency_ms)} ms · ${row.filename || "upload"}</span>
      </div>
    `).join("")
    : "<p class='meta'>No predictions yet.</p>";
}

document.getElementById("refresh-eval").addEventListener("click", async () => {
  const message = document.getElementById("eval-message");
  message.textContent = "Scoring the held-out test set…";
  try {
    const metrics = await readJson("/api/evaluate?refresh=true");
    renderTiles(metrics, { total: "·", average_latency_ms: "·" });
    renderMatrix(metrics);
    message.textContent = `Scored ${metrics.support} test images. Accuracy ${Number(metrics.accuracy).toFixed(3)}.`;
  } catch (error) {
    message.textContent = error.message;
  }
});

refreshHealth();
setInterval(refreshHealth, 5000);
setInterval(tickUptime, 1000);
loadSamples().catch(() => {
  document.getElementById("samples").textContent = "Sample leaves will appear after the dataset is downloaded.";
});
loadMonitor().catch(() => {});
