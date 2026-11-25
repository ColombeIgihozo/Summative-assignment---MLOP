"""Write notebook/bean_leaf_pipeline.ipynb. Run from the project root."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str) -> dict:
    lines = source.strip("\n") + "\n"
    payload = {
        "cell_type": kind,
        "metadata": {},
        "source": lines.splitlines(keepends=True),
    }
    if kind == "code":
        payload["outputs"] = []
        payload["execution_count"] = None
    return payload


CELLS = [
    cell(
        "markdown",
        """
# BeanLeaf Doctor

End-to-end image classification for the African Leadership University machine-learning pipeline assignment.

The photographs come from the Makerere AI Lab iBean dataset, collected in Ugandan fields with the National Crops Resources Research Institute. Each image is one bean leaf and belongs to one of three classes:

- `angular_leaf_spot`
- `bean_rust`
- `healthy`

This notebook acquires the images, preprocesses them, trains a regularized transfer-learning model, and evaluates it with accuracy, loss, precision, recall, F1, and AUC. The saved network is the custom pretrained model that the API later fine-tunes.
""",
    ),
    cell(
        "code",
        """
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebook":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.acquire import dataset_ready, download_beans, extract_local_zips
from src.config import FEATURE_PATH, MODEL_PATH, PRETRAINED_PATH, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.model import build_model, train_model, tune_learning_rate
from src.prediction import predict_image
from src.preprocessing import (
    CLASS_NAMES,
    count_images,
    feature_stories,
    load_rgb,
    save_feature_summary,
    to_model_array,
)

np.random.seed(42)
print("Project root:", ROOT)
""",
    ),
    cell(
        "markdown",
        """
## 1. Data acquisition

The original Google Cloud zip links for iBean now return HTTP 403. This project uses the public Hugging Face mirror, `AI-Lab-Makerere/beans`, which contains the same 1,295 labelled field photos. Images are saved under `data/train`, `data/validation`, and `data/test`, with one folder per class.
""",
    ),
    cell(
        "code",
        """
if dataset_ready():
    download_report = "Dataset already on disk. Skipping the download."
elif extract_local_zips():
    download_report = "Extracted the Hugging Face iBean zip archives into data/."
else:
    download_report = download_beans()

counts = {
    "train": count_images(TRAIN_DIR),
    "validation": count_images(VAL_DIR),
    "test": count_images(TEST_DIR),
}
print(download_report)
counts
""",
    ),
    cell(
        "markdown",
        """
## 2. Data processing

Field photos are not the same size, and phone cameras store a rotation flag. Every image goes through the same steps before it reaches the network:

1. Apply the EXIF orientation and convert to RGB, so a greyscale or rotated phone photo cannot change the channel order.
2. Resize to 224 by 224 pixels. MobileNetV2 was pretrained at this size.
3. Scale each pixel with `x / 127.5 - 1`, which maps 0–255 onto the `[-1, 1]` range the pretrained backbone expects.
4. During training only, randomly flip, rotate, and zoom. The validation and test photos are never augmented.

The same function used here is the one the API uses at prediction time, so a leaf is prepared identically offline and in production.
""",
    ),
    cell(
        "code",
        """
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for row, class_name in enumerate(CLASS_NAMES):
    paths = sorted((TRAIN_DIR / class_name).glob("*.jpg"))[:4]
    for col, path in enumerate(paths):
        axes[row, col].imshow(load_rgb(path))
        axes[row, col].set_title(class_name.replace("_", " ") if col == 0 else "")
        axes[row, col].axis("off")
fig.suptitle("Training leaves, four examples from each class")
fig.tight_layout()
plt.show()
""",
    ),
    cell(
        "code",
        """
sample_path = next((TRAIN_DIR / "healthy").glob("*.jpg"))
original = load_rgb(sample_path)
resized = original.resize((224, 224))
pixels = np.asarray(resized, dtype="float32")
scaled = pixels / 127.5 - 1.0
model_ready = to_model_array(original)

print("Original size:", original.size)
print("Model input shape:", model_ready.shape)
print("Scaled min/max:", round(float(scaled.min()), 3), round(float(scaled.max()), 3))
print("Helper matches the manual scaling:", bool(np.allclose(scaled, model_ready)))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(original)
axes[0].set_title("Original upload")
axes[1].imshow((scaled + 1) / 2)
axes[1].set_title("224x224 preview of the model input")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
    ),
    cell(
        "markdown",
        """
## 3. What the photos are showing

A convolutional network is accurate, but it is a poor explanation by itself. These four measurements are computed from the pixels so a person can see why the classes differ:

- **Green dominance** is the share of the average colour that is green. Chlorophyll keeps a healthy leaf greener.
- **Rust index** is red divided by green. Bean rust adds reddish-brown pustules.
- **Lesion contrast** is the spread of brightness. Angular leaf spot punches dark patches into the blade.
- **Edge density** is the average change between neighbouring pixels. Spots and pustules add edges.

Brightness is calculated as well, so a difference that is only caused by a darker photograph can be spotted.
""",
    ),
    cell(
        "code",
        """
summary = save_feature_summary(TRAIN_DIR, FEATURE_PATH, per_class=100)
stories = feature_stories(summary)
feature_names = ["green_dominance", "rust_index", "lesion_contrast", "edge_density"]
colours = ["#b64024", "#c9842a", "#1f6b45"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for axis, feature_name in zip(axes.ravel(), feature_names):
    means = [summary["features"][feature_name][name]["mean"] for name in CLASS_NAMES]
    spread = [summary["features"][feature_name][name]["std"] for name in CLASS_NAMES]
    axis.bar(CLASS_NAMES, means, yerr=spread, color=colours, capsize=4)
    axis.set_title(feature_name.replace("_", " "))
    axis.tick_params(axis="x", labelrotation=15)
fig.suptitle("Mean feature value by class, with the standard deviation")
fig.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize=(7, 5))
for class_name, colour in zip(CLASS_NAMES, colours):
    xs = summary["features"]["green_dominance"][class_name]["values"]
    ys = summary["features"]["rust_index"][class_name]["values"]
    axis.scatter(xs, ys, s=18, alpha=0.75, label=class_name, color=colour)
axis.set_xlabel("Green dominance")
axis.set_ylabel("Rust index")
axis.set_title("Greener leaves sit to the right. Rustier leaves sit higher.")
axis.legend()
fig.tight_layout()
plt.show()

for story in stories:
    display(Markdown(f"**{story['title']}.** {story['text']}"))
""",
    ),
    cell(
        "markdown",
        """
## 4. Model creation

The classifier is a custom head on MobileNetV2 pretrained with ImageNet. The backbone stays frozen for the first training run, and Batch Normalisation stays in inference mode so the leaf photos do not overwrite the pretrained statistics.

Optimisation choices:

- **Pretrained model:** ImageNet MobileNetV2, with a new 128-unit dense layer and a 3-class softmax.
- **Regularisation:** dropout of 0.4 and 0.2, plus L2 on the dense kernel.
- **Optimiser:** Adam. Two learning rates are scored on a short run, and the lower validation loss is kept.
- **Early stopping:** stop after three epochs without a better validation loss and restore the best weights.
- **Learning-rate schedule:** reduce the learning rate by half when validation loss stalls.
- **Augmentation:** horizontal flip, small rotation, and zoom, applied only to training batches.
""",
    ),
    cell(
        "code",
        """
tuning = tune_learning_rate(TRAIN_DIR, VAL_DIR)
print(json.dumps(tuning, indent=2))
selected_lr = tuning["selected_learning_rate"]

preview = build_model(learning_rate=selected_lr)
preview.summary()
del preview
""",
    ),
    cell(
        "code",
        """
metrics = train_model(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    learning_rate=selected_lr,
    tune=False,
)
print("Epochs completed:", metrics["epochs_ran"])
print("Selected learning rate:", metrics["learning_rate"])
""",
    ),
    cell(
        "markdown",
        """
## 5. Model test

The test folder is never used for training or for the learning-rate search. Loss is the categorical cross-entropy of the predicted probabilities. Precision, recall, and F1 are macro averages, so each disease counts equally. AUC is the macro one-vs-rest score.
""",
    ),
    cell(
        "code",
        """
history = metrics["history"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["loss"], marker="o", label="train")
axes[0].plot(history["val_loss"], marker="o", label="validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["accuracy"], marker="o", label="train")
axes[1].plot(history["val_accuracy"], marker="o", label="validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.tight_layout()
plt.show()

score_names = ["accuracy", "loss", "precision", "recall", "f1", "auc"]
scores = {name: metrics[name] for name in score_names}
print(json.dumps(scores, indent=2))

fig, axis = plt.subplots(figsize=(6, 5))
sns.heatmap(
    np.array(metrics["confusion_matrix"]),
    annot=True,
    fmt="d",
    cmap="YlGn",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axis,
)
axis.set_xlabel("Predicted")
axis.set_ylabel("True class")
axis.set_title("Test-set confusion matrix")
fig.tight_layout()
plt.show()

report = metrics["classification_report"]
for class_name in CLASS_NAMES:
    row = report[class_name]
    print(
        f"{class_name:20} precision {row['precision']:.3f}  "
        f"recall {row['recall']:.3f}  f1 {row['f1-score']:.3f}  support {int(row['support'])}"
    )
""",
    ),
    cell(
        "markdown",
        """
## 6. Prediction function

`predict_image` is the function the API calls. It loads the saved `.keras` file, preprocesses one photograph, and returns the class, the three probabilities, and the colour/texture features. The six photos below are taken from the test folders, so the folder name is the true label.
""",
    ),
    cell(
        "code",
        """
picks = []
for class_name in CLASS_NAMES:
    picks.extend(sorted((TEST_DIR / class_name).glob("*.jpg"))[:2])

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
correct = 0
for axis, path in zip(axes.ravel(), picks):
    truth = path.parent.name
    result = predict_image(path, MODEL_PATH)
    matched = result["label"] == truth
    correct += int(matched)
    axis.imshow(load_rgb(path))
    axis.set_title(f"true: {truth}\\npred: {result['label']} ({result['confidence']:.0%})")
    axis.axis("off")
fig.suptitle(f"Single-image predictions: {correct} of {len(picks)} matched the folder label")
fig.tight_layout()
plt.show()
print(f"{correct}/{len(picks)} correct")
""",
    ),
    cell(
        "markdown",
        """
## 7. Retraining from this custom model

The first training run writes two copies:

- `models/bean_leaf_model.keras` is the active model. Retraining updates it.
- `models/bean_leaf_pretrained.keras` stays as the original custom model.
- `models/bean_leaf_model.h5` is the same active network in HDF5 format.

The API saves newly uploaded leaves, records them in SQLite, and preprocesses them with the same RGB, resize, and scaling steps. A button starts fine-tuning. A second trigger starts it automatically once 15 new labelled images are waiting. Fine-tuning loads the active custom model, unfreezes the top of the backbone at a learning rate of `1e-4`, and mixes the new leaves with a replay sample so the earlier classes are not forgotten. The held-out test set is scored again after every retrain.
""",
    ),
    cell(
        "code",
        """
print("Active model:", MODEL_PATH, MODEL_PATH.exists())
print("Custom pretrained copy:", PRETRAINED_PATH, PRETRAINED_PATH.exists())
print("Test images held out:", count_images(TEST_DIR))
print(json.dumps(metrics["optimization"], indent=2))
""",
    ),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11 (beanleaf)",
                "language": "python",
                "name": "beanleaf",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }
    destination = ROOT / "notebook" / "bean_leaf_pipeline.ipynb"
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
