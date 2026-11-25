"""Build, train, evaluate, and retrain the bean-leaf classifier.

The network is a custom head on ImageNet MobileNetV2. The saved network is
the pretrained model used later: retraining loads that file and keeps
teaching it with newly uploaded leaves plus a replay sample of the
original field photos.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from src.config import (
    ACCEPTED_DIR,
    BATCH_SIZE,
    CLASS_PATH,
    FEATURE_PATH,
    H5_PATH,
    INCOMING_DIR,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PRETRAINED_PATH,
    RETRAIN_EPOCHS,
    TEST_DIR,
    TRAIN_DIR,
    TRAIN_EPOCHS,
    VAL_DIR,
)
from src.preprocessing import CLASS_NAMES, IMG_SIZE, iter_labeled_images, save_feature_summary

RANDOM_SEED = 42


def _tf():
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)
    return tf


def _metrics(tf):
    return [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_model(dropout: float = 0.4, l2_penalty: float = 1e-4, learning_rate: float = 1e-3):
    """Frozen MobileNetV2 backbone plus a regularized classification head."""
    tf = _tf()
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="leaf_image")
    # training=False keeps BatchNorm on the ImageNet statistics.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pooled_features")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_pool")(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_penalty),
        name="disease_embedding",
    )(x)
    x = tf.keras.layers.Dropout(dropout / 2, name="dropout_embedding")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="class_probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="bean_leaf_mobilenet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=_metrics(tf),
    )
    return model


def _prepare_batch(image, label, tf):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(directory: Path, batch_size: int = BATCH_SIZE, shuffle: bool = True, augment: bool = False):
    """Load a class-folder directory and apply the same preprocessing as prediction."""
    tf = _tf()
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )
    dataset = dataset.map(
        lambda image, label: _prepare_batch(image, label, tf),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.08),
            ],
            name="train_augmentation",
        )

        def _augment(image, label):
            return augmenter(image, training=True), label

        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def dataset_from_paths(paths: list[Path], labels: list[int], batch_size: int, shuffle: bool, augment: bool):
    """Build a dataset from uploaded files that may not contain every class."""
    tf = _tf()
    pairs = list(zip([str(path) for path in paths], labels))
    if shuffle:
        random.Random(RANDOM_SEED).shuffle(pairs)
    path_list, label_list = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(path_list), list(label_list)))

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, tf.one_hot(label, len(CLASS_NAMES))

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.08),
            ]
        )

        def _augment(image, label):
            return augmenter(image, training=True), label

        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _callbacks(tf):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
    ]


def evaluate_model(model, dataset) -> dict:
    """Accuracy, loss, precision, recall, F1, AUC, and the confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_labels = []
    probabilities = []
    for images, labels in dataset:
        probabilities.append(model.predict(images, verbose=0))
        true_labels.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(true_labels)
    y_prob = np.concatenate(probabilities)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, y_prob, labels=list(range(len(CLASS_NAMES))))),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": auc,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "support": int(len(y_true)),
    }
    return _jsonable(payload)


def _history_payload(history) -> dict:
    payload = {}
    for key, values in history.history.items():
        payload[key] = [float(value) for value in values]
    return payload


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model_files(model, metrics: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    # The first trained network is kept as the custom pretrained model.
    # Later retraining updates the active file and leaves this copy in place.
    if not PRETRAINED_PATH.exists():
        model.save(PRETRAINED_PATH)
    try:
        model.save(H5_PATH)
    except Exception as exc:  # pragma: no cover - format differences across Keras builds
        metrics["h5_warning"] = str(exc)
    CLASS_PATH.write_text(json.dumps(list(CLASS_NAMES)), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")


def tune_learning_rate(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    candidates: tuple[float, ...] = (1e-3, 5e-4),
    train_batches: int = 6,
    val_batches: int = 3,
) -> dict:
    """Compare a few learning rates on a short run and keep the best validation loss."""
    tf = _tf()
    results = []
    for learning_rate in candidates:
        tf.keras.backend.clear_session()
        model = build_model(learning_rate=learning_rate)
        train_ds = make_dataset(train_dir, shuffle=True, augment=True).take(train_batches)
        val_ds = make_dataset(val_dir, shuffle=False, augment=False).take(val_batches)
        history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2, shuffle=False)
        results.append(
            {
                "learning_rate": learning_rate,
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1]),
            }
        )
    best = min(results, key=lambda row: row["val_loss"])
    tf.keras.backend.clear_session()
    return {"candidates": results, "selected_learning_rate": best["learning_rate"]}


def train_model(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float | None = None,
    tune: bool = True,
) -> dict:
    """Train the custom model from the ImageNet backbone and score it on the test set."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    tuning = None
    if learning_rate is None and tune:
        tuning = tune_learning_rate(train_dir, val_dir)
        learning_rate = tuning["selected_learning_rate"]
    learning_rate = 1e-3 if learning_rate is None else learning_rate

    tf = _tf()
    tf.keras.backend.clear_session()
    model = build_model(learning_rate=learning_rate)
    train_ds = make_dataset(train_dir, shuffle=True, augment=True)
    val_ds = make_dataset(val_dir, shuffle=False, augment=False)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_callbacks(tf),
        verbose=2,
        shuffle=False,
    )
    test_ds = make_dataset(test_dir, shuffle=False, augment=False)
    metrics = evaluate_model(model, test_ds)
    metrics["history"] = _history_payload(history)
    metrics["learning_rate"] = learning_rate
    metrics["tuning"] = tuning
    metrics["epochs_ran"] = len(history.history["loss"])
    metrics["optimization"] = {
        "backbone": "MobileNetV2 ImageNet weights, frozen",
        "optimizer": "Adam",
        "regularization": "Dropout 0.4 / 0.2 and L2 on the dense layer",
        "early_stopping": "patience 3, restore best weights",
        "lr_schedule": "ReduceLROnPlateau",
        "augmentation": "horizontal flip, rotation, zoom",
    }
    save_model_files(model, metrics)
    save_feature_summary(train_dir, FEATURE_PATH)
    return metrics


def _collect(root: Path) -> list[tuple[Path, int]]:
    return [(path, CLASS_NAMES.index(label)) for label, path in iter_labeled_images(root)]


def _replay_sample(original: list[tuple[Path, int]], accepted: list[tuple[Path, int]], target: int) -> list[tuple[Path, int]]:
    pool = list(original) + list(accepted)
    if not pool or target <= 0:
        return []
    rng = random.Random(RANDOM_SEED)
    if len(pool) <= target:
        return pool
    # Keep the sample close to the class balance of the pool.
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for item in pool:
        by_class.setdefault(item[1], []).append(item)
    picked: list[tuple[Path, int]] = []
    per_class = max(1, target // max(1, len(by_class)))
    for rows in by_class.values():
        rng.shuffle(rows)
        picked.extend(rows[:per_class])
    rng.shuffle(picked)
    return picked[:target]


def prepare_for_finetune(model, learning_rate: float = 1e-4, unfreeze_layers: int = 30):
    """Unfreeze the top of the backbone and recompile at a smaller learning rate."""
    tf = _tf()
    base = None
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            base = layer
            break
    if base is None:
        raise RuntimeError("The saved model does not contain the MobileNetV2 backbone.")
    base.trainable = True
    cutoff = max(0, len(base.layers) - unfreeze_layers)
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=_metrics(tf),
    )
    return model


def retrain_model(
    incoming_dir: Path = INCOMING_DIR,
    accepted_dir: Path = ACCEPTED_DIR,
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    model_path: Path = MODEL_PATH,
    epochs: int = RETRAIN_EPOCHS,
    run_id: int | None = None,
) -> dict:
    """Fine-tune the already trained custom model on newly uploaded leaves."""
    from src.database import finish_run, update_run

    incoming = _collect(incoming_dir)
    if not incoming:
        raise RuntimeError("No uploaded images are waiting. Upload labelled leaves first.")
    if not Path(model_path).exists():
        raise RuntimeError(f"Pretrained custom model was not found at {model_path}.")

    accepted = _collect(accepted_dir)
    original = _collect(train_dir)
    replay = _replay_sample(original, accepted, target=max(24, len(incoming) * 2))
    combined = incoming + replay
    paths = [path for path, _label in combined]
    labels = [label for _path, label in combined]

    if run_id is not None:
        update_run(
            run_id,
            message=(
                f"Preprocessed {len(incoming)} uploaded images "
                f"(RGB, 224x224, MobileNet scaling) and mixed them with "
                f"{len(replay)} replay images from the original model."
            ),
            sample_count=len(incoming),
        )

    tf = _tf()
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path)
    model = prepare_for_finetune(model, learning_rate=1e-4)
    batch_size = 8 if len(combined) < 32 else BATCH_SIZE
    train_ds = dataset_from_paths(paths, labels, batch_size=batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(val_dir, batch_size=batch_size, shuffle=False, augment=False)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_callbacks(tf),
        verbose=2,
        shuffle=False,
    )
    test_ds = make_dataset(test_dir, batch_size=batch_size, shuffle=False, augment=False)
    metrics = evaluate_model(model, test_ds)
    metrics["history"] = _history_payload(history)
    metrics["learning_rate"] = 1e-4
    metrics["uploaded_images"] = len(incoming)
    metrics["replay_images"] = len(replay)
    metrics["trigger_note"] = "Fine-tuned the saved custom model rather than training from scratch."
    save_model_files(model, metrics)

    from src.database import init_db, mark_accepted

    init_db()
    original_paths = [str(path.resolve()) for path, _label in incoming]
    accepted_dir.mkdir(parents=True, exist_ok=True)
    for path, label in incoming:
        destination_dir = accepted_dir / CLASS_NAMES[label]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / path.name
        if path.exists():
            shutil.move(str(path), str(destination))
    mark_accepted(original_paths)

    if run_id is not None:
        finish_run(run_id, "completed", metrics, str(MODEL_PATH))
    return metrics


def _update_failed(run_id: int | None, message: str) -> None:
    if run_id is None:
        return
    from src.database import finish_run

    finish_run(run_id, "failed", {"error": message}, None, message=message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or retrain the bean-leaf model.")
    parser.add_argument("command", choices=["train", "retrain", "evaluate", "features"])
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "train":
            metrics = train_model(
                epochs=args.epochs or TRAIN_EPOCHS,
                tune=not args.no_tune,
            )
            print(json.dumps({key: metrics[key] for key in ("accuracy", "loss", "precision", "recall", "f1", "auc")}))
        elif args.command == "retrain":
            metrics = retrain_model(epochs=args.epochs or RETRAIN_EPOCHS, run_id=args.run_id)
            print(json.dumps({key: metrics[key] for key in ("accuracy", "f1", "uploaded_images")}))
        elif args.command == "evaluate":
            tf = _tf()
            model = tf.keras.models.load_model(MODEL_PATH)
            dataset = make_dataset(TEST_DIR, shuffle=False, augment=False)
            metrics = evaluate_model(model, dataset)
            print(json.dumps({key: metrics[key] for key in ("accuracy", "loss", "precision", "recall", "f1", "auc")}))
        elif args.command == "features":
            summary = save_feature_summary(TRAIN_DIR, FEATURE_PATH)
            print(json.dumps(summary["counts"]))
    except Exception as exc:
        _update_failed(args.run_id, str(exc))
        raise


if __name__ == "__main__":
    main()
