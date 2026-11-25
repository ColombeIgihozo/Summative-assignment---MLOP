"""Load, clean, and describe bean-leaf images.

Training images are field photographs from the Makerere iBean dataset.
Every image is converted to RGB, resized to 224x224, and scaled to the
MobileNetV2 range [-1, 1]. Separate colour and texture features are
extracted so the UI can explain the dataset without running the network.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_ALIASES = {
    "angular": "angular_leaf_spot",
    "angular_leaf_spot": "angular_leaf_spot",
    "angularleafspot": "angular_leaf_spot",
    "als": "angular_leaf_spot",
    "leaf_spot": "angular_leaf_spot",
    "rust": "bean_rust",
    "bean_rust": "bean_rust",
    "beanrust": "bean_rust",
    "healthy": "healthy",
    "health": "healthy",
    "normal": "healthy",
}

FEATURE_NAMES = (
    "green_dominance",
    "rust_index",
    "lesion_contrast",
    "edge_density",
    "brightness",
)


def normalize_label(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in CLASS_ALIASES:
        raise ValueError(
            f"Unknown class '{name}'. Use one of: {', '.join(CLASS_NAMES)}."
        )
    return CLASS_ALIASES[key]


def class_index(name: str) -> int:
    return CLASS_NAMES.index(normalize_label(name))


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def load_rgb(source) -> Image.Image:
    """Open a path, raw bytes, or PIL image and return an upright RGB image."""
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def to_model_array(image: Image.Image) -> np.ndarray:
    """Resize and scale one image to the MobileNetV2 input range [-1, 1]."""
    array = np.asarray(resize_image(image), dtype=np.float32)
    return array / 127.5 - 1.0


def batch_from_images(images: list[Image.Image]) -> np.ndarray:
    if not images:
        raise ValueError("No images were provided.")
    return np.stack([to_model_array(image) for image in images], axis=0)


def extract_features(image: Image.Image) -> dict[str, float]:
    """Summarise colour and texture so each class can be compared visually.

    green_dominance
        Share of the average colour that is green. Healthy leaves keep more
        chlorophyll, so this value stays higher.
    rust_index
        Red relative to green. Bean rust forms reddish-brown pustules, which
        lifts this ratio.
    lesion_contrast
        Spread of brightness inside the leaf. Angular leaf spot punches dark
        lesions into green tissue, so contrast rises.
    edge_density
        Average horizontal and vertical brightness change. Spots and pustules
        add edges that a smooth healthy leaf does not have.
    brightness
        Overall exposure. Included so we can check that class differences are
        not just an artefact of darker photographs.
    """
    sample = np.asarray(resize_image(image, (128, 128)), dtype=np.float32)
    red, green, blue = sample[:, :, 0], sample[:, :, 1], sample[:, :, 2]
    total = float(red.mean() + green.mean() + blue.mean()) + 1e-6
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gradient_y = np.abs(np.diff(gray, axis=0)).mean()
    gradient_x = np.abs(np.diff(gray, axis=1)).mean()
    return {
        "green_dominance": float(green.mean() / total),
        "rust_index": float(red.mean() / (green.mean() + 1e-6)),
        "lesion_contrast": float(gray.std() / 255.0),
        "edge_density": float((gradient_x + gradient_y) / 255.0),
        "brightness": float(gray.mean() / 255.0),
    }


def iter_labeled_images(root: Path):
    """Yield (class_name, path) for images stored in class folders."""
    root = Path(root)
    if not root.exists():
        return
    for class_name in CLASS_NAMES:
        folder = root / class_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and is_image_file(path):
                yield class_name, path


def summarize_features(root: Path, per_class: int = 120, seed: int = 7) -> dict:
    """Aggregate interpretable features for charts and the written story."""
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[dict[str, float]]] = {name: [] for name in CLASS_NAMES}
    for class_name, path in iter_labeled_images(root):
        grouped[class_name].append(extract_features(load_rgb(path)))

    summary = {"classes": CLASS_NAMES, "features": {}, "counts": {}}
    for class_name, rows in grouped.items():
        summary["counts"][class_name] = len(rows)
        if len(rows) > per_class:
            picked = rng.choice(len(rows), size=per_class, replace=False)
            rows = [rows[int(i)] for i in picked]
        for feature in FEATURE_NAMES:
            values = [row[feature] for row in rows]
            summary["features"].setdefault(feature, {})[class_name] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "values": [round(float(value), 4) for value in values],
            }
    return summary


def feature_stories(summary: dict) -> list[dict]:
    """Turn feature averages into short, class-by-class interpretations."""
    pretty = {
        "angular_leaf_spot": "Angular leaf spot",
        "bean_rust": "Bean rust",
        "healthy": "Healthy",
    }

    def means(feature: str) -> dict[str, float]:
        return {
            name: float(summary["features"][feature][name]["mean"])
            for name in CLASS_NAMES
        }

    def winner(feature: str) -> str:
        values = means(feature)
        return max(values, key=values.get)

    green = means("green_dominance")
    rust = means("rust_index")
    edges = means("edge_density")
    bright = means("brightness")
    darkest = min(bright, key=bright.get)
    brightest = max(bright, key=bright.get)
    bright_gap = bright[brightest] - bright[darkest]
    return [
        {
            "feature": "green_dominance",
            "title": "Green share of the leaf",
            "means": green,
            "text": (
                f"{pretty[winner('green_dominance')]} has the strongest green share "
                f"({green[winner('green_dominance')]:.3f}). Chlorophyll reflects green light, "
                "so leaves that are still intact score higher. Rust and angular lesions replace "
                "some of that tissue and pull the green share down."
            ),
        },
        {
            "feature": "rust_index",
            "title": "Red relative to green",
            "means": rust,
            "text": (
                f"{pretty[winner('rust_index')]} scores the highest rust index "
                f"({rust[winner('rust_index')]:.3f}), while healthy leaves sit much lower "
                f"({rust['healthy']:.3f}). Bean rust grows reddish-brown pustules, which add red "
                "without adding green. This is the clearest split in the field photos."
            ),
        },
        {
            "feature": "brightness",
            "title": "Exposure is not the diagnosis",
            "means": bright,
            "text": (
                f"{pretty[darkest]} photographs are the darkest ({bright[darkest]:.3f}) and "
                f"{pretty[brightest]} are the brightest ({bright[brightest]:.3f}), a gap of "
                f"{bright_gap:.3f}. Rust pustules and field shade can darken a frame. A dim "
                "healthy leaf is still greener, so brightness is a warning about the camera, "
                "not a class by itself."
            ),
        },
        {
            "feature": "edge_density",
            "title": "Edges from pustules",
            "means": edges,
            "text": (
                f"{pretty[winner('edge_density')]} has the most edge density "
                f"({edges[winner('edge_density')]:.3f}). The gap between classes is small next "
                "to the colour ratios. Texture supports the rust-index story. It does not replace it."
            ),
        },
    ]


def save_feature_summary(root: Path, destination: Path, per_class: int = 120) -> dict:
    summary = summarize_features(root, per_class=per_class)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary), encoding="utf-8")
    return summary


def count_images(root: Path) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for class_name, _path in iter_labeled_images(root):
        counts[class_name] += 1
    counts["total"] = sum(counts[name] for name in CLASS_NAMES)
    return counts
