"""Download the Makerere iBean leaf photographs into data/train, data/validation, and data/test."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.config import DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.preprocessing import CLASS_NAMES, count_images

DATASET_ID = "AI-Lab-Makerere/beans"
SPLIT_DIRS = {
    "train": TRAIN_DIR,
    "validation": VAL_DIR,
    "test": TEST_DIR,
}


def dataset_ready(min_train: int = 900, min_test: int = 100) -> bool:
    return count_images(TRAIN_DIR)["total"] >= min_train and count_images(TEST_DIR)["total"] >= min_test


def extract_local_zips(destination: Path = DATA_DIR) -> bool:
    """Unpack data/train.zip, data/validation.zip, and data/test.zip when they exist.

    The archives come from the Hugging Face copy of iBean and already contain
    the class folders. Extracting them avoids a second download.
    """
    extracted = False
    for split in SPLIT_DIRS:
        archive_path = destination / f"{split}.zip"
        if not archive_path.exists():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        extracted = True
    return extracted


def download_beans(destination: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """Save the Hugging Face mirror of the iBean dataset as class folders."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID)
    counts = {}
    for split, folder in SPLIT_DIRS.items():
        label_feature = dataset[split].features["labels"]
        saved = 0
        for index, row in enumerate(dataset[split]):
            label = label_feature.int2str(int(row["labels"]))
            if label not in CLASS_NAMES:
                raise RuntimeError(f"Unexpected label '{label}' in split {split}.")
            class_dir = folder / label
            class_dir.mkdir(parents=True, exist_ok=True)
            path = class_dir / f"{label}_{index:04d}.jpg"
            if not path.exists():
                row["image"].convert("RGB").save(path, format="JPEG", quality=90)
            saved += 1
        counts[split] = count_images(folder)
        counts[split]["saved"] = saved
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the iBean leaf images.")
    parser.add_argument("--destination", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if dataset_ready():
        print("Dataset already present.")
        print(count_images(TRAIN_DIR))
        return
    if extract_local_zips(args.destination) and dataset_ready():
        print("Extracted local zip archives.")
        print(count_images(TRAIN_DIR))
        return
    print(f"Downloading {DATASET_ID} into {args.destination} ...")
    print(download_beans(args.destination))


if __name__ == "__main__":
    main()
