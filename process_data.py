from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = ROOT / "data"
EXTRACT_DIR = ROOT / "dataset"
PROCESSED_DIR = ROOT / "processed"
IMAGE_SIZE = (224, 224)
LABELS = ["fire", "nofire", "smoke", "smokefire"]
SPLITS = ["train", "val", "test"]


def extract_dataset() -> Path:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found at {ARCHIVE_PATH}")

    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.iterdir()):
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
            archive.extractall(EXTRACT_DIR)

    dataset_root = EXTRACT_DIR / "Forect Fire" / "Forest Fire_Dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected extracted dataset folder at {dataset_root}")
    return dataset_root


def preprocess_image(image_path: Path, out_path: Path) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize(IMAGE_SIZE)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        processed = (array * 255.0).astype("uint8")
        Image.fromarray(processed, mode="RGB").save(out_path)


def build_processed_dataset() -> None:
    dataset_root = extract_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        for label in LABELS:
            src_dir = split_dir / label
            dst_dir = PROCESSED_DIR / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                continue

            for image_path in sorted(src_dir.glob("*.jpg")):
                out_path = dst_dir / image_path.name
                preprocess_image(image_path, out_path)
                records.append(
                    {
                        "split": split,
                        "label": label,
                        "image_path": str(out_path.relative_to(ROOT)),
                        "source": str(image_path.relative_to(ROOT)),
                    }
                )

    metadata = pd.DataFrame(records)
    metadata.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    print(f"Processed {len(metadata)} images into {PROCESSED_DIR}")


if __name__ == "__main__":
    build_processed_dataset()
