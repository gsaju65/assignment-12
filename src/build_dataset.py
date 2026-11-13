"""Crop every labeled object out of the VisDrone images and turn each crop into a feature vector.

An object is kept if it belongs to one of the classes below, is at least 20 px on each side, and is not heavily occluded. VisDrone images come in sequences
from the same flight that look alike, so at most 40 objects of each class are kept per sequence, the same idea as the 4-per-flight cap in Project 2.

Classes (VisDrone category in brackets):
    person       pedestrian (1), people (2)
    car_van      car (4), van (5)
    truck_bus    truck (6), bus (9)
    two_wheeler  bicycle (3), motor (10)
Tricycles (7, 8), ignored regions (0) and "others" (11) are left out.

Features (902 per crop):
    HOG (histogram of oriented gradients) of the crop resized to 48x48 grayscale: 900 numbers that describe its edges and shape
    log aspect ratio of the box: people are tall and thin, buses are long
    log of box area / image area: how big the object looks from the drone

Output: data/processed/dataset.npz with the features (X), class (y) and flight sequence (groups) of every crop.

Usage:
    python src/build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "VisDrone2019-DET-test-dev"
PROCESSED = ROOT / "data" / "processed"

COLUMNS = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]
CLASSES = {1: "person", 2: "person", 4: "car_van", 5: "car_van",
           6: "truck_bus", 9: "truck_bus", 3: "two_wheeler", 10: "two_wheeler"}
SEED = 42
MIN_SIDE_PX = 20       # smaller boxes have too few pixels to classify
MAX_OCCLUSION = 1      # VisDrone occlusion: 0 none, 1 partial, 2 heavy
MAX_PER_SEQUENCE = 40  # per class, so a few long flights full of parked cars can't dominate
CROP_PX = 48


def load_labels(label_path):
    """Return the objects in one image that are in a class and pass the size and occlusion filters."""
    labels = pd.read_csv(label_path, header=None, names=COLUMNS)
    keep = (labels["category"].isin(list(CLASSES)) & (labels["score"] == 1)
            & (labels[["w", "h"]].min(axis=1) >= MIN_SIDE_PX) & (labels["occlusion"] <= MAX_OCCLUSION))
    return labels[keep].assign(image_id=label_path.stem)


def features(crop, w, h, image_area):
    """HOG of the 48x48 grayscale crop, plus the box's log aspect ratio and log relative size."""
    pixels = np.asarray(crop.convert("L").resize((CROP_PX, CROP_PX)), dtype=np.float32) / 255
    shape = hog(pixels, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return np.append(shape, [np.log(w / h), np.log(w * h / image_area)])


def main():
    # 1. List every usable object
    label_paths = sorted((SOURCE / "annotations").glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"No labels found in {SOURCE}. Run src/download_data.py first.")
    objects = pd.concat([load_labels(path) for path in label_paths], ignore_index=True)
    objects["sequence"] = objects["image_id"].str.split("_").str[0]
    objects["label"] = objects["category"].map(CLASSES)

    # 2. Shuffle with a fixed seed, then keep the first 40 objects of each class in each sequence
    sample = objects.sample(frac=1, random_state=SEED).groupby(["sequence", "label"]).head(MAX_PER_SEQUENCE)

    # 3. Crop each object and compute its features
    X, y, groups = [], [], []
    for image_id, boxes in sample.groupby("image_id"):
        with Image.open(SOURCE / "images" / f"{image_id}.jpg") as im:
            image_area = im.width * im.height
            for box in boxes.itertuples():
                crop = im.crop((box.x, box.y, box.x + box.w, box.y + box.h))
                X.append(features(crop, box.w, box.h, image_area))
                y.append(box.label)
                groups.append(box.sequence)
    assert len(X) == len(sample)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "dataset.npz"
    np.savez_compressed(out, X=np.array(X, dtype=np.float32), y=np.array(y), groups=np.array(groups))

    print(f"Usable objects: {len(objects)} from {objects['sequence'].nunique()} sequences")
    print(f"Saved {len(X)} crops x {len(X[0])} features to {out.name}")
    print("Crops per class:")
    print(sample["label"].value_counts().to_string())


if __name__ == "__main__":
    main()