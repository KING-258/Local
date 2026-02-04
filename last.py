"""
End-to-end YOLOv8 multi-class defect detection pipeline

Input:
- data/
    IMG_xxx.jpg
    IMG_xxx.txt   (YOLO format)

Output:
- det_dataset/
    images/train
    images/val
    labels/train
    labels/val
    data.yaml
- YOLO trained model (.pt)
"""

import os
import shutil
import random
import yaml
from collections import Counter
from ultralytics import YOLO

# ---------------- USER SETTINGS ---------------- #
SOURCE_DIR = "/home/king/Downloads/CTS6U_augmented_dataset/data"            # your current folder
DATASET_DIR = "./det_dataset"    # YOLO dataset output
TRAIN_SPLIT = 0.8

MODEL = "yolov8n.pt"
IMG_SIZE = 640
EPOCHS = 150
BATCH = 8
LR0 = 0.001
WEIGHT_DECAY = 0.0005
PATIENCE = 30
# ----------------------------------------------- #


def collect_pairs(src):
    pairs = []
    for f in os.listdir(src):
        if f.lower().endswith(".jpg"):
            txt = f.replace(".jpg", ".txt")
            if os.path.exists(os.path.join(src, txt)):
                pairs.append((f, txt))
    return pairs


def scan_classes(label_files):
    counter = Counter()
    for lbl in label_files:
        with open(lbl) as f:
            for line in f:
                if line.strip():
                    counter[int(line.split()[0])] += 1
    return counter


def make_dirs():
    for p in [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]:
        os.makedirs(os.path.join(DATASET_DIR, p), exist_ok=True)


def main():
    print("\n📦 Preparing YOLO dataset...")

    if not os.path.exists(SOURCE_DIR):
        raise RuntimeError("❌ 'data/' folder not found")

    pairs = collect_pairs(SOURCE_DIR)
    if len(pairs) == 0:
        raise RuntimeError("❌ No image–label pairs found")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    make_dirs()

    # Copy files
    for split, items in [("train", train_pairs), ("val", val_pairs)]:
        for img, lbl in items:
            shutil.copy(
                os.path.join(SOURCE_DIR, img),
                os.path.join(DATASET_DIR, "images", split, img)
            )
            shutil.copy(
                os.path.join(SOURCE_DIR, lbl),
                os.path.join(DATASET_DIR, "labels", split, lbl)
            )

    # Scan classes
    label_paths = [
        os.path.join(DATASET_DIR, "labels", "train", x[1])
        for x in train_pairs
    ]
    class_counts = scan_classes(label_paths)
    num_classes = max(class_counts.keys()) + 1

    print("\n📊 Detected defect classes:")
    for k, v in sorted(class_counts.items()):
        print(f"  Class {k}: {v} boxes")
    print(f"\n✅ Total classes: {num_classes}")

    # Create data.yaml
    data_yaml = {
        "path": DATASET_DIR,
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": [f"defect_{i}" for i in range(num_classes)]
    }

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print("\n🧠 data.yaml created")

    # ---------------- TRAIN YOLO ---------------- #
    print("\n🚀 Starting YOLO training...\n")

    model = YOLO(MODEL)
    model.train(
        data=os.path.join(DATASET_DIR, "data.yaml"),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE
    )

    print("\n✅ Training complete")
    print("📁 Model saved in: runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
