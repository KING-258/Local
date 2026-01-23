#!/usr/bin/env python3
from pathlib import Path
import shutil

from ultralytics import YOLO
import torch

PART_IMAGES_DIR = Path("/home/king/Downloads/Training/Positives")  # TODO
NEG_IMAGES_DIR  = Path("/home/king/Downloads/Training/negatives")  # TODO

DATASET_ROOT    = PART_IMAGES_DIR.parent / "part_dataset"
IMAGES_TRAIN_DIR = DATASET_ROOT / "images" / "train"
LABELS_TRAIN_DIR = DATASET_ROOT / "labels" / "train"
DATA_YAML_PATH   = DATASET_ROOT / "data_part.yaml"

MODEL_NAME = "yolov8n.pt"

EPOCHS   = 50
BATCH    = 16
IMGSZ    = 640
WORKERS  = 4

PROJECT  = "runs/part_train"
RUN_NAME = "part_detector"
EXIST_OK = True

PART_CLASS_ID = 0
# ------------------------------------------------------------- #


def gather_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def build_dataset():
    # Clean + recreate dataset root
    if DATASET_ROOT.exists():
        print(f"[INFO] Removing existing dataset root: {DATASET_ROOT}")
        shutil.rmtree(DATASET_ROOT)
    IMAGES_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    pos_paths = gather_images(PART_IMAGES_DIR)
    neg_paths = gather_images(NEG_IMAGES_DIR)

    if not pos_paths:
        raise RuntimeError(f"No positive images found in {PART_IMAGES_DIR}")
    if not neg_paths:
        print(f"[WARN] No negatives found in {NEG_IMAGES_DIR} – model may over-fire.")

    print(f"[INFO] Positives: {len(pos_paths)}, negatives: {len(neg_paths)}")

    # 1) Positives: copy image + big 'part' box label
    for img_path in pos_paths:
        dst_img = IMAGES_TRAIN_DIR / f"pos_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        label_path = LABELS_TRAIN_DIR / (dst_img.stem + ".txt")

        # Big box around center (normalized)
        x_center = 0.5
        y_center = 0.5
        width = 0.9
        height = 0.9
        line = f"{PART_CLASS_ID} {x_center} {y_center} {width} {height}\n"

        with open(label_path, "w") as f:
            f.write(line)

    # 2) Negatives: copy image + EMPTY label file (background)
    for img_path in neg_paths:
        dst_img = IMAGES_TRAIN_DIR / f"neg_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        label_path = LABELS_TRAIN_DIR / (dst_img.stem + ".txt")
        # Empty file -> YOLO treats this as "no objects in this image"
        open(label_path, "w").close()

    print(f"[INFO] Images copied to: {IMAGES_TRAIN_DIR}")
    print(f"[INFO] Labels written to: {LABELS_TRAIN_DIR}")

    # 3) Write data_part.yaml
    data_yaml_text = f"""path: {DATASET_ROOT}
train: images/train
val: images/train

names:
  - part
"""
    with open(DATA_YAML_PATH, "w") as f:
        f.write(data_yaml_text)

    print(f"[INFO] Wrote data yaml to: {DATA_YAML_PATH}")


def train():
    device = "0" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device)

    model = YOLO(MODEL_NAME)
    print("[INFO] Starting training...")

    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=WORKERS,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=EXIST_OK,
        device=device,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Model weights: {PROJECT}/{RUN_NAME}/weights/best.pt")
    return results


def main():
    build_dataset()
    train()


if __name__ == "__main__":
    main()