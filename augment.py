"""
YOLO-safe augmentation for defect detection
Augments training images + bounding boxes together
"""

import os
import cv2
import random
import albumentations as A
from pathlib import Path

# ---------------- CONFIG ---------------- #
DATASET_ROOT = "STH3_YOLO"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images/train")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels/train")

AUG_PER_IMAGE = 6   # 5–8 is safe for inspection
# ---------------------------------------- #

augment = A.Compose(
    [
        A.Rotate(limit=7, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.04,
            scale_limit=0.05,
            rotate_limit=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=(5, 20), p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)


def read_yolo_label(label_path):
    boxes = []
    classes = []
    with open(label_path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.split())
            boxes.append([x, y, w, h])
            classes.append(int(c))
    return boxes, classes


def write_yolo_label(path, boxes, classes):
    with open(path, "w") as f:
        for c, b in zip(classes, boxes):
            f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")


def main():
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]

    print(f"Found {len(images)} training images")
    print(f"Generating {AUG_PER_IMAGE} augmentations per image")

    aug_id = 0

    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        lbl_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        boxes, classes = read_yolo_label(lbl_path)

        for i in range(AUG_PER_IMAGE):
            augmented = augment(
                image=image,
                bboxes=boxes,
                class_labels=classes
            )

            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            if len(aug_boxes) == 0:
                continue

            new_img_name = f"aug_{aug_id}_{img_name}"
            new_lbl_name = new_img_name.replace(".jpg", ".txt")

            cv2.imwrite(os.path.join(IMAGES_DIR, new_img_name), aug_img)
            write_yolo_label(
                os.path.join(LABELS_DIR, new_lbl_name),
                aug_boxes,
                aug_classes
            )

            aug_id += 1

    print(f"Augmentation complete. Generated {aug_id} new images.")


if __name__ == "__main__":
    main()
