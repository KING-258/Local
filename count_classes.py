import os
from collections import defaultdict

# -------- USER CONFIG -------- #
LABEL_DIR = "/home/king/Downloads/CTS6U_augmented_dataset/CTS6U_dataset_augmented/labels/train"
# ----------------------------- #


def is_label_file(fname):
    return fname.endswith(".txt")


def main():

    if not os.path.exists(LABEL_DIR):
        print(f"❌ Folder not found: {LABEL_DIR}")
        return

    label_files = [f for f in os.listdir(LABEL_DIR) if is_label_file(f)]

    print(f"Found {len(label_files)} label files\n")

    class_counts = defaultdict(int)
    total_boxes = 0

    for fname in label_files:

        path = os.path.join(LABEL_DIR, fname)

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # YOLO format: class x y w h
            class_id = int(parts[0])

            class_counts[class_id] += 1
            total_boxes += 1

    if not class_counts:
        print("⚠️ No labels found in files.")
        return

    print("========== CLASS STATISTICS ==========\n")

    for cid in sorted(class_counts.keys()):
        print(f"Class {cid}: {class_counts[cid]} boxes")

    print("\n=====================================")

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Total unique classes: {len(class_counts)}")

    print(f"\nClass IDs found: {sorted(class_counts.keys())}")


if __name__ == "__main__":
    main()
