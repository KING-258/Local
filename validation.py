from ultralytics import YOLO

# -------- USER CONFIG -------- #
MODEL_PATH = "./defect-CTS6U.pt"
DATA_YAML = "/home/king/Downloads/CTS6U_augmented_dataset/CTS6U_dataset_augmented/data.yaml"
IMG_SIZE = 640
CONF = 0.25
IOU = 0.5
# ----------------------------- #


def main():

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Running validation...\n")

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        split="val",
        verbose=False
    )

    names = model.names
    nc = metrics.box.nc  # number of classes

    print("\n=========== CLASS-WISE METRICS ===========\n")

    for i in range(nc):

        # Returns: p, r, ap50, ap50-95
        p, r, ap50, ap5095 = metrics.box.class_result(i)

        name = names[i]

        print(f"Class {i} ({name})")
        print(f"  Precision   : {p:.4f}")
        print(f"  Recall      : {r:.4f}")
        print(f"  mAP@0.5     : {ap50:.4f}")
        print(f"  mAP@0.5:0.95: {ap5095:.4f}")
        print("-----------------------------------------")


    print("\n============= OVERALL METRICS =============\n")

    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
