import os
import cv2
from ultralytics import YOLO

# ---------------- USER CONFIG ---------------- #
MODEL_PATH = "./best.pt"                 # path to your .pt model
INPUT_DIR = "/home/king/check"           # folder with images to check
OUTPUT_DIR = "./output"                  # results folder

CONF_THRESHOLD = 0.5                   # defect confidence threshold
SAVE_ANNOTATED = True                  # save images with boxes drawn
# --------------------------------------------- #


def is_image(fname):
    return fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))


def main():
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    defect_dir = os.path.join(OUTPUT_DIR, "defect")
    ok_dir = os.path.join(OUTPUT_DIR, "ok")

    os.makedirs(defect_dir, exist_ok=True)
    os.makedirs(ok_dir, exist_ok=True)

    images = [f for f in os.listdir(INPUT_DIR) if is_image(f)]
    print(f"Found {len(images)} images")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        results = model(img, conf=CONF_THRESHOLD, verbose=False)

        detected = False

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                detected = True
                if SAVE_ANNOTATED:
                    img = r.plot()
                break

        if detected:
            out_path = os.path.join(defect_dir, img_name)
        else:
            out_path = os.path.join(ok_dir, img_name)

        cv2.imwrite(out_path, img)

        status = "DEFECT" if detected else "OK"
        print(f"{img_name}: {status}")

    print("\nInference complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
