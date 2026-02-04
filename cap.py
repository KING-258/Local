import cv2
import time

DEV = "/dev/video2"
WIDTH = 1280
HEIGHT = 720
TARGET_FPS = 30

cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)

# Force MJPEG + FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

# Confirm driver settings
print("Driver FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Resolution:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

prev = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()

    if now - prev >= 1.0:
        fps = frame_count / (now - prev)
        frame_count = 0
        prev = now

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
