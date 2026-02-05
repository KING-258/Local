from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
    QStackedWidget,
    QGraphicsDropShadowEffect,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # optional dependency for cloud sync
    import requests  # type: ignore[import]
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


log = logging.getLogger("local_app_monitor")


class _FrameStore:
    """Thread-safe store for the latest JPEG frame for MJPEG streaming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def update_bgr(self, frame: np.ndarray) -> None:
        """Encode a BGR frame as JPEG and store it.

        Called from the LiveWorker thread. We keep only the most recent frame.
        """
        try:
            ok, buf = cv2.imencode(".jpg", frame)
        except Exception:
            return
        if not ok:
            return
        data = buf.tobytes()
        with self._lock:
            self._jpeg = data

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg


_FRAME_STORE = _FrameStore()


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):  # type: ignore[misc]
    """HTTP server that handles each request in its own thread."""

    daemon_threads = True


class _MJPEGRequestHandler(BaseHTTPRequestHandler):
    """Very small MJPEG streaming handler serving /stream.

    This is designed for LAN use only. It exposes the latest frame from
    _FRAME_STORE as a multipart/x-mixed-replace stream.
    """

    server_version = "LocalMJPEG/0.1"

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path.split("?")[0] != "/stream":
            self.send_error(404, "Not found")
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        boundary = "frame"
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.end_headers()

        try:
            while True:
                jpeg = _FRAME_STORE.get_jpeg()
                if jpeg is None:
                    time.sleep(0.03)
                    continue

                try:
                    self.wfile.write(b"--" + boundary.encode("ascii") + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break
                except Exception:  # pragma: no cover - logging only
                    log.exception("MJPEG stream write failed")
                    break

                # Small sleep to avoid spinning too fast; ~30 FPS at 0.03s
                time.sleep(0.03)
        except Exception:  # pragma: no cover - logging only
            log.exception("MJPEG handler error")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Reduce console noise; log at debug level instead of stdout.
        log.debug("MJPEG: " + format, *args)


_STREAM_HTTPD: _ThreadedHTTPServer | None = None


def start_stream_server_if_enabled() -> None:
    """Start the MJPEG HTTP server in a background thread if enabled.

    Controlled by LOCAL_STREAM_ENABLED (default 0/disabled).
    Optional envs:
      - LOCAL_STREAM_PORT (default 8090)
      - LOCAL_STREAM_HOST (default 0.0.0.0)
    """
    global _STREAM_HTTPD

    enabled_raw = os.getenv("LOCAL_STREAM_ENABLED", "0").strip().lower()
    if enabled_raw in ("", "0", "false", "no"):  # feature explicitly disabled
        log.info("MJPEG stream server disabled (set LOCAL_STREAM_ENABLED=1 to enable)")
        return

    if _STREAM_HTTPD is not None:
        return

    try:
        port = int(os.getenv("LOCAL_STREAM_PORT", "8090"))
    except Exception:
        port = 8090
    host = os.getenv("LOCAL_STREAM_HOST", "0.0.0.0").strip() or "0.0.0.0"

    try:
        httpd = _ThreadedHTTPServer((host, port), _MJPEGRequestHandler)
    except OSError as exc:
        log.error("Failed to start MJPEG server on %s:%d: %s", host, port, exc)
        return

    _STREAM_HTTPD = httpd

    def _serve() -> None:
        log.info("Starting MJPEG stream server on http://%s:%d/stream", host, port)
        try:
            httpd.serve_forever()
        except Exception:  # pragma: no cover - logging only
            log.exception("MJPEG server crashed")

    thread = threading.Thread(target=_serve, name="mjpeg-server", daemon=True)
    thread.start()


def stop_stream_server() -> None:
    global _STREAM_HTTPD
    if _STREAM_HTTPD is not None:
        try:
            _STREAM_HTTPD.shutdown()
        except Exception:
            log.exception("Failed to shut down MJPEG server cleanly")
        _STREAM_HTTPD = None


def get_preferred_device() -> str:
    env = os.getenv("LOCAL_DEVICE")
    if env:
        return env

    if torch is None:
        return "cpu"

    try:
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                log.info("Using CUDA device: %s", name)
            except Exception:
                log.info("Using CUDA device 0")
            return "cuda"
    except Exception:
        log.warning("CUDA probe failed; falling back to CPU.", exc_info=True)

    return "cpu"


class LiveWorker(QThread):
    frame_ready = Signal(QImage)
    detection_state = Signal(bool, bool)
    fps_updated = Signal(float)
    error_occurred = Signal(str)

    def __init__(
        self,
        defect_model_path: Optional[Path],
        confidence: float = 0.65,
        camera_index: int = 2,
        parent: Optional[QWidget] = None,
        part_model_path: Optional[Path] = None,
    ) -> None:
        """Worker running a two-stage pipeline:

        1) Part model (if provided) decides whether the part is present.
        2) Defect model runs ONLY if the part is present.

        detection_state emits (has_part, has_defect).
        """
        super().__init__(parent)
        self._defect_model_path = defect_model_path
        self._part_model_path = part_model_path
        self._confidence = confidence
        self._camera_index = camera_index
        self._running = False
        self._frame_index = 0
        self._last_has_part = False
        self._last_has_defect = False

        # Cache for drawing bounding boxes on every frame so they don't
        # disappear between inference frames. We also hold them briefly
        # after detections stop to reduce flicker.
        self._last_part_boxes: list[tuple[int, int, int, int, float]] = []
        self._last_defect_boxes: list[tuple[int, int, int, int, float]] = []
        self._no_detection_since_time: float | None = None
        try:
            self._box_hold_seconds = float(os.getenv("LOCAL_BOX_HOLD_SECONDS", "0.3"))
        except Exception:
            self._box_hold_seconds = 0.3

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        from ultralytics import YOLO

        part_model = None
        defect_model = None
        if self._part_model_path is not None:
            try:
                part_model = YOLO(str(self._part_model_path))
            except Exception as exc:
                self.error_occurred.emit(f"Failed to load part model: {exc}")
                part_model = None
        if self._defect_model_path is not None:
            try:
                defect_model = YOLO(str(self._defect_model_path))
            except Exception as exc:
                self.error_occurred.emit(f"Failed to load defect model: {exc}")
                defect_model = None






        WIDTH = 1280
        HEIGHT = 720
        TARGET_FPS = 30
        # Very Important Check Cam Index by running v4l2-ctl --list-devices
        cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
            exit
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)





        if not cap.isOpened():
            self.error_occurred.emit(f"Unable to open webcam (index {self._camera_index})")
            return

        # Try to normalise camera settings so integrated and external webcams
        # behave similarly. Drivers may ignore some of these hints, but they
        # help avoid one camera defaulting to a very heavy 4K/YUYV mode.
        try:
            target_fps = int(os.getenv("LOCAL_CAM_FPS", "30"))
            target_width = int(os.getenv("LOCAL_CAM_WIDTH", "1280"))
            target_height = int(os.getenv("LOCAL_CAM_HEIGHT", "720"))
        except Exception:
            target_fps, target_width, target_height = 30, 1280, 720

        try:
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        except Exception:
            pass

        self._running = True

        device = get_preferred_device()
        log.info("Live worker using device: %s", device)

        # Allow controlling how often we run heavy YOLO inference relative to
        # the camera frame rate. Example: LOCAL_INFER_STRIDE=3 will run
        # inference on every 3rd frame but still display all frames.
        try:
            infer_stride = int(os.getenv("LOCAL_INFER_STRIDE", "2"))
            if infer_stride < 1:
                infer_stride = 1
        except Exception:
            infer_stride = 2

        # How often to push frames into the MJPEG streaming buffer.
        try:
            stream_stride = int(os.getenv("LOCAL_STREAM_STRIDE", "1"))
            if stream_stride < 1:
                stream_stride = 1
        except Exception:
            stream_stride = 1

        last_time = time.monotonic()

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error_occurred.emit("Failed to read frame from webcam")
                    break

                self._frame_index += 1
                run_inference = (self._frame_index % infer_stride) == 0

                now = time.monotonic()
                dt = now - last_time
                if dt > 0:
                    fps = 1.0 / dt
                    self.fps_updated.emit(fps)
                last_time = now

                has_part = self._last_has_part
                has_defect = self._last_has_defect
                detection_updated = False

                # If we don't have a part model, treat "part present" as false
                # so the defect model never runs. This enforces a strict
                # "part then defect" pipeline.
                if part_model is None:
                    has_part = False

                if run_inference and part_model is not None:
                    try:
                        part_results = part_model.predict(
                            frame,
                            conf=0.5,
                            imgsz=640,
                            device=device,
                            verbose=False,
                        )
                    except Exception as exc:
                        self.error_occurred.emit(f"Part inference failed: {exc}")
                        part_results = None

                    if not part_results:
                        has_part = False
                    else:
                        pres = part_results[0]
                        part_boxes = self._extract_boxes(pres)
                        if not part_boxes:
                            has_part = False
                        else:
                            has_part = True
                            self._last_part_boxes = part_boxes
                            detection_updated = True

                if run_inference and has_part and defect_model is not None:
                    try:
                        results = defect_model.predict(
                            frame,
                            conf=self._confidence,
                            imgsz=640,
                            device=device,
                            verbose=False,
                        )
                    except Exception as exc:
                        self.error_occurred.emit(f"Defect inference failed: {exc}")
                        results = None

                    if results:
                        res = results[0]
                        defect_boxes = self._extract_boxes(res)
                        if defect_boxes:
                            has_defect = True
                            self._last_defect_boxes = defect_boxes
                            detection_updated = True
                        else:
                            has_defect = False

                # Update box hold timer used by _draw_cached_boxes
                if detection_updated:
                    self._no_detection_since_time = None
                elif self._no_detection_since_time is None:
                    self._no_detection_since_time = now

                # Cache last known detection state so non-inference frames
                # still emit a meaningful signal for the jitter logic.
                self._last_has_part = has_part
                self._last_has_defect = has_defect

                # Draw cached boxes on every frame so bounding boxes don't
                # flicker when inference is skipped.
                frame = self._draw_cached_boxes(frame, now)

                # Optionally publish this frame into the MJPEG buffer for
                # the Edge Device live feed. We only encode every
                # LOCAL_STREAM_STRIDE frames to keep CPU usage manageable.
                if stream_stride > 0 and (self._frame_index % stream_stride) == 0:
                    try:
                        _FRAME_STORE.update_bgr(frame)
                    except Exception:
                        log.debug("Failed to update MJPEG frame store", exc_info=True)

                self.detection_state.emit(has_part, has_defect)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    frame,
                    timestamp,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy())
        finally:
            cap.release()

    @staticmethod
    def _extract_boxes(res) -> list[tuple[int, int, int, int, float]]:
        """Convert a YOLO result object into a simple list of boxes.

        Each entry is (x1, y1, x2, y2, conf).
        """
        boxes_out: list[tuple[int, int, int, int, float]] = []
        if not hasattr(res, "boxes") or res.boxes is None:
            return boxes_out

        for box in res.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                x1, y1, x2, y2 = xyxy.tolist()
            except Exception:
                continue

            boxes_out.append((x1, y1, x2, y2, conf))

        return boxes_out

    @staticmethod
    def _draw_boxes(
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int, float]],
        color: tuple[int, int, int],
        label_prefix: str,
    ) -> np.ndarray:
        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_prefix} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def _draw_cached_boxes(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Draw the last detected part/defect boxes on every frame.

        Boxes are kept for a short time (LOCAL_BOX_HOLD_SECONDS) after
        detections disappear to avoid visible flicker.
        """
        if self._no_detection_since_time is not None:
            if (now - self._no_detection_since_time) > self._box_hold_seconds:
                self._last_part_boxes = []
                self._last_defect_boxes = []
                self._no_detection_since_time = None

        frame = self._draw_boxes(frame, self._last_part_boxes, (255, 0, 0), "part")
        frame = self._draw_boxes(frame, self._last_defect_boxes, (0, 0, 255), "def")
        return frame


class ModelSelectionDialog(QDialog):
    """Simple dialog to choose a trained VisionM model to save locally.

    It lists models returned from /api/models for the current company/project
    and exposes the chosen modelId via `selected_model_id` when accepted.
    """

    def __init__(self, models: list[dict], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select VisionM model")
        self.resize(460, 320)
        self._models = models
        self.selected_model_id: Optional[str] = None

        layout = QVBoxLayout(self)

        info = QLabel(
            "Select a trained model for this company/project and click Save "
            "to download it into the current folder."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget(self)
        for m in models:
            version = m.get("modelVersion") or "v?"
            status = m.get("status") or "completed"
            created = m.get("createdAt") or ""
            metrics = m.get("metrics") or {}
            m_ap = metrics.get("mAP50")
            if isinstance(m_ap, (int, float)):
                metrics_str = f"  mAP50={m_ap:.2f}"
            else:
                metrics_str = ""
            label = f"{version}  [{status}]  {created}{metrics_str}"
            self.list_widget.addItem(QListWidgetItem(label))
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, parent=self)
        buttons.accepted.connect(self._on_save_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save_clicked(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No selection", "Please select a model first.")
            return

        model = self._models[row]
        model_id = model.get("modelId")
        if not model_id:
            QMessageBox.warning(self, "Invalid model", "Selected entry has no modelId.")
            return

        self.selected_model_id = str(model_id)
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Local – Live View")
        self.resize(1280, 720)

        # Theme stylesheets (dark / light). Buttons stay customized in both.
        self._dark_stylesheet = """
            QMainWindow {
                background-color: #111111;
            }
            QLabel {
                color: #f5f5f5;
                font-size: 13px;
            }
            QComboBox {
                background-color: #202020;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 4px 8px;
                color: #f5f5f5;
            }
            QComboBox QAbstractItemView {
                background-color: #202020;
                color: #f5f5f5;
                selection-background-color: #333333;
            }
            QWidget#InfoPanel {
                background-color: #181818;
                border-radius: 10px;
                border: 1px solid #303030;
            }
            QPushButton {
                background-color: #202020;
                border: 1px solid #404040;
                border-radius: 20px;
                padding: 6px 14px;
                color: #f5f5f5;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            QPushButton:checked {
                background-color: #3f51b5;
                border-color: #5c6bc0;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #181818;
                alternate-background-color: #151515;
                gridline-color: #303030;
                color: #f5f5f5;
                selection-background-color: #333333;
                selection-color: #ffffff;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #202020;
                color: #f5f5f5;
                padding: 3px 6px;
                border: 1px solid #303030;
                font-weight: 500;
            }
        """

        self._light_stylesheet = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #111111;
                font-size: 13px;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 4px 8px;
                color: #111111;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #111111;
                selection-background-color: #eeeeee;
            }
            QWidget#InfoPanel {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #dddddd;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 20px;
                padding: 6px 14px;
                color: #111111;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #3f51b5;
                border-color: #5c6bc0;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f7f7f7;
                gridline-color: #dddddd;
                color: #111111;
                selection-background-color: #e0e0ff;
                selection-color: #111111;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #111111;
                padding: 3px 6px;
                border: 1px solid #dddddd;
                font-weight: 500;
            }
        """

        self._dark_mode = True
        self.setStyleSheet(self._dark_stylesheet)

        # Ensure label colors are consistent with the initial theme
        self._worker: Optional[LiveWorker] = None
        # Single-camera index (no dropdown). Change here if you want a different
        # default camera, e.g. 1 for external USB webcam.
        self._camera_index_fixed: int = 1

        central = QWidget(self)
        layout = QGridLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 2)
        self.setCentralWidget(central)
        self.video_label = QLabel("Live video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #202020; border: 1px solid #404040;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 0, 0, 2, 1)
        self.dashboard_widget = QWidget(self)
        self.dashboard_widget.setObjectName("InfoPanel")
        self.dashboard_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        info_layout = QGridLayout(self.dashboard_widget)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setHorizontalSpacing(10)
        info_layout.setVerticalSpacing(8)
        project_name = os.getenv("LOCAL_PROJECT_NAME", "Nazar Project")
        dataset_name = os.getenv("LOCAL_DATASET_NAME", "Nazar Dataset")
        base_model = os.getenv("LOCAL_BASE_MODEL", "yolov11n")
        self._defect_count = 0
        self._good_count = 0
        self._is_defect_active = False
        self._part_active = False
        self._current_part_has_defect = False
        self._last_part_seen_time: float | None = None
        self._episodes: list[dict] = []
        self._stats_file = Path.cwd() / "local_stats.json"
        self._load_episodes_from_file()
        # Local persistence database (episodes + sync metadata)
        self._db_path = Path.cwd() / "local_stats.db"
        self._init_db()
        # On startup, reconcile any backlog of unsynced episodes in small batches
        # so that Mongo and the local SQLite database stay in sync.
        self._drain_unsynced_episodes()

        row = 0
        header = QLabel("SESSION")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setStyleSheet(
            "font-size: 15px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #dddddd;"
        )
        info_layout.addWidget(header, row, 0, 1, 2)

        row += 1
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #303030;")
        info_layout.addWidget(sep, row, 0, 1, 2)

        row += 1
        project_label = QLabel("Project:")
        project_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        project_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(project_label, row, 0)
        self.project_value = QLabel(project_name)
        self.project_value.setStyleSheet("font-weight: 500; color: #f0f0f0;")
        info_layout.addWidget(self.project_value, row, 1)

        row += 1
        dataset_label = QLabel("Dataset:")
        dataset_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        dataset_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(dataset_label, row, 0)
        self.dataset_value = QLabel(dataset_name)
        self.dataset_value.setStyleSheet("font-weight: 500; color: #f0f0f0;")
        info_layout.addWidget(self.dataset_value, row, 1)

        row += 1
        base_label = QLabel("Base model (YOLO):")
        base_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        base_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(base_label, row, 0)
        self.base_model_value = QLabel(base_model)
        self.base_model_value.setStyleSheet("font-weight: 500; color: #f0f0f0;")
        info_layout.addWidget(self.base_model_value, row, 1)
        row += 1
        fps_label = QLabel("FPS:")
        fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fps_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(fps_label, row, 0)
        self.fps_value = QLabel("–")
        self.fps_value.setStyleSheet("color: #ffb74d; font-family: 'JetBrains Mono', monospace;")
        info_layout.addWidget(self.fps_value, row, 1)
        row += 1
        cam_label = QLabel("Camera:")
        cam_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cam_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(cam_label, row, 0)
        # Single-camera setup: always use index 0, no dropdown
        self.camera_label = QLabel("Camera 0")
        self.camera_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.camera_label.setStyleSheet("font-weight: 500; color: #f0f0f0;")
        info_layout.addWidget(self.camera_label, row, 1)
        row += 1
        part_label = QLabel("Part model (.pt):")
        part_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        part_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(part_label, row, 0)
        self.part_model_combo = QComboBox()
        row += 1
        defect_label = QLabel("Defect model (.pt):")
        defect_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        defect_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(defect_label, row, 0)
        self.defect_model_combo = QComboBox()

        self._model_paths: list[Path] = []
        self._populate_models()
        self.part_model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.defect_model_combo.currentIndexChanged.connect(self._on_model_changed)
        info_layout.addWidget(self.part_model_combo, row - 1, 1)
        info_layout.addWidget(self.defect_model_combo, row, 1)
        row += 1

        # Button to browse trained models from the VisionM backend
        self.fetch_model_button = QPushButton("Browse VisionM models")
        self.fetch_model_button.clicked.connect(self._browse_visionm_models)
        info_layout.addWidget(self.fetch_model_button, row, 1)
        row += 1
        flag_text = QLabel("Flag (status):")
        flag_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        flag_text.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(flag_text, row, 0)

        self.flag_frame = QFrame()
        self.flag_frame.setFrameShape(QFrame.Box)
        self.flag_frame.setFixedSize(300, 300)
        self.flag_frame.setStyleSheet("background-color: #00aa00; border: 2px solid #202020;")

        self.flag_label = QLabel("NO DETECTIONS")
        self.flag_label.setAlignment(Qt.AlignCenter)
        self.flag_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        flag_container = QWidget()
        flag_layout = QGridLayout(flag_container)
        flag_layout.setContentsMargins(0, 8, 0, 0)
        flag_layout.setVerticalSpacing(6)
        flag_layout.addWidget(self.flag_frame, 0, 0, alignment=Qt.AlignCenter)
        flag_layout.addWidget(self.flag_label, 1, 0, alignment=Qt.AlignCenter)
        info_layout.addWidget(flag_container, row, 1)
        row += 1
        stats_header = QLabel("STATS")
        stats_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        stats_header.setStyleSheet(
            "font-size: 15px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #dddddd;"
        )
        info_layout.addWidget(stats_header, row, 0, 1, 2)

        row += 1
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #303030;")
        info_layout.addWidget(sep2, row, 0, 1, 2)

        row += 1
        stats_label = QLabel("Stats:")
        stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        stats_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        info_layout.addWidget(stats_label, row, 0)
        self.stats_value = QLabel("")
        self.stats_value.setStyleSheet("font-family: 'JetBrains Mono', monospace; color: #dddddd;")
        info_layout.addWidget(self.stats_value, row, 1)
        self._update_stats_label()
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Use a QStackedWidget so switching between Dashboard and Monitor
        # does not change the overall size hint or push the bottom buttons
        # off-screen.
        self.stack_widget = QStackedWidget(self)
        self.stack_widget.addWidget(self.dashboard_widget)

        self.monitor_widget = QWidget(self)
        self.monitor_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        monitor_layout = QVBoxLayout(self.monitor_widget)
        monitor_layout.setContentsMargins(18, 18, 18, 18)
        monitor_layout.setSpacing(8)

        monitor_title = QLabel("MONITOR / REPORTS")
        monitor_title.setStyleSheet(
            "font-size: 15px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #dddddd;"
        )
        monitor_layout.addWidget(monitor_title)

        filter_row = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("font-weight: 500; color: #bbbbbb;")
        self.monitor_filter_combo = QComboBox()
        self.monitor_filter_combo.addItems(["All", "Today", "This week", "This month"])
        self.monitor_filter_combo.currentIndexChanged.connect(self._refresh_monitor_view)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.monitor_filter_combo)
        filter_row.addStretch()
        monitor_layout.addLayout(filter_row)

        self.monitor_stats_label = QLabel("")
        self.monitor_stats_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #f5f5f5;"
        )
        monitor_layout.addWidget(self.monitor_stats_label)
        self.monitor_table = QTableWidget(0, 4)
        self.monitor_table.setHorizontalHeaderLabels(["Date", "Total", "Good", "Defects"])
        self.monitor_table.horizontalHeader().setStretchLastSection(True)
        self.monitor_table.verticalHeader().setVisible(False)
        self.monitor_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.monitor_table.setMinimumHeight(180)
        self.monitor_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        monitor_layout.addWidget(self.monitor_table)
        reports_header = QLabel("REPORTS")
        reports_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        reports_header.setStyleSheet(
            "font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #dddddd;"
        )
        monitor_layout.addWidget(reports_header)
        self.hist_figure = Figure(figsize=(4, 1.8), tight_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_ax = self.hist_figure.add_subplot(111)
        monitor_layout.addWidget(self.hist_canvas)

        self.ctrl_figure = Figure(figsize=(4, 1.8), tight_layout=True)
        self.ctrl_canvas = FigureCanvas(self.ctrl_figure)
        self.ctrl_ax = self.ctrl_figure.add_subplot(111)
        monitor_layout.addWidget(self.ctrl_canvas)

        self.monitor_widget.setVisible(False)
        self.stack_widget.addWidget(self.monitor_widget)

        right_layout.addWidget(self.stack_widget)
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, 0, 0, 0)
        buttons_row.setSpacing(8)
        buttons_row.addStretch()
        self.dashboard_button = QPushButton("Dashboard")
        self.dashboard_button.setCheckable(True)
        self.dashboard_button.setChecked(True)
        self.dashboard_button.clicked.connect(self._show_dashboard)
        self.monitor_button = QPushButton("Monitor")
        self.monitor_button.setCheckable(True)
        self.monitor_button.setChecked(False)
        self.monitor_button.clicked.connect(self._show_monitor)
        buttons_row.addWidget(self.dashboard_button)
        buttons_row.addWidget(self.monitor_button)

        # Theme toggle button (sun / moon)
        self.theme_toggle_button = QPushButton("🌙")
        self.theme_toggle_button.setCheckable(False)
        self.theme_toggle_button.setFixedWidth(40)
        self.theme_toggle_button.setToolTip("Toggle dark / light mode")
        self.theme_toggle_button.clicked.connect(self._toggle_theme)
        buttons_row.addWidget(self.theme_toggle_button)

        right_layout.addLayout(buttons_row)

        # Apply a soft drop shadow to the right-side container for depth
        self._apply_card_shadow(self.dashboard_widget)
        self._apply_card_shadow(self.monitor_widget)

        layout.addWidget(right_container, 0, 1, 2, 1)

        self._last_detection_time: float | None = None
        self._start_worker_with_current_model()

        # Optional background sync timer to VisionM backend
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(60_000)  # 60 seconds
        self._sync_timer.timeout.connect(self._sync_to_cloud)
        self._sync_timer.start()

    def _toggle_theme(self) -> None:
        """Toggle between dark and light stylesheets."""
        self._dark_mode = not self._dark_mode
        if self._dark_mode:
            self.setStyleSheet(self._dark_stylesheet)
            self.theme_toggle_button.setText("🌙")
        else:
            self.setStyleSheet(self._light_stylesheet)
            self.theme_toggle_button.setText("☀")
        self._apply_label_theme()

    def _apply_label_theme(self) -> None:
        """Adjust per-widget label colors to match the current theme."""
        if self._dark_mode:
            primary = "#f0f0f0"
            mono = "#dddddd"
            monitor_stats = "#f5f5f5"
        else:
            primary = "#111111"
            mono = "#333333"
            monitor_stats = "#222222"

        # Main value labels
        self.project_value.setStyleSheet(f"font-weight: 500; color: {primary};")
        self.dataset_value.setStyleSheet(f"font-weight: 500; color: {primary};")
        self.base_model_value.setStyleSheet(f"font-weight: 500; color: {primary};")
        self.camera_label.setStyleSheet(f"font-weight: 500; color: {primary};")
        # Stats text
        self.stats_value.setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; color: {mono};"
        )
        self.monitor_stats_label.setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; font-size: 14px; color: {monitor_stats};"
        )

    def _apply_card_shadow(self, widget: QWidget) -> None:
        """Apply a subtle drop shadow to a panel-like widget."""
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(24)
        effect.setXOffset(0)
        effect.setYOffset(8)
        effect.setColor(Qt.black)
        widget.setGraphicsEffect(effect)

    # ------------------------------------------------------------------
    # Model browser / download from VisionM backend
    # ------------------------------------------------------------------

    def _browse_visionm_models(self) -> None:
        """Show a list of trained models and let the user choose one to save.

        This calls /api/models for the current company/project and opens a
        dialog where the user can pick a model. Only when they click "Save"
        do we download /api/models/:id/download into the current directory.
        """
        if requests is None:
            QMessageBox.warning(
                self,
                "Network unavailable",
                "The Python 'requests' package is not available.",
            )
            return

        api_base = os.getenv("LOCAL_API_BASE_URL", "").strip()
        if not api_base:
            QMessageBox.warning(
                self,
                "Missing API URL",
                "Set LOCAL_API_BASE_URL to your VisionBackend base URL.",
            )
            return

        company = os.getenv("LOCAL_COMPANY_NAME", "").strip()
        project = os.getenv("LOCAL_PROJECT_NAME", "").strip() or self.project_value.text().strip()
        if not company or not project:
            QMessageBox.warning(
                self,
                "Missing details",
                "Company or project is not set. Please set LOCAL_COMPANY_NAME and LOCAL_PROJECT_NAME.",
            )
            return

        try:
            # 1) List models for this company + project
            list_url = api_base.rstrip("/") + "/models"
            params = {"company": company, "project": project}
            resp = requests.get(list_url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            models = payload.get("models") or []
            if not models:
                QMessageBox.information(
                    self,
                    "No models",
                    "No trained models were found for this company/project.",
                )
                return

            # 2) Let the user pick a model to save
            dlg = ModelSelectionDialog(models, parent=self)
            if dlg.exec() != QDialog.Accepted or not dlg.selected_model_id:
                return

            selected_id = dlg.selected_model_id
            # Find the full model dict (for optional naming/metrics)
            selected_model = next(
                (m for m in models if str(m.get("modelId")) == selected_id),
                None,
            )

            # 3) Download chosen checkpoint
            download_url = api_base.rstrip("/") + f"/models/{selected_id}/download"
            resp = requests.get(download_url, stream=True, timeout=60)
            resp.raise_for_status()

            # Infer filename from Content-Disposition if present
            filename: Optional[str] = None
            cd = resp.headers.get("Content-Disposition")
            if cd and "filename=" in cd:
                try:
                    filename = cd.split("filename=")[-1].strip().strip('"')
                except Exception:
                    filename = None
            if not filename:
                version = (selected_model or {}).get("modelVersion") or "model"
                filename = f"{project}_{version}.pt"

            target_path = Path.cwd() / filename
            with target_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            log.info("Downloaded model %s to %s", selected_id, target_path)
            self._populate_models()
            QMessageBox.information(
                self,
                "Model saved",
                f"Saved trained model to {target_path.name}",
            )
        except Exception as exc:
            log.exception("Failed to browse/download model from VisionM")
            QMessageBox.warning(
                self,
                "Download failed",
                f"Could not download model from VisionM: {exc}",
            )

    def _populate_cameras(self) -> None:
        """Deprecated: camera dropdown removed (single-camera mode)."""
        return

    def _populate_models(self) -> None:
        self.part_model_combo.clear()
        self.defect_model_combo.clear()
        self._model_paths.clear()
        cwd = Path.cwd()
        pt_files = sorted(p for p in cwd.glob("*.pt") if p.is_file())
        if not pt_files:
            self.part_model_combo.addItem("(no .pt files found)")
            self.defect_model_combo.addItem("(no .pt files found)")
            self.part_model_combo.setEnabled(False)
            self.defect_model_combo.setEnabled(False)
            return
        for p in pt_files:
            name = p.name
            self.part_model_combo.addItem(name)
            self.defect_model_combo.addItem(name)
            self._model_paths.append(p)
        self.part_model_combo.setEnabled(True)
        self.defect_model_combo.setEnabled(True)

    def _current_part_model_path(self) -> Optional[Path]:
        if not self._model_paths:
            return None
        idx = self.part_model_combo.currentIndex()
        if idx < 0 or idx >= len(self._model_paths):
            return self._model_paths[0]
        return self._model_paths[idx]

    def _current_defect_model_path(self) -> Optional[Path]:
        if not self._model_paths:
            return None
        idx = self.defect_model_combo.currentIndex()
        if idx < 0 or idx >= len(self._model_paths):
            return self._model_paths[0]
        return self._model_paths[idx]

    def _update_stats_label(self) -> None:
        total = self._good_count + self._defect_count
        self.stats_value.setText(
            f"Good: {self._good_count} | Defects: {self._defect_count} | Total: {total}"
        )
        
    def _load_episodes_from_file(self) -> None:
        self._episodes = []
        if self._stats_file.exists():
            try:
                with self._stats_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for e in data:
                        if isinstance(e, dict) and e.get("result") in ("good", "defect") and "ts" in e:
                            self._episodes.append(e)
            except Exception:
                self._episodes = []
        self._good_count = sum(1 for e in self._episodes if e.get("result") == "good")
        self._defect_count = sum(1 for e in self._episodes if e.get("result") == "defect")

    def _init_db(self) -> None:
        """Initialise the SQLite database for episodes + sync metadata.

        JSON remains the primary source for in-memory stats; the DB mirrors it
        and is seeded from JSON on first run.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                # Episodes table mirrors the JSON log
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        result TEXT NOT NULL CHECK (result IN ('good', 'defect'))
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)"
                )

                # Single-row table tracking last successful cloud sync
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        last_synced_ts TEXT,
                        last_sync_at TEXT
                    )
                    """
                )
                cur.execute(
                    "INSERT OR IGNORE INTO sync_state (id, last_synced_ts, last_sync_at) VALUES (1, NULL, NULL)"
                )

                # Seed episodes from JSON into DB on first run
                cur.execute("SELECT COUNT(*) FROM episodes")
                (count,) = cur.fetchone()
                if count == 0 and self._episodes:
                    payload = [
                        (e["ts"], e["result"])
                        for e in self._episodes
                        if "ts" in e and "result" in e
                    ]
                    cur.executemany(
                        "INSERT INTO episodes (ts, result) VALUES (?, ?)", payload
                    )
                conn.commit()
        except Exception as exc:
            log.error("Failed to initialise SQLite DB: %s", exc)

    def _save_episodes_to_file(self) -> None:
        try:
            with self._stats_file.open("w", encoding="utf-8") as f:
                json.dump(self._episodes, f, indent=2)
        except Exception:
            pass

    def _insert_episode_db(self, ts: str, result: str) -> None:
        """Insert a single episode row into the SQLite database.

        Errors are logged but do not interrupt the main application.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO episodes (ts, result) VALUES (?, ?)", (ts, result)
                )
                conn.commit()
        except Exception as exc:
            log.error("Failed to insert episode into SQLite DB: %s", exc)

    def _get_sync_state(self) -> tuple[Optional[str], Optional[str]]:
        """Return (last_synced_ts, last_sync_at) from SQLite.

        Both values are ISO-8601 strings or None on error/first run.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT last_synced_ts, last_sync_at FROM sync_state WHERE id = 1"
                )
                row = cur.fetchone()
        except Exception as exc:
            log.error("Failed to read sync_state from SQLite DB: %s", exc)
            return None, None

        if not row:
            return None, None
        return row[0], row[1]

    def _update_sync_state(self, last_synced_ts: Optional[str]) -> None:
        """Update sync_state row after a successful cloud sync.

        We store timestamps as UTC ISO-8601 with a trailing 'Z'. Use
        timezone-aware datetimes to avoid deprecated utcnow().
        """
        from datetime import UTC

        now_iso = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE sync_state SET last_synced_ts = ?, last_sync_at = ? WHERE id = 1",
                    (last_synced_ts, now_iso),
                )
                conn.commit()
        except Exception as exc:
            log.error("Failed to update sync_state in SQLite DB: %s", exc)

    def _edge_log(self, level: str, event_type: str, message: str) -> None:
        """Send a lightweight edge connectivity/sync event to VisionM.

        This is best-effort only; failures are logged but ignored.
        """
        if requests is None:
            return

        api_base = os.getenv("LOCAL_API_BASE_URL", "").strip()
        if not api_base:
            return

        company_name = os.getenv("LOCAL_COMPANY_NAME", "").strip() or None
        shop_id = os.getenv("LOCAL_SHOP_ID", os.uname().nodename)

        try:
            url = api_base.rstrip("/") + "/local-sync/edge-log"
            payload = {
                "company": company_name,
                "shopId": shop_id,
                "level": level,
                "eventType": event_type,
                "message": message,
            }
            requests.post(url, json=payload, timeout=3)
        except Exception:
            log.debug("EDGE-LOG failed", exc_info=True)

    def _log_episode(self, result: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        episode = {"ts": ts, "result": result}
        self._episodes.append(episode)
        self._save_episodes_to_file()
        self._insert_episode_db(ts, result)

        # Trigger a cloud sync whenever a new episode is recorded so that
        # the TRY/VisionM UI updates as soon as local_stats.json is written.
        # _sync_to_cloud reads from SQLite and only pushes genuinely new
        # episodes in bounded batches, so repeated calls are safe.
        try:
            self._sync_to_cloud()
        except Exception:
            # Never let sync errors break the UI loop
            log.exception("_sync_to_cloud failed after logging episode")

    def _drain_unsynced_episodes(self) -> None:
        """On startup, bring Mongo in sync with the local SQLite episode log.

        We repeatedly call `_sync_to_cloud` in a loop until `last_synced_ts`
        stops advancing, meaning there are no more unsynced episodes. This is
        a best-effort operation; any errors are logged and do not block the UI.
        """
        if requests is None:
            return

        api_base = os.getenv("LOCAL_API_BASE_URL", "").strip()
        if not api_base:
            return

        # Limit the number of iterations so we don't block startup forever
        for _ in range(50):
            before_ts, _ = self._get_sync_state()
            self._sync_to_cloud()
            after_ts, _ = self._get_sync_state()
            if before_ts == after_ts:
                break

        # Log that Local came online and reconciliation completed (best-effort).
        try:
            self._edge_log("info", "online", "Local app started and reconciliation completed")
        except Exception:
            pass

    def _sync_to_cloud(self) -> None:
        """Best-effort push of new episodes to the VisionM backend.

        This uses environment variables for configuration:
        - LOCAL_API_BASE_URL (e.g. "http://192.168.1.24:3000/api")
        - LOCAL_COMPANY_NAME (must match VisionM workspace/company name)
        - LOCAL_SHOP_ID (identifier for this local client / line)

        Episodes are read in *batches* from SQLite so the HTTP payload stays small
        and we avoid 413 Payload Too Large errors when there is a backlog.
        """
        log.info("SYNC: starting sync_to_cloud")
        if requests is None:
            # HTTP client not available – skip silently
            return

        api_base = os.getenv("LOCAL_API_BASE_URL", "").strip()
        if not api_base:
            return

        company_name = os.getenv("LOCAL_COMPANY_NAME", "").strip()
        shop_id = os.getenv("LOCAL_SHOP_ID", os.uname().nodename)

        last_synced_ts, _ = self._get_sync_state()

        # If the stored cursor is ahead of all locally known episodes (e.g.
        # due to a previous bug or clock change), treat it as invalid and
        # reset. ISO-8601 strings compare correctly in lexicographic order,
        # so we can compare them directly.
        if self._episodes and last_synced_ts is not None:
            try:
                episode_ts_values = [
                    e.get("ts")
                    for e in self._episodes
                    if isinstance(e, dict) and e.get("ts")
                ]
                if episode_ts_values:
                    max_local_ts = max(episode_ts_values)
                    if last_synced_ts > max_local_ts:
                        log.warning(
                            "SYNC: last_synced_ts=%s is ahead of latest local episode %s; resetting cursor",
                            last_synced_ts,
                            max_local_ts,
                        )
                        last_synced_ts = None
                        # Persist reset so subsequent calls see a sane cursor
                        self._update_sync_state(None)
            except Exception:
                log.exception("SYNC: failed to validate last_synced_ts against local episodes")

        # Read a bounded batch of unsynced episodes directly from SQLite.
        # This keeps the JSON payload small even if there is a large backlog.
        BATCH_LIMIT = 200
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                if last_synced_ts is not None:
                    cur.execute(
                        "SELECT ts, result FROM episodes WHERE ts > ? ORDER BY id ASC LIMIT ?",
                        (last_synced_ts, BATCH_LIMIT),
                    )
                else:
                    cur.execute(
                        "SELECT ts, result FROM episodes ORDER BY id ASC LIMIT ?",
                        (BATCH_LIMIT,),
                    )
                rows = cur.fetchall()
        except Exception as exc:
            log.error("SYNC: failed to read episodes from SQLite: %s", exc)
            return

        new_episodes: list[dict[str, str]] = []

        if rows:
            # Normal path: take unsynced rows from SQLite
            new_episodes = [{"ts": ts, "result": result} for (ts, result) in rows]
        else:
            # Fallback path: SQLite reports nothing newer, but the in-memory
            # episode list (backed by local_stats.json) may contain episodes
            # that were never inserted into the DB (e.g. from older runs).
            # In that case, scan self._episodes and pick anything newer than
            # last_synced_ts. The backend upserts on (company, shopId, ts, result)
            # so sending duplicates is safe.
            try:
                candidates = []
                for e in self._episodes:
                    if not isinstance(e, dict):
                        continue
                    ts = e.get("ts")
                    result = e.get("result")
                    if not ts or result not in ("good", "defect"):
                        continue
                    if last_synced_ts is not None and not (ts > last_synced_ts):
                        continue
                    candidates.append({"ts": ts, "result": result})

                # Sort by timestamp and respect batch limit
                candidates.sort(key=lambda ep: ep["ts"])
                if candidates:
                    new_episodes = candidates[:BATCH_LIMIT]
            except Exception:
                log.exception("SYNC: failed to build JSON-based fallback episode batch")

        if not new_episodes:
            log.info("SYNC: no new episodes to sync (last_synced_ts=%s)", last_synced_ts)
            return

        payload = {
            "company": company_name or None,
            "shopId": shop_id,
            "episodes": new_episodes,
        }

        try:
            url = api_base.rstrip("/") + "/local-sync/upload"
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            log.info(
                "SYNC: posted %d episodes (batch limit %d), status=%s",
                len(new_episodes),
                BATCH_LIMIT,
                resp.status_code,
            )
            # Record a connectivity log on success
            self._edge_log(
                "info",
                "sync_ok",
                f"Synced {len(new_episodes)} episodes (batch limit {BATCH_LIMIT}), status={resp.status_code}",
            )
        except Exception as exc:
            log.error("Failed to sync episodes to VisionM backend: %s", exc)
            # Log sync failure as edge event (but don't raise)
            self._edge_log("error", "sync_error", f"Sync failed: {exc}")
            return

        # On success, record the latest timestamp we just tried to sync.
        # Because we query ORDER BY id ASC, the last row is the newest.
        last_ts = max(e["ts"] for e in new_episodes)
        self._update_sync_state(last_ts)

    def _show_dashboard(self) -> None:
        # Switch stacked widget to Dashboard without resizing window
        self.stack_widget.setCurrentWidget(self.dashboard_widget)
        self.dashboard_widget.setVisible(True)
        self.monitor_widget.setVisible(False)
        self.dashboard_button.setChecked(True)
        self.monitor_button.setChecked(False)

    def _show_monitor(self) -> None:
        # Switch stacked widget to Monitor without resizing window
        self.stack_widget.setCurrentWidget(self.monitor_widget)
        self.dashboard_widget.setVisible(False)
        self.monitor_widget.setVisible(True)
        self.dashboard_button.setChecked(False)
        self.monitor_button.setChecked(True)
        self._refresh_monitor_view()

    def _refresh_monitor_view(self) -> None:
        now = datetime.now()
        today = now.date()
        filt = self.monitor_filter_combo.currentText()

        def in_range(ts_str: str) -> bool:
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                return False
            d = ts.date()
            if filt == "Today":
                return d == today
            if filt == "This week":
                return d >= today - timedelta(days=7)
            if filt == "This month":
                return d.year == today.year and d.month == today.month
            return True
        buckets: dict[str, dict[str, int]] = {}
        for e in self._episodes:
            ts_str = e.get("ts", "")
            if not in_range(ts_str):
                continue
            try:
                d = datetime.fromisoformat(ts_str).date().isoformat()
            except Exception:
                continue
            bucket = buckets.setdefault(d, {"good": 0, "defect": 0})
            if e.get("result") == "good":
                bucket["good"] += 1
            elif e.get("result") == "defect":
                bucket["defect"] += 1
        good_total = sum(v["good"] for v in buckets.values())
        defects_total = sum(v["defect"] for v in buckets.values())
        total = good_total + defects_total
        if total > 0:
            defect_rate = defects_total / total
            rate_text = f"{defect_rate:.2%}"
        else:
            rate_text = "N/A"
        self.monitor_stats_label.setText(
            f"Good: {good_total}    Defects: {defects_total}    Total: {total}    Defect rate: {rate_text}"
        )
        dates = sorted(buckets.keys(), reverse=True)
        self.monitor_table.setRowCount(len(dates))
        for row, d in enumerate(dates):
            counts = buckets[d]
            day_total = counts["good"] + counts["defect"]
            self.monitor_table.setItem(row, 0, QTableWidgetItem(d))
            self.monitor_table.setItem(row, 1, QTableWidgetItem(str(day_total)))
            self.monitor_table.setItem(row, 2, QTableWidgetItem(str(counts["good"])))
            self.monitor_table.setItem(row, 3, QTableWidgetItem(str(counts["defect"])))
        self.hist_ax.clear()
        if dates:
            defect_counts = []
            x = list(range(len(dates)))
            for d in dates:
                counts = buckets[d]
                defect_counts.append(counts["defect"])
            self.hist_ax.bar(x, defect_counts, color="#ef5350")
            self.hist_ax.set_xticks(x)
            self.hist_ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
            self.hist_ax.set_ylabel("Defects")
            ymax = max(defect_counts) if defect_counts else 1
            self.hist_ax.set_ylim(0, max(1, ymax * 1.2))
        else:
            self.hist_ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#cccccc")
            self.hist_ax.set_xticks([])
            self.hist_ax.set_yticks([])
        self.hist_canvas.draw_idle()
        self.ctrl_ax.clear()
        if dates:
            x = list(range(len(dates)))
            rates = []
            for d in dates:
                counts = buckets[d]
                tot = counts["good"] + counts["defect"]
                rates.append(counts["defect"] / tot if tot else 0.0)
            avg_rate = sum(rates) / len(rates)
            self.ctrl_ax.plot(x, rates, marker="o", color="#29b6f6", linewidth=1.5)
            self.ctrl_ax.axhline(avg_rate, color="#ffb74d", linestyle="--", linewidth=1)
            self.ctrl_ax.set_xticks(x)
            self.ctrl_ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
            self.ctrl_ax.set_ylabel("Defect rate")
            self.ctrl_ax.set_ylim(0, 0.2)
        else:
            self.ctrl_ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#cccccc")
            self.ctrl_ax.set_xticks([])
            self.ctrl_ax.set_yticks([])
        self.ctrl_canvas.draw_idle()

    def _current_camera_index(self) -> int:
        # Always use the fixed camera index (single-camera setup)
        return self._camera_index_fixed

    def _on_model_changed(self, _idx: int) -> None:
        self._stop_worker()
        self._start_worker_with_current_model()

    def _on_camera_changed(self, _idx: int) -> None:
        # Deprecated: camera dropdown removed.
        pass

    def _start_worker_with_current_model(self) -> None:
        if self._worker is not None:
            return
        defect_model_path = self._current_defect_model_path()
        part_model_path = self._current_part_model_path()
        camera_index = self._current_camera_index()
        self._worker = LiveWorker(
            defect_model_path,
            confidence=0.65,
            camera_index=camera_index,
            parent=self,
            part_model_path=part_model_path,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.detection_state.connect(self._on_detection_state)
        self._worker.fps_updated.connect(self._on_fps_updated)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None

    def _on_frame_ready(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _on_detection_state(self, has_part: bool, has_defect: bool) -> None:
        HOLD_SECONDS = 0.4
        PART_GAP_SECONDS = 0.5
        now = time.monotonic()
        if has_part:
            self._last_part_seen_time = now
            if not self._part_active:
                self._part_active = True
                self._current_part_has_defect = False
        else:
            if self._part_active and self._last_part_seen_time is not None:
                if (now - self._last_part_seen_time) >= PART_GAP_SECONDS:
                    if not self._current_part_has_defect:
                        self._good_count += 1
                        self._log_episode("good")
                    self._part_active = False
                    self._current_part_has_defect = False
        if has_part and has_defect:
            if not self._is_defect_active:
                self._defect_count += 1
                self._current_part_has_defect = True
                self._is_defect_active = True
                self._log_episode("defect")
            self._last_detection_time = now
            self.flag_frame.setStyleSheet(
                "background-color: #cc0000; border: 2px solid #202020;"
            )
            self.flag_label.setStyleSheet(
                "font-size: 18px; font-weight: 600; color: #ffecec;"
            )
            self.flag_label.setText("DETECTED")
        else:
            if self._is_defect_active and self._last_detection_time is not None:
                if (now - self._last_detection_time) >= HOLD_SECONDS:
                    self._is_defect_active = False
            if not self._is_defect_active:
                self.flag_frame.setStyleSheet(
                    "background-color: #00aa00; border: 2px solid #202020;"
                )
                self.flag_label.setStyleSheet(
                    "font-size: 18px; font-weight: 600; color: #eaffea;"
                )
                self.flag_label.setText("NO DETECTIONS")

        self._update_stats_label()

    def _on_fps_updated(self, fps: float) -> None:
        self.fps_value.setText(f"{fps:.1f} FPS")

    def _on_error(self, message: str) -> None:
        log.error("Live worker error: %s", message)
        self.video_label.setText(message)

    def _on_worker_finished(self) -> None:
        self._worker = None

    def closeEvent(self, event) -> None:
        self._stop_worker()
        # Stop MJPEG server when window closes (best-effort).
        stop_stream_server()
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Local")

    # Start MJPEG HTTP server (for Edge Device live feed) if enabled.
    start_stream_server_if_enabled()

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
