from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
    QStackedWidget,
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





        cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)    # Very Important Check Cam Index by running v4l2-ctl --list-devices










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
                        if getattr(pres, "boxes", None) is None or len(pres.boxes) == 0:
                            has_part = False
                        else:
                            has_part = True
                            frame = self._draw_part_boxes(frame, pres)

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
                        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                            has_defect = True
                        else:
                            has_defect = False
                        frame = self._draw_defect_boxes(frame, res)

                # Cache last known detection state so non-inference frames
                # still emit a meaningful signal for the jitter logic.
                self._last_has_part = has_part
                self._last_has_defect = has_defect

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
    def _draw_part_boxes(frame: np.ndarray, res) -> np.ndarray:
        """Draw part detections (from the part model) in blue."""
        if not hasattr(res, "boxes") or res.boxes is None:
            return frame

        for box in res.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                x1, y1, x2, y2 = xyxy.tolist()
            except Exception:
                continue

            color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"part {conf:.2f}"
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

    @staticmethod
    def _draw_defect_boxes(frame: np.ndarray, res) -> np.ndarray:
        """Draw defect detections (from the defect model) in red."""
        if not hasattr(res, "boxes") or res.boxes is None:
            return frame

        for box in res.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                x1, y1, x2, y2 = xyxy.tolist()
            except Exception:
                continue

            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"def {conf:.2f}"
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Local – Live View")
        self.resize(1280, 720)
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #111111;
            }
            QLabel {
                color: #f5f5f5;
            }
            QComboBox {
                background-color: #202020;
                border: 1px solid #404040;
                border-radius: 4px;
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
                border-radius: 8px;
                border: 1px solid #303030;
            }
            QPushButton {
                background-color: #202020;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px 10px;
                color: #f5f5f5;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:checked {
                background-color: #3f51b5;
                border-color: #5c6bc0;
            }
            QTableWidget {
                background-color: #181818;
                alternate-background-color: #151515;
                gridline-color: #303030;
                color: #f5f5f5;
                selection-background-color: #333333;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #202020;
                color: #f5f5f5;
                padding: 3px 6px;
                border: 1px solid #303030;
            }
            """
        )

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
        right_layout.addLayout(buttons_row)

        layout.addWidget(right_container, 0, 1, 2, 1)

        self._last_detection_time: float | None = None
        self._start_worker_with_current_model()

        # Optional background sync timer to VisionM backend
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(60_000)  # 60 seconds
        self._sync_timer.timeout.connect(self._sync_to_cloud)
        self._sync_timer.start()
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
        """Update sync_state row after a successful cloud sync."""
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
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

    def _log_episode(self, result: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        episode = {"ts": ts, "result": result}
        self._episodes.append(episode)
        self._save_episodes_to_file()
        self._insert_episode_db(ts, result)

        # For normal operation, only trigger cloud sync when a defect is detected.
        # This keeps network traffic focused on the most critical events while the
        # startup reconciliation handles any accumulated backlog.
        if result != "defect":
            return

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

        if not rows:
            log.info("SYNC: no new episodes to sync (last_synced_ts=%s)", last_synced_ts)
            return

        new_episodes = [{"ts": ts, "result": result} for (ts, result) in rows]

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
        except Exception as exc:
            log.error("Failed to sync episodes to VisionM backend: %s", exc)
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
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Local")

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
