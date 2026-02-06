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
os.environ["LOCAL_STREAM_ENABLED"] = "1"
os.environ["LOCAL_SHOP_ID"] = "Shop1-LineA"
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
    QLineEdit,
    QFormLayout,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from dataclasses import dataclass

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # optional dependency for cloud sync
    import requests  # type: ignore[import]
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:  # optional dependency for Supabase auth / queries
    from supabase import create_client, Client  # type: ignore[import]
except Exception:  # pragma: no cover
    create_client = None  # type: ignore[assignment]
    Client = object  # type: ignore[assignment]


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

# ---------------------------------------------------------------------------
# Supabase + backend configuration (Local app)
# ---------------------------------------------------------------------------

# VisionBackend API base URL used by the Local app
# Fallback to a sensible default if env is not set so .strip() calls are safe.
API_BASE_URL = (os.getenv("LOCAL_API_BASE_URL") or "http://127.0.0.1:3000/api").strip()

# Optional static workspace/company default; normally we derive this from Supabase
COMPANY_NAME = os.getenv("LOCAL_COMPANY_NAME") or ""
SHOP_ID = "Shop1-LineA"
# Background image for login overlay, expected at ./landing-bg.jpg next to this script
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent

LANDING_BG_PATH = BASE_DIR / "landing-bg.jpg"

# Supabase Auth configuration for login (anon/public key only)
SUPABASE_URL = os.getenv("LOCAL_SUPABASE_URL", "https://cynoykvmlktoqbmscccu.supabase.co").strip()
SUPABASE_ANON_KEY = os.getenv(
    "LOCAL_SUPABASE_ANON_KEY",
    os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_HDUPMB6Rt1veRr5hYAvWvQ__IsZ5CnF"),
).strip()

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase_client() -> Client:
    """Return a cached supabase-py client instance.

    Raises a RuntimeError with a helpful message if supabase-py is not
    installed or Supabase configuration is missing.
    """
    if create_client is None:
        raise RuntimeError(
            "The 'supabase-py' package is required for Supabase login. "
            "Install it with 'pip install supabase-py'."
        )
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError(
            "Supabase configuration is missing. Please set LOCAL_SUPABASE_URL and LOCAL_SUPABASE_ANON_KEY."
        )

    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _SUPABASE_CLIENT


# Simple in-memory session holder for this run only
@dataclass
class SupabaseSession:
    access_token: str
    refresh_token: Optional[str] = None


class LoginDialog(QDialog):
    """Simple dialog to log in via Supabase (no sign-up).

    Shown on top of the Local app with a dark theme so it feels seamless.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sign in to VisionM")
        self.resize(420, 220)
        self.session: Optional[SupabaseSession] = None

        # Make dialog feel like part of the app (dark theme, no help button)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #111111;
                border: none;
            }
            QLabel {
                color: #f5f5f5;
                border: none;
                font-size: 13px;
            }
            QLineEdit {
                background-color: #202020;
                border: none;
                border-radius: 6px;
                padding: 4px 8px;
                color: #f5f5f5;
            }
            QDialogButtonBox QPushButton {
                background-color: #202020;
                border: none;
                border-radius: 18px;
                padding: 6px 14px;
                color: #f5f5f5;
                font-weight: 500;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #2a2a2a;
            }
            QDialogButtonBox QPushButton:pressed {
                background-color: #1e1e1e;
            }
            """
        )

        layout = QVBoxLayout(self)

        info = QLabel("Log in with your VisionM account. New accounts must be created on the web portal.")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.email_edit = QLineEdit(self)
        self.email_edit.setPlaceholderText("you@example.com")
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Email:", self.email_edit)
        form.addRow("Password:", self.password_edit)
        layout.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:  # type: ignore[override]
        email = self.email_edit.text().strip()
        password = self.password_edit.text()

        if not email or not password:
            self.error_label.setText("Email and password are required.")
            return

        self.error_label.setText("")
        try:
            client = get_supabase_client()

            # Prefer modern supabase-py API, fall back to older sign_in if
            # sign_in_with_password is not available.
            try:
                res = client.auth.sign_in_with_password({"email": email, "password": password})
                session_obj = getattr(res, "session", None) or res
            except AttributeError:
                res = client.auth.sign_in(email=email, password=password)
                session_obj = getattr(res, "session", None) or res

            if session_obj is None:
                self.error_label.setText("Login failed: no session returned.")
                return

            access_token = getattr(session_obj, "access_token", None)
            refresh_token = getattr(session_obj, "refresh_token", None)
            if not access_token and isinstance(session_obj, dict):
                access_token = session_obj.get("access_token")
                refresh_token = session_obj.get("refresh_token")

            if not access_token:
                self.error_label.setText("Login failed: access token not returned.")
                return

            # Session is in-memory only; each app run requires a fresh login.
            self.session = SupabaseSession(access_token=access_token, refresh_token=refresh_token)
            super().accept()
        except Exception as exc:  # pragma: no cover - network / runtime errors
            self.error_label.setText(f"Login error: {exc}")


def ensure_supabase_session(parent: Optional[QWidget] = None) -> SupabaseSession:
    """Always prompt for a Supabase login for this run.

    We do not reuse any previous session; closing the app means re-login next time.
    """
    dlg = LoginDialog(parent)
    result = dlg.exec()
    if result != QDialog.Accepted or dlg.session is None:
        raise RuntimeError("Login was cancelled")
    return dlg.session


def fetch_projects_for_company(company: str, access_token: str) -> list[str]:
    """Fetch project names for the given company.

    Preferred source is Supabase (via supabase-py) so that the Local app can
    authorise and discover projects even when the VisionBackend API is not
    available. If Supabase lookup fails for any reason, we fall back to the
    /api/dashboard/projects endpoint on the backend.
    """
    company = (company or "").strip()
    if not company:
        raise RuntimeError(
            "Workspace/company name is empty; cannot fetch projects. "
            "This usually means your Supabase profile is not linked to a workspace."
        )

    # First try via Supabase if supabase-py and configuration are available.
    client: Optional[Client]
    try:
        client = get_supabase_client()
    except Exception:  # pragma: no cover - fall back to backend
        client = None

    if client is not None:
        try:
            # Resolve company id from companies table
            resp = client.table("companies").select("id").eq("name", company).execute()
            rows = getattr(resp, "data", None) or getattr(resp, "data", None)
            if isinstance(rows, list) and rows:
                company_id = rows[0].get("id")
            else:
                company_id = None

            project_names: list[str] = []
            if company_id is not None:
                proj_resp = (
                    client.table("projects")
                    .select("name")
                    .eq("company_id", company_id)
                    .execute()
                )
                proj_rows = getattr(proj_resp, "data", None) or getattr(proj_resp, "data", None)
                if isinstance(proj_rows, list):
                    for proj in proj_rows:
                        name = proj.get("name")
                        if isinstance(name, str) and name.strip():
                            project_names.append(name.strip())

            if project_names:
                return sorted(set(project_names))
        except Exception:  # pragma: no cover - Supabase project lookup failed
            log.debug("Supabase project lookup failed; falling back to backend.", exc_info=True)

    # Fallback: use VisionBackend /api/dashboard/projects
    if requests is None:
        raise RuntimeError("The 'requests' package is required to fetch projects from the backend")

    base = API_BASE_URL
    if not base:
        raise RuntimeError("LOCAL_API_BASE_URL / API_BASE_URL is not configured")

    url = base.rstrip("/") + "/dashboard/projects"
    headers = {
        "X-User-Id": "local-edge-client",       # synthetic ID for this Local instance
        "X-User-Role": "workspace_admin",       # role that can view projects
        "X-User-Company": company,               # workspace/company name from Supabase profile
    }
    resp = requests.get(url, params={"company": company}, headers=headers, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"Failed to fetch projects: {resp.status_code} {resp.text}")

    data = resp.json()
    raw_projects = data.get("projects") or []
    names: list[str] = []
    for proj in raw_projects:
        if not isinstance(proj, dict):
            continue
        name = proj.get("project")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    # Deduplicate and sort for stable display
    return sorted(set(names))


def fetch_company_from_supabase(access_token: str) -> str:
    """Derive the workspace/company name for the logged-in user via Supabase.

    Flow:
      1) GET /auth/v1/user to obtain the current user's id
      2) GET /rest/v1/profiles?id=eq.<id> to read company_id
      3) GET /rest/v1/companies?id=eq.<company_id> to read company name

    Returns the company name string used by VisionM (e.g. "RuzareInfoTech").
    Raises RuntimeError on failure.
    """
    if not access_token:
        raise RuntimeError("Access token is required to resolve workspace/company")

    client = get_supabase_client()

    # 1) Get current auth user from Supabase
    user_resp = client.auth.get_user(access_token)
    user_obj = getattr(user_resp, "user", None) or user_resp
    user_id = getattr(user_obj, "id", None)
    if not user_id and isinstance(user_obj, dict):
        user_id = user_obj.get("id")
    if not user_id:
        raise RuntimeError("Supabase user response did not include an id")

    # 2) Look up profile row for this user
    prof_resp = (
        client.table("profiles")
        .select("id,email,role,company_id")
        .eq("id", user_id)
        .execute()
    )
    profiles = getattr(prof_resp, "data", None) or getattr(prof_resp, "data", None)
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError("No profile row found for this user")
    profile = profiles[0]
    company_id = profile.get("company_id")
    if not company_id:
        raise RuntimeError("Profile does not have a company_id; user is not linked to a workspace")

    # 3) Resolve company name from companies table
    comp_resp = (
        client.table("companies")
        .select("name")
        .eq("id", company_id)
        .execute()
    )
    companies = getattr(comp_resp, "data", None) or getattr(comp_resp, "data", None)
    if not isinstance(companies, list) or not companies:
        raise RuntimeError("No company row found for this profile")
    company_name = companies[0].get("name")
    if not company_name or not isinstance(company_name, str):
        raise RuntimeError("Company row did not include a valid name")

    return company_name.strip()


class ProjectSelectionDialog(QDialog):
    """Dialog to let the user choose a project for this Local client."""

    def __init__(self, projects: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select VisionM project")
        self.resize(420, 260)
        self.selected_project: Optional[str] = None

        layout = QVBoxLayout(self)

        label = QLabel(
            f"Select a project for company '{COMPANY_NAME}'. This Local client will use this project "
            "for models and dashboards."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        self.list_widget = QListWidget(self)
        for name in projects:
            self.list_widget.addItem(QListWidgetItem(name))
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No selection", "Please select a project.")
            return
        item = self.list_widget.item(row)
        self.selected_project = item.text()
        self.accept()


def ensure_project_selection(access_token: str, parent: Optional[QWidget] = None) -> str:
    """Ensure a project is selected for this Local app run.

    If only one project exists, it is used automatically. Otherwise a dialog is shown.
    """
    projects = fetch_projects_for_company(COMPANY_NAME, access_token)
    if not projects:
        raise RuntimeError(f"No projects found for company '{COMPANY_NAME}'.")

    if len(projects) == 1:
        return projects[0]

    dlg = ProjectSelectionDialog(projects, parent=parent)
    result = dlg.exec()
    if result != QDialog.Accepted or not dlg.selected_project:
        raise RuntimeError("Project selection was cancelled")
    return dlg.selected_project


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
        self.setWindowTitle("Local  Live View")
        self.resize(1280, 720)

        # Auth / workspace context will be populated after login
        self._access_token: str | None = None
        self._company_name: str = COMPANY_NAME  # may be overridden from Supabase profile
        self._project_name: str = ""
        self._api_base_url = API_BASE_URL

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
                border: none;
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
                border: none;
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
        self.video_label.setStyleSheet("background-color: #202020; border: none;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 0, 0, 2, 1)
        self.dashboard_widget = QWidget(self)
        self.dashboard_widget.setObjectName("InfoPanel")
        self.dashboard_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        info_layout = QGridLayout(self.dashboard_widget)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setHorizontalSpacing(10)
        info_layout.setVerticalSpacing(8)
        project_name = self._project_name or "(no project selected)"
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
        # Backlog reconciliation will run after login/project selection when
        # we know the correct company/workspace for this client.

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

        # Live pipeline (camera + sync) is started only after login + project selection
        self._live_started: bool = False
        self._sync_timer: Optional[QTimer] = None
        # Guard to prevent overlapping background sync jobs
        self._sync_in_progress: bool = False
        # Track whether VisionBackend appears reachable; used to enable
        # an "offline" mode where Supabase login still works even if the
        # TRY/VisionBackend server is down.
        self._backend_available: bool = True

        # Build full-screen login overlay inspired by the web landing page
        self._build_login_overlay()

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
    # Login overlay (full-screen, web-landing inspired)
    # ------------------------------------------------------------------

    def _build_login_overlay(self) -> None:
        """Create a full-window login screen with landing-bg.jpg background.

        This keeps everything in a single window and visually matches the
        VisionM web landing page (hero text + centered login card).
        """
        self.login_overlay = QWidget(self)
        self.login_overlay.setObjectName("LoginOverlay")
        self.login_overlay.setGeometry(self.rect())
        self.login_overlay.setAttribute(Qt.WA_StyledBackground, True)

        bg_path = LANDING_BG_PATH.as_posix() if LANDING_BG_PATH.exists() else ""
        if bg_path:
            style = f"""
            QWidget#LoginOverlay {{
                background-color: #050816;
                background-image: url("{bg_path}");
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
            }}
            """
        else:
            style = """
            QWidget#LoginOverlay {
                background-color: #050816;
            }
            """
        self.login_overlay.setStyleSheet(style)

        outer = QVBoxLayout(self.login_overlay)
        # No outer margins so the background image and content truly fill the window
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        center = QWidget(self.login_overlay)
        center_layout = QVBoxLayout(center)
        center_layout.setSpacing(24)
        center_layout.setAlignment(Qt.AlignCenter)

        hero_title = QLabel(
            "Manage Your Dataset Projects <span style='color:#7c3aed;'>Efficiently</span>"
        )
        hero_title.setTextFormat(Qt.RichText)
        hero_title.setAlignment(Qt.AlignCenter)
        hero_title.setStyleSheet(
            "font-size: 34px; font-weight: 800; color: #f9fafb; letter-spacing: 0.5px;"
        )

        hero_sub = QLabel(
            "VisionM helps teams collaborate on computer vision datasets with secure "
            "project management, workspace controls, and seamless file uploads."
        )
        hero_sub.setWordWrap(True)
        hero_sub.setAlignment(Qt.AlignCenter)
        hero_sub.setStyleSheet(
            "font-size: 15px; color: rgba(249,250,251,0.9); max-width: 720px;"
        )

        card = QWidget(center)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(12)
        card.setStyleSheet(
            "background-color: rgba(15,23,42,0.92); "
            "border-radius: 16px; "
            "border: 1px solid rgba(148,163,184,0.45);"
        )

        logo = QLabel("VisionM")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "font-size: 24px; font-weight: 700; letter-spacing: 0.08em; color: #a855f7; border: none;"
        )

        card_title = QLabel("Sign in to continue")
        card_title.setAlignment(Qt.AlignCenter)
        card_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #e5e7eb; border: none;")

        hint = QLabel("Use the same email and password as on the VisionM web platform.")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("font-size: 12px; color: #9ca3af; border: none;")

        self.login_email_edit = QLineEdit(card)
        self.login_email_edit.setPlaceholderText("you@example.com")
        self.login_password_edit = QLineEdit(card)
        self.login_password_edit.setPlaceholderText("Password")
        self.login_password_edit.setEchoMode(QLineEdit.Password)

        for w in (self.login_email_edit, self.login_password_edit):
            w.setStyleSheet(
                "background-color: rgba(15,23,42,0.9);"
                "border-radius: 10px;"
                "border: none;"
                "padding: 10px 14px;"
                "min-height: 44px;"
                "font-size: 15px;"
                "color: #f9fafb;"
            )

        self.login_button = QPushButton("Sign In")
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setStyleSheet(
            "background-color: #4f46e5; color: white; font-weight: 600; "
            "border-radius: 999px; padding: 8px 18px;"
        )
        self.login_button.clicked.connect(self._on_login_clicked)

        self.login_status_label = QLabel("")
        self.login_status_label.setWordWrap(True)
        self.login_status_label.setAlignment(Qt.AlignCenter)
        self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")

        self.project_area = QWidget(card)
        self.project_area.setVisible(False)
        project_layout = QVBoxLayout(self.project_area)
        project_layout.setContentsMargins(0, 8, 0, 0)
        project_layout.setSpacing(6)

        # Label uses dynamic company name which we update after login
        self.project_label = QLabel("")
        self.project_label.setAlignment(Qt.AlignCenter)
        self.project_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")

        self.project_combo = QComboBox(self.project_area)
        self.project_combo.setStyleSheet(
            "background-color: rgba(15,23,42,0.9);"
            "border-radius: 10px;"
            "border: none;"
            "padding: 8px 12px;"
            "min-height: 40px;"
            "font-size: 15px;"
            "color: #f9fafb;"
        )

        self.project_continue_button = QPushButton("Continue")
        self.project_continue_button.setCursor(Qt.PointingHandCursor)
        self.project_continue_button.setStyleSheet(
            "background-color: #22c55e; color: black; font-weight: 600; "
            "border-radius: 999px; padding: 6px 16px;"
        )
        self.project_continue_button.clicked.connect(self._on_project_continue_clicked)

        project_buttons = QHBoxLayout()
        project_buttons.setContentsMargins(0, 0, 0, 0)
        project_buttons.addStretch()
        project_buttons.addWidget(self.project_continue_button)
        project_buttons.addStretch()

        project_layout.addWidget(self.project_label)
        project_layout.addWidget(self.project_combo)
        project_layout.addLayout(project_buttons)

        card_layout.addWidget(logo)
        card_layout.addWidget(card_title)
        card_layout.addWidget(hint)
        card_layout.addSpacing(4)
        card_layout.addWidget(self.login_email_edit)
        card_layout.addWidget(self.login_password_edit)
        card_layout.addWidget(self.login_button)
        card_layout.addWidget(self.login_status_label)
        card_layout.addWidget(self.project_area)

        center_layout.addWidget(hero_title)
        center_layout.addWidget(hero_sub)
        center_layout.addSpacing(16)
        center_layout.addWidget(card)

        outer.addWidget(center, alignment=Qt.AlignCenter)
        outer.addStretch()

        self.login_overlay.raise_()
        self.login_overlay.show()

    def _on_login_clicked(self) -> None:
        """Handle Supabase email/password login from the overlay."""
        email = self.login_email_edit.text().strip()
        password = self.login_password_edit.text()
        if not email or not password:
            self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
            self.login_status_label.setText("Email and password are required.")
            return

        self.login_status_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
        self.login_status_label.setText("Signing in...")

        try:
            client = get_supabase_client()

            # Prefer modern supabase-py API, fall back to older sign_in if
            # sign_in_with_password is not available.
            try:
                res = client.auth.sign_in_with_password({"email": email, "password": password})
                session_obj = getattr(res, "session", None) or res
            except AttributeError:
                res = client.auth.sign_in(email=email, password=password)
                session_obj = getattr(res, "session", None) or res

            if session_obj is None:
                self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
                self.login_status_label.setText("Login failed: no session returned.")
                return

            access_token = getattr(session_obj, "access_token", None)
            if not access_token and isinstance(session_obj, dict):
                access_token = session_obj.get("access_token")

            if not access_token:
                self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
                self.login_status_label.setText("Login failed: access token not returned.")
                return

            self._access_token = access_token
            self.login_status_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
            self.login_status_label.setText("Logged in. Loading workspace and projects...")
            self._load_company_and_projects()
        except Exception as exc:
            self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
            self.login_status_label.setText(f"Login error: {exc}")

    def _load_company_and_projects(self) -> None:
        """Resolve company from Supabase then load projects for that workspace."""
        try:
            company = fetch_company_from_supabase(self._access_token or "")
            self._company_name = company
            # Also expose workspace to any code that still relies on env vars
            os.environ["LOCAL_COMPANY_NAME"] = self._company_name

            # Update project label now that we know the workspace
            if hasattr(self, "project_label") and self.project_label is not None:
                self.project_label.setText(
                    f"Select project for company '{self._company_name}'"
                )
        except Exception as exc:
            self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
            self.login_status_label.setText(f"Failed to determine workspace: {exc}")
            return

        self._load_projects_for_login()

    def _load_projects_for_login(self) -> None:
        """Fetch projects for the resolved company and show selector if needed.

        If the VisionBackend is not reachable, we fall back to an **offline
        mode**: the user remains logged in via Supabase, local models (.pt
        files) and detection stats from local_stats.json / local_stats.db
        continue to work, and detections will be synced automatically once
        the backend becomes available again.
        """
        try:
            projects = fetch_projects_for_company(self._company_name, self._access_token or "")
            self._backend_available = True
        except Exception as exc:
            # Backend is down or unreachable – allow login to succeed and
            # start the Local app in offline mode.
            self._backend_available = False
            self._project_name = "(offline)"
            self.login_status_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
            self.login_status_label.setText(
                "VisionBackend is not reachable. Starting in offline mode: "
                "detections will be stored locally and synced when the "
                "connection is available."
            )
            self._finalise_login_and_start()
            return

        if not projects:
            self.login_status_label.setStyleSheet("font-size: 12px; color: #f97373;")
            self.login_status_label.setText(
                f"No projects found for company '{self._company_name}'."
            )
            return

        if len(projects) == 1:
            self._project_name = projects[0]
            self._finalise_login_and_start()
            return

        self.project_combo.clear()
        for name in projects:
            self.project_combo.addItem(name)
        self.project_area.setVisible(True)
        self.login_status_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
        self.login_status_label.setText(
            "Select a project for this Local client and click Continue."
        )

    def _on_project_continue_clicked(self) -> None:
        idx = self.project_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "No selection", "Please select a project.")
            return
        name = self.project_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "Invalid selection", "Please select a valid project.")
            return
        self._project_name = name
        self._finalise_login_and_start()

    def _finalise_login_and_start(self) -> None:
        """Show a loading state and start live pipeline once login + project are set.

        We also populate LOCAL_COMPANY_NAME / LOCAL_PROJECT_NAME so any
        legacy code or scripts that still read these env vars continue to
        work without needing a long shell invocation.

        The initial camera startup + first full sync can take a little while,
        especially when there is a large local backlog. To avoid the app
        looking "frozen", we keep the full-screen overlay visible with a
        loading message while this work runs, then hide it when done.
        """
        # Keep env in sync with the chosen workspace + project
        if self._company_name:
            os.environ["LOCAL_COMPANY_NAME"] = self._company_name
        if self._project_name:
            os.environ["LOCAL_PROJECT_NAME"] = self._project_name

        self.project_value.setText(self._project_name or "(no project selected)")

        # Reuse the login overlay as a simple loading screen during startup
        if hasattr(self, "login_email_edit"):
            self.login_email_edit.setEnabled(False)
        if hasattr(self, "login_password_edit"):
            self.login_password_edit.setEnabled(False)
        if hasattr(self, "login_button"):
            self.login_button.setEnabled(False)
        if hasattr(self, "project_area"):
            self.project_area.setVisible(False)

        if hasattr(self, "login_status_label") and self.login_status_label is not None:
            self.login_status_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
            self.login_status_label.setText(
                "Starting camera and syncing local episodes to VisionM...\n"
                "This may take up to a minute on first run."
            )

        # Ensure overlay covers the window while heavy initial work runs
        self.login_overlay.setGeometry(self.rect())
        self.login_overlay.show()
        self.login_overlay.raise_()

        # Defer heavy startup work slightly so the UI can paint the loading state
        QTimer.singleShot(0, self._start_live_pipeline_and_close_overlay)

    def _start_live_pipeline_and_close_overlay(self) -> None:
        """Wrapper that starts the live pipeline then hides the loading overlay.

        The heavy work (initial sync) runs in a background thread so the UI
        stays responsive. We keep the overlay visible briefly so the user
        sees the loading state, then reveal the main UI.
        """
        self._start_live_pipeline()
        if hasattr(self, "login_overlay"):
            def _hide_overlay() -> None:
                # Called back on the GUI thread via QTimer
                self.login_overlay.hide()

            # Give the overlay a short moment to be visible
            QTimer.singleShot(1500, _hide_overlay)

    def _start_live_pipeline(self) -> None:
        if self._live_started:
            return
        self._live_started = True

        # Start MJPEG HTTP server (if enabled) and spawn live worker + sync timer
        start_stream_server_if_enabled()
        self._start_worker_with_current_model()

        # Immediately reconcile any backlog of episodes now that we know the
        # company/project. The reconciliation itself schedules a background
        # sync job so the GUI thread is not blocked.
        try:
            self._drain_unsynced_episodes()
        except Exception:
            log.exception("Initial backlog reconciliation failed")

        # Periodic sync, also in the background
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(60_000)
        self._sync_timer.timeout.connect(self._sync_to_cloud_background)
        self._sync_timer.start()

    # ------------------------------------------------------------------
    # Model browser / download from VisionM backend
    # ------------------------------------------------------------------
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

        api_base = self._api_base_url.strip()
        if not api_base:
            QMessageBox.warning(
                self,
                "Missing API URL",
                "Set LOCAL_API_BASE_URL to your VisionBackend base URL.",
            )
            return

        company = self._company_name
        project = self._project_name or self.project_value.text().strip()
        if not company or not project:
            QMessageBox.warning(
                self,
                "Missing details",
                "Company or project is not set. Please ensure login and project selection completed.",
            )
            return

        try:
            # 1) List models for this company + project
            list_url = api_base.rstrip("/") + "/models"
            params = {"company": company, "project": project}
            headers = {}
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            resp = requests.get(list_url, params=params, headers=headers, timeout=10)
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
        """Legacy helper (no-op now that we always send all episodes).

        Kept for compatibility but no longer used.
        """
        return None, None

    def _update_sync_state(self, last_synced_ts: Optional[str]) -> None:
        """Legacy helper (no-op now that we always send all episodes)."""
        return

    def _reset_sync_cursor(self) -> None:
        """Legacy helper (no-op) – sync cursor is no longer tracked locally."""
        return

    def _edge_log(self, level: str, event_type: str, message: str) -> None:
        """Send a lightweight edge connectivity/sync event to VisionM.

        This is best-effort only; failures are logged but ignored.
        """
        if requests is None:
            return

        api_base = self._api_base_url.strip()
        if not api_base:
            return

        company_name = self._company_name or None
        shop_id = SHOP_ID

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

        # Trigger a cloud sync in the background whenever a new episode is
        # recorded so that the VisionM UI updates shortly after local_stats
        # is written. The actual work runs off the GUI thread, so the UI
        # remains responsive even if there is network delay.
        try:
            self._sync_to_cloud_background()
        except Exception:
            # Never let sync scheduling errors break the UI loop
            log.exception("_sync_to_cloud_background failed after logging episode")

    def _drain_unsynced_episodes(self) -> None:
        """On startup, push whatever episodes exist in SQLite to the backend.

        We no longer track a local "last_synced_ts" cursor; the backend
        upserts on (company, shopId, ts, result), so resending episodes is
        safe. This helper is now just a thin wrapper around `_sync_to_cloud`.
        """
        if requests is None:
            return

        api_base = self._api_base_url.strip()
        if not api_base:
            return

        # Run the actual sync in the background so the GUI thread is not
        # blocked during a large initial upload.
        self._sync_to_cloud_background()

        # Log that Local came online and reconciliation was requested
        # (best-effort; the sync job itself logs its own outcome).
        try:
            self._edge_log("info", "online", "Local app started and reconciliation scheduled")
        except Exception:
            pass

    def _sync_to_cloud_background(self) -> None:
        """Run `_sync_to_cloud` in a background thread to keep the UI responsive."""
        if requests is None:
            return
        if self._sync_in_progress:
            # Avoid overlapping jobs when timer/events fire quickly
            log.debug("SYNC: background job already in progress; skipping new request")
            return

        self._sync_in_progress = True
        log.info("SYNC: scheduling background sync_to_cloud job")

        def _worker() -> None:
            try:
                self._sync_to_cloud()
            except Exception:
                # _sync_to_cloud already logs details; this is a safety net
                log.exception("SYNC: unexpected error in background _sync_to_cloud")
            finally:
                # Reset the flag back on the GUI thread
                def _clear_flag() -> None:
                    self._sync_in_progress = False
                QTimer.singleShot(0, _clear_flag)

        threading.Thread(target=_worker, name="sync-to-cloud", daemon=True).start()

    def _sync_to_cloud(self) -> None:
        """Best-effort push of episodes to the VisionM backend (blocking).

        This method performs the actual sync work and is typically invoked
        via `_sync_to_cloud_background` so that network and I/O do not block
        the GUI thread.

        Configuration:
        - LOCAL_API_BASE_URL (e.g. "http://192.168.1.24:3000/api")
        - LOCAL_COMPANY_NAME (must match VisionM workspace/company name)
        - LOCAL_SHOP_ID (identifier for this local client / line)

        Episodes are read in *batches* from SQLite so the HTTP payload stays
        small and we avoid 413 Payload Too Large errors when there is a
        backlog.
        """
        log.info("SYNC: starting sync_to_cloud (blocking)")
        if requests is None:
            # HTTP client not available – skip silently
            return

        api_base = self._api_base_url.strip()
        if not api_base:
            return

        # Do not attempt to sync until we know which workspace/company this
        # Local instance belongs to (i.e. after Supabase login).
        company_name = (self._company_name or "").strip()
        if not company_name:
            log.info("SYNC: skipping because company is not set yet (not logged in)")
            return

        shop_id = SHOP_ID

        # Always read all episodes from SQLite and let the backend upsert on
        # (company, shopId, ts, result). This avoids complicated local cursor
        # logic and keeps behaviour simple when switching workspaces.
        try:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT ts, result FROM episodes ORDER BY id ASC")
                rows = cur.fetchall()
        except Exception as exc:
            log.error("SYNC: failed to read episodes from SQLite: %s", exc)
            return

        if not rows:
            log.info("SYNC: no episodes found in SQLite; nothing to sync")
            return

        # Send episodes in bounded batches to avoid 413 Payload Too Large.
        BATCH_LIMIT = 200
        url = api_base.rstrip("/") + "/local-sync/upload"
        total_sent = 0

        for i in range(0, len(rows), BATCH_LIMIT):
            chunk = rows[i : i + BATCH_LIMIT]
            new_episodes = [{"ts": ts, "result": result} for (ts, result) in chunk]
            payload = {
                "company": company_name,
                "shopId": shop_id,
                "episodes": new_episodes,
            }

            try:
                resp = requests.post(url, json=payload, timeout=5)
                resp.raise_for_status()
                total_sent += len(new_episodes)
                log.info(
                    "SYNC: posted batch %d-%d of %d episodes (batch size %d), status=%s",
                    i + 1,
                    i + len(new_episodes),
                    len(rows),
                    len(new_episodes),
                    resp.status_code,
                )
            except Exception as exc:
                log.error("Failed to sync episodes batch to VisionM backend: %s", exc)
                # Log sync failure as edge event (but don't raise)
                self._edge_log("error", "sync_error", f"Sync batch failed: {exc}")
                return

        # Record a connectivity log summarising the multi-batch sync.
        self._edge_log(
            "info",
            "sync_ok",
            f"Synced {total_sent} episodes in batches of up to {BATCH_LIMIT}",
        )

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

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "login_overlay") and self.login_overlay.isVisible():
            self.login_overlay.setGeometry(self.rect())

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

    if requests is None:
        logging.error("The 'requests' package is required for login and backend communication.")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Local")

    # Single window: MainWindow always starts on the full-screen login overlay.
    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
