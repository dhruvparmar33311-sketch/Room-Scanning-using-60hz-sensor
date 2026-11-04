
"""
RADAR SCAN STUDIO
  LEFT : one dot per radar reading (grey dot = reading the sensor missed).
         Hover a dot to see its X / Y / Z coordinates.
  RIGHT: when the scan completes, a seamless CAD-style 3D model of the room is built
         from those same points. Missed readings are filled in from their neighbours
         and drawn GREY, so the model has no holes.

ESP32 serial input (115200 baud), one point per line:  <step>,<servo_angle>,<distance_mm>
Lines starting with '#' are ESP32 status messages (#SCAN_START, #ROW,.., #SCAN_END).

Install:   pip install pyqtgraph PyQt5 PyOpenGL numpy pyserial
Run:       python radar_scan_studio.py              (COM10)
           python radar_scan_studio.py --port COM5

Keys:  SPACE start scan | X clear | B build 3D model now | T change 3D style
       P export CSV | R reset 3D view
3D mouse: left-drag orbit | middle-drag pan | wheel zoom
Left plot: wheel = zoom in to separate the dots, drag = pan

Coordinates: scanner is the origin. +X right, +Y forward, +Z up (millimetres).

Close the Arduino Serial Monitor first - only one program can use COM10.
Do not name your own files serial.py, numpy.py or pyqtgraph.py.
"""

import argparse
import csv
import math
import queue
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import serial
from numpy.lib.stride_tricks import sliding_window_view
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ============================ CONFIG ============================
DEFAULT_PORT = "COM10"
DEFAULT_BAUD = 115200

# Scan grid - must match the ESP32 sketch (radar_scanner_ltr.ino)
SERVO_FIRST, SERVO_LAST, SERVO_STEP = 130, 170, 5
STEP_FIRST, STEP_LAST, STEP_SPACING = -512, 511, 3
STEPS_PER_REV = 2048

SERVO_LEVEL_DEG = 90           # servo angle where the radar looks level
PAN_SIGN = 1.0                 # -1.0 if the room looks mirrored left/right
TILT_SIGN = 1.0                # -1.0 if the room looks flipped up/down

MIN_VALID_MM = 10.0
MAX_VALID_MM = 20000.0
MIN_POINTS_FOR_MODEL = 50
GREY_IF_MISSING_CORNERS = 2   # a surface patch is grey if at least this many of its 4 corners were missed (1-4)
MESH_MEDIAN = 3               # odd; smooths radar noise along each row of the 3D model (1 = off)

DOT_SIZE = 5                   # pixels (zoom the left plot to separate the dots)
HOVER_RADIUS_PX = 14
REFRESH_HZ = 6

STYLES = ["CONCEPTUAL", "WIREFRAME", "DEPTH"]
COL_STEEL = np.array([0.40, 0.65, 0.92])     # measured surfaces (CAD blue)
COL_GREY = np.array([0.62, 0.62, 0.64])      # missed readings
MODEL_BG = (30, 36, 44)
# ================================================================

ROWS = list(range(SERVO_FIRST, SERVO_LAST + 1, SERVO_STEP))
NROW = len(ROWS)
NCOL = (STEP_LAST - STEP_FIRST) // STEP_SPACING + 1
STEPS = STEP_FIRST + STEP_SPACING * np.arange(NCOL)
DEG_PER_STEP = 360.0 / STEPS_PER_REV
HAS_TEXT = hasattr(gl, "GLTextItem")

_STOPS_T = np.array([0.0, 0.5, 1.0])
_STOPS_RGB = np.array([[0.30, 0.00, 0.03], [1.00, 0.05, 0.04], [1.00, 0.62, 0.12]])
N_BR = 48


# ------------------------- maths -------------------------
def to_cartesian(steps, servo, dist):
    """+Y forward (step 0), +X right, +Z up. Servo above level tilts DOWN. Units: mm."""
    pan = np.radians(np.asarray(steps, float) * DEG_PER_STEP * PAN_SIGN)
    tilt = np.radians((np.asarray(servo, float) - SERVO_LEVEL_DEG) * TILT_SIGN)
    dist = np.asarray(dist, float)
    ct = np.cos(tilt)
    return np.column_stack((dist * ct * np.sin(pan),
                            dist * ct * np.cos(pan),
                            -dist * np.sin(tilt))).astype(np.float32)


def thermal_rgb(t):
    """t in 0..1 (1 = near/hot). Returns array (...,3) 0..1."""
    t = np.asarray(t, float)
    return np.stack([np.interp(t, _STOPS_T, _STOPS_RGB[:, c]) for c in range(3)], axis=-1)


def dist_to_t(d, near, far):
    return np.clip((far - d) / max(far - near, 1.0), 0.0, 1.0)


def cell_of(step, servo):
    """(row, col) of a reading in the scan grid, or None if it does not belong to it."""
    r = int(round((servo - SERVO_FIRST) / SERVO_STEP))
    c = int(round((step - STEP_FIRST) / STEP_SPACING))
    if not (0 <= r < NROW and 0 <= c < NCOL):
        return None
    if abs(servo - (SERVO_FIRST + r * SERVO_STEP)) > 0.5:
        return None
    if abs(step - (STEP_FIRST + c * STEP_SPACING)) > 0.5:      # must sit exactly on the grid
        return None
    return r, c


def _directional(values, valid, axis_deg):
    """Estimate every cell from the nearest valid cells to its left/right (along axis 1)."""
    n_r, n_c = values.shape
    cols = np.tile(np.arange(n_c), (n_r, 1))
    rr = np.tile(np.arange(n_r)[:, None], (1, n_c))
    left = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
    right = np.minimum.accumulate(np.where(valid, cols, n_c)[:, ::-1], axis=1)[:, ::-1]
    has_l, has_r = left >= 0, right < n_c
    vl = values[rr, np.clip(left, 0, n_c - 1)]
    vr = values[rr, np.clip(right, 0, n_c - 1)]
    a = (cols - left).astype(float)
    b = (right - cols).astype(float)
    est = np.full(values.shape, np.nan)
    gap = np.full(values.shape, np.inf)
    both = has_l & has_r
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = (vl * b + vr * a) / (a + b)
    est[both] = interp[both]
    gap[both] = (a[both] + b[both]) * axis_deg
    only_l = has_l & ~has_r
    est[only_l] = vl[only_l]
    gap[only_l] = 3.0 * a[only_l] * axis_deg          # extrapolation is trusted less
    only_r = ~has_l & has_r
    est[only_r] = vr[only_r]
    gap[only_r] = 3.0 * b[only_r] * axis_deg
    return est, gap


def fill_missing(dist):
    """dist (R,C) with NaN where the sensor missed a reading.
    Returns (filled, missing_mask). Gaps are filled from the neighbours along the row AND the
    column, weighted by how close the nearest real readings are. Returns (None, mask) if empty."""
    valid = ~np.isnan(dist)
    missing = ~valid
    if not valid.any():
        return None, missing
    if not missing.any():
        return dist.astype(float).copy(), missing
    col_deg = STEP_SPACING * DEG_PER_STEP
    est_h, gap_h = _directional(dist, valid, col_deg)
    est_vt, gap_vt = _directional(dist.T, valid.T, float(SERVO_STEP))
    est_v, gap_v = est_vt.T, gap_vt.T
    with np.errstate(divide="ignore", invalid="ignore"):
        wh = np.where(np.isfinite(gap_h) & (gap_h > 0), 1.0 / gap_h, 0.0)
        wv = np.where(np.isfinite(gap_v) & (gap_v > 0), 1.0 / gap_v, 0.0)
    den = wh + wv
    num = wh * np.nan_to_num(est_h) + wv * np.nan_to_num(est_v)
    filled = dist.astype(float).copy()
    ok = missing & (den > 0)
    filled[ok] = num[ok] / den[ok]
    left = np.isnan(filled)
    if left.any():
        filled[left] = float(np.nanmedian(dist))
    return filled, missing


def smooth_rows(grid, win):
    if win <= 1:
        return grid
    pad = win // 2
    p = np.pad(grid, ((0, 0), (pad, pad)), mode="edge")
    return np.median(sliding_window_view(p, win, axis=1), axis=-1)


def build_mesh(filled, missing):
    """Seamless surface through every grid cell. Returns (xyz, faces, face_missing, face_dist)."""
    R, C = filled.shape
    S, V = np.meshgrid(STEPS.astype(float), np.array(ROWS, float))
    xyz = to_cartesian(S.ravel(), V.ravel(), filled.ravel())
    idx = np.arange(R * C).reshape(R, C)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    n_miss = (missing[:-1, :-1].astype(int) + missing[:-1, 1:] + missing[1:, :-1] + missing[1:, 1:])
    m = (n_miss >= GREY_IF_MISSING_CORNERS).ravel()          # grey = mostly estimated
    dm = ((filled[:-1, :-1] + filled[:-1, 1:] + filled[1:, :-1] + filled[1:, 1:]) / 4.0).ravel()
    return xyz, faces, np.concatenate((m, m)), np.concatenate((dm, dm))


def face_shade(xyz, faces):
    """CAD-like facet shading: a light at the scanner plus a light from above (sign-free)."""
    v0, v1, v2 = (xyz[faces[:, i]].astype(float) for i in range(3))
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-6
    n = n / np.where(ok, ln, 1.0)[:, None]
    cen = (v0 + v1 + v2) / 3.0
    lc = np.linalg.norm(cen, axis=1)
    l1 = cen / np.where(lc > 1e-6, lc, 1.0)[:, None]
    s1 = np.abs(np.einsum("ij,ij->i", n, l1))
    s2 = np.abs(n[:, 2])
    shade = np.where(ok, 0.30 + 0.55 * s1 + 0.20 * s2, 0.5)
    return np.clip(shade, 0.25, 1.0).astype(np.float32)


def grid_lines(xyz, R, C, col_every):
    """Line segments (pairs of points) along every row and every col_every-th column."""
    P = xyz.reshape(R, C, 3)
    ra, rb = P[:, :-1].reshape(-1, 3), P[:, 1:].reshape(-1, 3)
    row_seg = np.stack((ra, rb), axis=1).reshape(-1, 3)
    cols = sorted(set(list(range(0, C, col_every)) + [C - 1]))
    ca, cb = P[:-1, cols].reshape(-1, 3), P[1:, cols].reshape(-1, 3)
    col_seg = np.stack((ca, cb), axis=1).reshape(-1, 3)
    return np.vstack((row_seg, col_seg)).astype(np.float32)


# ------------------------- serial -------------------------
class SerialReader(threading.Thread):
    """Reads the COM port in a background thread, validates lines, auto-reconnects."""

    def __init__(self, port, baud, out_queue):
        super().__init__(daemon=True)
        self.port, self.baud, self.q = port, baud, out_queue
        self.stop_event = threading.Event()
        self._ser = None
        self._lock = threading.Lock()

    def send(self, data: bytes) -> bool:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.write(data)
                    return True
                except (serial.SerialException, OSError):
                    return False
        return False

    def run(self):
        while not self.stop_event.is_set():
            ser = None
            try:
                ser = serial.Serial()
                ser.port = self.port
                ser.baudrate = self.baud
                ser.timeout = 0.1
                ser.dtr = False
                ser.rts = False
                ser.open()
                with self._lock:
                    self._ser = ser
                ser.reset_input_buffer()
                self.q.put(("msg", f"Connected to {self.port} @ {self.baud}"))
                buf = bytearray()
                while not self.stop_event.is_set():
                    chunk = ser.read(max(1, ser.in_waiting))
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if len(buf) > 65536:
                        del buf[:-1024]
                    while True:
                        i = buf.find(b"\n")
                        if i < 0:
                            break
                        line = bytes(buf[:i])
                        del buf[:i + 1]
                        self._handle(line)
            except (serial.SerialException, OSError) as e:
                self.q.put(("msg", f"Serial error: {e} (retrying in 2 s)"))
                self.stop_event.wait(2.0)
            finally:
                with self._lock:
                    self._ser = None
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass

    def _handle(self, raw: bytes):
        text = raw.decode("ascii", errors="ignore").strip()
        if not text:
            return
        if text.startswith("#"):
            self.q.put(("msg", text))
            return
        parts = text.split(",")
        if len(parts) != 3:
            return
        try:
            step, servo, dist = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return
        if not (math.isfinite(dist) and MIN_VALID_MM <= dist <= MAX_VALID_MM):
            return
        self.q.put(("pt", (step, servo, dist, text)))


# ------------------------- main window -------------------------
ACCENT = "#4fc3f7"
STYLE = f"""
QMainWindow, QWidget {{ background: #0b0e14; color: #d6e2ee;
                       font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }}
QPushButton {{ background: #121a26; border: 1px solid {ACCENT}; padding: 6px 12px; color: {ACCENT}; }}
QPushButton:hover {{ background: #1b2a3d; }}
QPushButton:pressed {{ background: {ACCENT}; color: #0b0e14; }}
QLabel#head {{ color: {ACCENT}; font-weight: bold; padding: 4px 6px; }}
QLabel#hover {{ font-family: Consolas, 'Courier New', monospace; padding: 4px 6px;
                border-top: 1px solid #1b2a3d; }}
QLabel#status {{ padding: 5px 10px; border-top: 1px solid #1b2a3d; }}
QLabel#placeholder {{ color: #6f8499; font-size: 15px; }}
QSplitter::handle {{ background: #1b2a3d; }}
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, port, baud):
        super().__init__()
        self.setWindowTitle("RADAR SCAN STUDIO")
        self.resize(1500, 880)
        self.setStyleSheet(STYLE)

        # ---- scan state ----
        self.dist = np.full((NROW, NCOL), np.nan, dtype=np.float32)
        self.recv = np.zeros((NROW, NCOL), dtype=bool)
        self.last_col = np.full(NROW, -1, dtype=int)
        self.cur_row = -1
        self.scan_active = False
        self.scan_done = False
        self.conn_text = "CONNECTING..."
        self.dirty = False
        self.next_refresh = 0.0
        self.snap_warned = False
        self.filled = None            # gap-filled distances (after the model is built)
        self.geom = None
        self.model_items = []
        self.style_idx = 0
        self.shown = np.zeros(NROW * NCOL, dtype=bool)
        self.n_missed = 0

        S, V = np.meshgrid(STEPS.astype(float), np.array(ROWS, float))
        self.cell_x = (S * DEG_PER_STEP * PAN_SIGN).ravel()
        self.cell_y = ((V - SERVO_LEVEL_DEG) * TILT_SIGN).ravel()

        self.brush_lut = []
        for k in range(N_BR):
            r, g, b = (thermal_rgb(k / (N_BR - 1)) * 255).astype(int)
            self.brush_lut.append(pg.mkBrush(int(r), int(g), int(b), 255))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"scan_log_{stamp}.csv"
        self.log = open(self.log_path, "w", buffering=1)
        self.log.write("step,servo_deg,dist_mm\n")

        self._build_ui()
        self.reset_view()

        self.q = queue.Queue()
        self.reader = SerialReader(port, baud, self.q)
        self.reader.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)
        print(f"[*] Logging raw data to {self.log_path}")
        print("[*] SPACE scan | X clear | B build 3D | T style | P export CSV | R reset 3D view")

    # ---------- UI ----------
    def _button(self, bar, text, key, fn):
        b = QtWidgets.QPushButton(text)
        b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        b.setShortcut(QtGui.QKeySequence(key))
        b.clicked.connect(fn)
        bar.addWidget(b)

    def _axis_style(self, plot):
        for ax in ("bottom", "left"):
            a = plot.getAxis(ax)
            a.setPen(pg.mkPen("#6f8499"))
            a.setTextPen(pg.mkPen("#9db4c8"))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        self._button(bar, "SCAN [Space]", "Space", self.start_scan)
        self._button(bar, "CLEAR [X]", "X", self.clear)
        self._button(bar, "BUILD 3D [B]", "B", self.build_model)
        self._button(bar, "STYLE [T]", "T", self.next_style)
        self._button(bar, "EXPORT CSV [P]", "P", self.export_csv)
        self._button(bar, "RESET 3D VIEW [R]", "R", self.reset_view)
        bar.addStretch(1)
        outer.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---- left: scan dots ----
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(6, 0, 3, 0)
        lv.setSpacing(0)
        head = QtWidgets.QLabel("SCAN POINTS   -   hover a dot to read its X / Y / Z   (grey = missed reading)")
        head.setObjectName("head")
        lv.addWidget(head)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#0b0e14")
        self.plot = self.plot_widget.getPlotItem()
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "PAN angle (deg)   left < 0 < right")
        self.plot.setLabel("left", "TILT (deg, down +)")
        self._axis_style(self.plot)
        xs, ys = self.cell_x, self.cell_y
        self.plot.setXRange(float(xs.min()) - 3, float(xs.max()) + 3, padding=0)
        self.plot.setYRange(float(ys.min()) - 4, float(ys.max()) + 4, padding=0)
        self.recv_scatter = pg.ScatterPlotItem(pxMode=True, size=DOT_SIZE, symbol="o", pen=pg.mkPen(None))
        self.miss_scatter = pg.ScatterPlotItem(pxMode=True, size=DOT_SIZE, symbol="o", pen=pg.mkPen(None))
        self.hl = pg.ScatterPlotItem(pxMode=True, size=16, symbol="o",
                                     pen=pg.mkPen("w", width=2), brush=pg.mkBrush(0, 0, 0, 0))
        for it in (self.miss_scatter, self.recv_scatter, self.hl):
            self.plot.addItem(it)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        lv.addWidget(self.plot_widget, 1)
        self.hover_label = QtWidgets.QLabel("Hover over a dot to see its coordinates")
        self.hover_label.setObjectName("hover")
        lv.addWidget(self.hover_label)

        # ---- right: 3D model ----
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(3, 0, 6, 0)
        rv.setSpacing(0)
        self.model_head = QtWidgets.QLabel("3D MODEL   (CAD view, built from the points on the left)")
        self.model_head.setObjectName("head")
        rv.addWidget(self.model_head)
        self.stack = QtWidgets.QStackedWidget()
        self.placeholder = QtWidgets.QLabel("The 3D model appears here when the scan completes.\n"
                                            "(Press B to build it from the points received so far.)")
        self.placeholder.setObjectName("placeholder")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor(MODEL_BG)
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.gl)
        rv.addWidget(self.stack, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([650, 850])
        outer.addWidget(split, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        outer.addWidget(self.status)
        self.setCentralWidget(central)

    # ---------- hover ----------
    def point_xyz(self, step, servo, d):
        p = to_cartesian([step], [servo], [d])[0]
        return float(p[0]), float(p[1]), float(p[2])

    def clear_hover(self):
        QtWidgets.QToolTip.hideText()
        self.hl.clear()
        self.hover_label.setText("Hover over a dot to see its coordinates")

    def on_mouse_moved(self, scene_pos):
        if not self.shown.any():
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self.clear_hover()
            return
        mp = vb.mapSceneToView(scene_pos)
        pw, ph = vb.viewPixelSize()
        if pw <= 0 or ph <= 0:
            return
        dx = (self.cell_x - mp.x()) / pw
        dy = (self.cell_y - mp.y()) / ph
        d2 = np.where(self.shown, dx * dx + dy * dy, np.inf)
        k = int(np.argmin(d2))
        if d2[k] > HOVER_RADIUS_PX ** 2:
            self.clear_hover()
            return
        r, c = divmod(k, NCOL)
        step, servo = int(STEPS[c]), ROWS[r]
        if self.recv[r, c]:
            d = float(self.dist[r, c])
            x, y, z = self.point_xyz(step, servo, d)
            tip = (f"X  {x:10.1f} mm\nY  {y:10.1f} mm\nZ  {z:10.1f} mm\n"
                   f"range {d:.1f} mm  (step {step}, servo {servo} deg)")
            line = f"X {x:.1f} mm   Y {y:.1f} mm   Z {z:.1f} mm   | range {d:.1f} mm | step {step}, servo {servo} deg"
        else:
            tip = f"MISSED READING (grey)\nstep {step}, servo {servo} deg"
            line = f"MISSED READING  | step {step}, servo {servo} deg"
            if self.filled is not None:
                x, y, z = self.point_xyz(step, servo, float(self.filled[r, c]))
                tip += f"\nestimated: X {x:.0f}  Y {y:.0f}  Z {z:.0f} mm"
                line += f"  | estimated X {x:.0f}  Y {y:.0f}  Z {z:.0f} mm"
        self.hover_label.setText(line)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), tip, self.plot_widget)
        self.hl.setData(x=[float(self.cell_x[k])], y=[float(self.cell_y[k])])

    # ---------- dots ----------
    def refresh_dots(self):
        complete = (np.arange(NROW) < self.cur_row) | self.scan_done
        limit = np.where(complete, NCOL - 1, self.last_col)
        cols = np.arange(NCOL)[None, :]
        miss = (~self.recv) & (cols <= limit[:, None])
        self.shown = (self.recv | miss).ravel()
        self.n_missed = int(miss.sum())

        ri = np.flatnonzero(self.recv.ravel())
        if len(ri):
            d = self.dist.ravel()[ri].astype(float)
            near, far = np.percentile(d, [3, 97])
            if far - near < 300:
                far = near + 300
            t = dist_to_t(d, near, far)
            bins = np.minimum((t * (N_BR - 1)).astype(int), N_BR - 1)
            self.recv_scatter.setData(x=self.cell_x[ri], y=self.cell_y[ri],
                                      brush=[self.brush_lut[b] for b in bins],
                                      pen=pg.mkPen(None), size=DOT_SIZE)
        else:
            self.recv_scatter.clear()
        mi = np.flatnonzero(miss.ravel())
        if len(mi):
            self.miss_scatter.setData(x=self.cell_x[mi], y=self.cell_y[mi],
                                      brush=pg.mkBrush(140, 140, 140, 255),
                                      pen=pg.mkPen(None), size=DOT_SIZE)
        else:
            self.miss_scatter.clear()
        self.dirty = False

    # ---------- 3D model ----------
    def clear_model(self):
        for it in self.model_items:
            try:
                self.gl.removeItem(it)
            except Exception:
                pass
        self.model_items = []

    def _add3d(self, item):
        self.gl.addItem(item)
        self.model_items.append(item)

    def build_model(self):
        n = int(self.recv.sum())
        if n < MIN_POINTS_FOR_MODEL:
            print(f"[!] Only {n} points so far - need at least {MIN_POINTS_FOR_MODEL} to build the model")
            return
        if NROW < 2 or NCOL < 2:
            print("[!] Scan grid too small to build a surface")
            return
        d = self.dist.astype(float).copy()
        d[~self.recv] = np.nan
        filled, missing = fill_missing(d)
        if filled is None:
            return
        self.filled = filled
        xyz, faces, fmiss, fdist = build_mesh(smooth_rows(filled, MESH_MEDIAN), missing)
        self.geom = dict(xyz=xyz, faces=faces, fmiss=fmiss, fdist=fdist,
                         shade=face_shade(xyz, faces))
        self.render_model()
        self.stack.setCurrentIndex(1)
        self.reset_view()
        print(f"[*] 3D model built: {len(xyz)} vertices, {len(faces)} faces, "
              f"{int(missing.sum())} of {missing.size} readings were missed (grey)")

    def render_model(self):
        g = self.geom
        if g is None:
            return
        self.clear_model()
        xyz, faces, fmiss, shade = g["xyz"], g["faces"], g["fmiss"], g["shade"]
        style = STYLES[self.style_idx]
        self.model_head.setText(f"3D MODEL   (CAD view: {style} - press T to change style)")

        if style != "WIREFRAME":
            if style == "DEPTH":
                near, far = np.percentile(g["fdist"], [3, 97])
                if far - near < 300:
                    far = near + 300
                base = thermal_rgb(dist_to_t(g["fdist"], near, far))
            else:
                base = np.tile(COL_STEEL, (len(faces), 1))
            base = np.where(fmiss[:, None], COL_GREY[None, :], base)
            fc = np.ones((len(faces), 4), dtype=np.float32)
            fc[:, :3] = base * shade[:, None]
            self._add3d(gl.GLMeshItem(vertexes=xyz, faces=faces, faceColors=fc, smooth=False,
                                      drawEdges=False, shader=None, glOptions="opaque"))
        if style != "DEPTH":
            col = (0.05, 0.10, 0.20, 0.9) if style == "CONCEPTUAL" else (0.35, 0.90, 1.0, 1.0)
            every = 8 if style == "CONCEPTUAL" else 3
            lines = grid_lines(xyz * 0.996, NROW, NCOL, every)   # nudged toward the scanner
            self._add3d(gl.GLLinePlotItem(pos=lines, color=col, width=1.0,
                                          antialias=True, mode="lines"))

        # CAD floor grid + UCS icon at the scanner
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        ext = float(max(hi - lo)) or 1000.0
        half = float(max(np.abs(xyz[:, 0]).max(), np.abs(xyz[:, 1]).max(), 1000.0))
        size = float(np.ceil(2.2 * half / 1000.0) * 1000.0)
        grid = gl.GLGridItem(color=(110, 130, 150, 110))
        grid.setSize(x=size, y=size, z=0)
        grid.setSpacing(x=500, y=500, z=1)
        grid.translate(0, 0, float(lo[2]))
        self._add3d(grid)
        L = 0.12 * ext
        for vec, col in (((L, 0, 0), (1.0, 0.25, 0.25, 1)), ((0, L, 0), (0.25, 1.0, 0.25, 1)),
                         ((0, 0, L), (0.35, 0.55, 1.0, 1))):
            self._add3d(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], vec], dtype=np.float32),
                                          color=col, width=3, antialias=True))
        if HAS_TEXT:
            try:
                for txt, vec, rgb in (("X", (L, 0, 0), (255, 70, 70)), ("Y", (0, L, 0), (70, 255, 70)),
                                      ("Z", (0, 0, L), (90, 140, 255))):
                    self._add3d(gl.GLTextItem(pos=(vec[0] * 1.1, vec[1] * 1.1, vec[2] * 1.1), text=txt,
                                              color=QtGui.QColor(*rgb), font=QtGui.QFont("Arial", 11)))
            except Exception:
                pass

    def next_style(self):
        self.style_idx = (self.style_idx + 1) % len(STYLES)
        self.render_model()

    def reset_view(self):
        if self.geom is not None:
            xyz = self.geom["xyz"]
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            c = (lo + hi) / 2.0
            ext = float(max(hi - lo)) or 3000.0
            self.gl.opts["center"] = pg.Vector(float(c[0]), float(c[1]), float(c[2]))
            self.gl.setCameraPosition(distance=ext * 1.7, elevation=28, azimuth=-90)
        else:
            self.gl.opts["center"] = pg.Vector(0, 2500, -500)
            self.gl.setCameraPosition(distance=9000, elevation=28, azimuth=-90)

    # ---------- data ----------
    def start_scan(self):
        ok = self.reader.send(b"S")
        print("[>] Scan command sent" if ok else "[!] Not connected - cannot send scan command")

    def clear(self):
        self.dist[:] = np.nan
        self.recv[:] = False
        self.last_col[:] = -1
        self.cur_row = -1
        self.scan_done = False
        self.filled = None
        self.geom = None
        self.shown[:] = False
        self.n_missed = 0
        self.clear_model()
        self.stack.setCurrentIndex(0)
        self.recv_scatter.clear()
        self.miss_scatter.clear()
        self.clear_hover()
        print("[*] Cleared")

    def export_csv(self):
        if not self.recv.any():
            print("[!] Nothing to export")
            return
        path = f"scan_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "col", "step", "servo_deg", "status", "dist_mm", "x_mm", "y_mm", "z_mm"])
                for r in range(NROW):
                    for c in range(NCOL):
                        step, servo = int(STEPS[c]), ROWS[r]
                        if self.recv[r, c]:
                            d, status = float(self.dist[r, c]), "measured"
                        elif self.filled is not None:
                            d, status = float(self.filled[r, c]), "missed_estimated"
                        else:
                            continue
                        x, y, z = self.point_xyz(step, servo, d)
                        w.writerow([r, c, step, servo, status, f"{d:.1f}", f"{x:.1f}", f"{y:.1f}", f"{z:.1f}"])
            print(f"[*] Exported -> {path}")
        except OSError as e:
            print(f"[!] Could not export: {e}")

    def handle_message(self, text):
        print(f"[ESP32] {text}" if text.startswith("#") else f"[*] {text}")
        if text.startswith("Connected"):
            self.conn_text = "ONLINE"
        elif text.startswith("Serial error"):
            self.conn_text = "OFFLINE - retrying"
        elif text.startswith("#SCAN_START"):
            self.clear()
            self.scan_active = True
        elif text.startswith("#SCAN_END"):
            self.scan_active = False
            self.scan_done = True
            self.refresh_dots()
            self.build_model()

    def apply_point(self, step, servo, dist):
        cell = cell_of(step, servo)
        if cell is None:
            if not self.snap_warned:
                print(f"[!] Reading outside the expected scan grid ignored (step {step}, servo {servo}). "
                      "Check the grid settings at the top of this file match the ESP32 sketch.")
                self.snap_warned = True
            return
        r, c = cell
        self.dist[r, c] = dist
        self.recv[r, c] = True
        self.last_col[r] = max(self.last_col[r], c)
        self.cur_row = max(self.cur_row, r)
        self.dirty = True

    def tick(self):
        batch = []
        for _ in range(5000):
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "msg":
                self.handle_message(payload)
            else:
                batch.append(payload)
        for step, servo, dist, raw in batch:
            self.apply_point(step, servo, dist)
        if batch:
            try:
                self.log.write("\n".join(b[3] for b in batch) + "\n")
            except OSError:
                pass
        now = time.time()
        if self.dirty and now >= self.next_refresh:
            self.refresh_dots()
            self.next_refresh = now + 1.0 / REFRESH_HZ
        total = NROW * NCOL
        state = "SCANNING" if self.scan_active else ("DONE" if self.scan_done else "STANDBY")
        self.status.setText(
            f"LINK {self.conn_text}   |   {state}   |   RECEIVED {int(self.recv.sum())} / {total}   |   "
            f"MISSED {self.n_missed}   |   3D STYLE {STYLES[self.style_idx]}")

    def closeEvent(self, event):
        self.timer.stop()
        self.reader.stop_event.set()
        self.reader.join(timeout=2.0)
        try:
            self.log.close()
        except Exception:
            pass
        event.accept()


def main():
    ap = argparse.ArgumentParser(description="Radar scan studio")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = ap.parse_args()
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.port, args.baud)
    win.show()
    signal.signal(signal.SIGINT, lambda *_: win.close())
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()
