"""Colour calibration dialog: capture a ColorChecker chart and compute a correction matrix."""
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout

from . import style
from .widgets import button, caption, label


def _to_pixmap(bgr, max_w, max_h):
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _annotate(img, corners, measured=None, ok=True):
    """The captured image with the detected chart drawn on it: a thick outline (green = accepted,
    red = rejected) and, when accepted, a disc on each of the 24 patches filled with the colour the
    detector measured there, so a real detection is obvious at a glance (the discs vanish into the patches)."""
    out = img.copy()
    if corners is None or len(corners) != 4:
        return out
    w = out.shape[1]
    t = max(3, w // 250)
    colour = (60, 200, 60) if ok else (40, 40, 230)  # BGR
    pts = np.round(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, colour, t, cv2.LINE_AA)
    if measured is not None:
        tl, tr, br, bl = (np.asarray(c, float) for c in corners)
        rad = max(6, int(np.linalg.norm(tr - tl) / 6 * 0.22))
        for k in range(24):
            r, c = divmod(k, 6)
            u, v = (c + 0.5) / 6, (r + 0.5) / 4
            p = (tl + (tr - tl) * u) * (1 - v) + (bl + (br - bl) * u) * v
            centre = (int(round(p[0])), int(round(p[1])))
            cv2.circle(out, centre, rad, tuple(int(x) for x in measured[k][::-1]), -1, cv2.LINE_AA)
            cv2.circle(out, centre, rad, (255, 255, 255), max(2, t // 2), cv2.LINE_AA)
    return out


class CalibrationDialog(QDialog):
    """Live-view dialog that captures a ColorChecker chart and saves a colour correction matrix."""

    def __init__(self, shell, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.setWindowTitle("Colour calibration")
        self.setMinimumSize(860, 560)
        self.result_matrix = None
        self._frame_img = None
        self._still = None  # the captured image with the detection drawn on it; live frames wait behind it

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(caption("place the colorchecker in the centre of the frame"))
        self.note = label("The 24-patch chart must be fully visible inside the red box.\n"
                          "Do not change lighting or camera settings after calibration.",
                          "dim", wrap=True)
        layout.addWidget(self.note)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(400)
        self.view.setStyleSheet(f"background: {style.CANVAS}; border-radius: 4px;")
        layout.addWidget(self.view, 1)

        self.status = label("waiting for live view ...", "dim")
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.b_calibrate = button("calibrate", self._calibrate, primary=True)
        self.b_skip = button("skip", self.reject)
        row.addStretch(1)
        row.addWidget(self.b_skip)
        row.addWidget(self.b_calibrate)
        layout.addLayout(row)

        self.shell.camera.frame.connect(self._on_frame)
        self.shell.camera.set_live(True)

    def _on_frame(self, img):
        self._frame_img = img
        if self._still is not None:
            return  # showing the detection result; TRY AGAIN / DONE clears it
        h, w = img.shape[:2]
        pm = _to_pixmap(img, self.view.width(), self.view.height())
        # draw red dashed centre rectangle (where the chart should go)
        painter = QPainter(pm)
        pen = QPen(QColor(style.GUIDE), 2.5)
        pen.setStyle(Qt.CustomDashLine)
        pen.setDashPattern([10, 6])
        painter.setPen(pen)
        pw, ph = pm.width(), pm.height()
        rx, ry = int(pw * 0.25), int(ph * 0.25)
        rw, rh = int(pw * 0.5), int(ph * 0.5)
        painter.drawRect(rx, ry, rw, rh)
        painter.end()
        self.view.setPixmap(pm)

    def _show_still(self, img):
        self._still = img
        self.view.setPixmap(_to_pixmap(img, self.view.width(), self.view.height()))

    def _calibrate(self):
        if self._still is not None and self.result_matrix is None:
            # back from a rejected attempt: show the live view again first
            self._still = None
            self.b_calibrate.setText("CALIBRATE")
            self.status.setText("live view: place the chart inside the red box, then CALIBRATE")
            self.status.setStyleSheet("")
            return
        if self._frame_img is None:
            return QMessageBox.warning(self, "Calibration", "No live view frame yet.")

        self.status.setText("capturing calibration image ...")
        self.b_calibrate.setEnabled(False)

        # capture a full-resolution image for detection
        import colourcal
        project = self.shell.project
        cal_dir = project.root / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_img = cal_dir / "colourcal.jpg"
        self._cal_path = cal_dir / "colourcal.json"

        self.shell.camera.set_live(False)

        def on_captured(info):
            try:
                self.shell.camera.captured.disconnect(on_captured)
            except (RuntimeError, TypeError):
                pass
            # read the captured image
            if "rgb" in info:
                img = cv2.imread(str(info["rgb"]))
            else:
                img = cv2.imread(str(info["left"]))
            if img is None:
                self._fail("Could not read captured image")
                return
            cv2.imwrite(str(cal_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._detect(img)

        def on_failed(msg, tag):
            try:
                self.shell.camera.failed.disconnect(on_failed)
            except (RuntimeError, TypeError):
                pass
            self._fail(f"Capture failed: {msg}")

        self.shell.camera.captured.connect(on_captured)
        self.shell.camera.failed.connect(on_failed)
        out_dir = cal_dir / "colourcal_raw"
        self.shell.camera.capture(out_dir, tag="colourcal")

    def _detect(self, img):
        import colourcal

        measured, corners = colourcal.detect_checker(img)
        if measured is None:
            self._fail("ColorChecker not detected. Make sure all 24 patches are visible "
                       "and well-lit inside the red box.", still=img)
            return
        # the detector can "find" a chart where there is none: check the patches look like one
        ok, why = colourcal.validate_measurement(measured)
        if not ok:
            self._fail(f"Not a usable ColorChecker: {why}. Place the chart inside the red box and try again.",
                       still=_annotate(img, corners, ok=False))
            return

        M = colourcal.colour_matrix(measured)
        from colourcal.correct import calibration_error, save_calibration
        err = calibration_error(measured, M)
        ok, why = colourcal.validate_fit(err)
        if not ok:
            self._fail(f"Calibration rejected: {why}. Nothing was saved; try again.",
                       still=_annotate(img, corners, ok=False))
            return
        save_calibration(self._cal_path, M, measured, err)
        self.result_matrix = M
        # the proof: the chart outlined in green, each patch under a disc of the colour measured there
        self._show_still(_annotate(img, corners, measured, ok=True))

        self.status.setText(f"calibrated — {why}")
        self.status.setStyleSheet(f"color: {style.ACCENT}; font-weight: 600;")
        self.note.setText("Colour calibration saved: the chart is outlined in green and each disc shows the colour "
                          "measured on its patch (they should blend in). Do not change lighting or camera settings.")
        self.b_calibrate.setText("DONE")
        self.b_calibrate.setEnabled(True)
        self.b_calibrate.clicked.disconnect()
        self.b_calibrate.clicked.connect(self.accept)
        self.b_skip.setVisible(False)
        self.shell.camera.set_live(True)
        self.shell.camera.frame.connect(self._on_frame)

    def _fail(self, msg, still=None):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color: {style.BAD}; font-weight: 600;")
        if still is not None:
            self._show_still(still)  # what was captured, with any (rejected) detection in red
            self.b_calibrate.setText("TRY AGAIN")
        self.b_calibrate.setEnabled(True)
        self.shell.camera.set_live(True)
        self.shell.camera.frame.connect(self._on_frame)

    def closeEvent(self, e):
        self.shell.camera.set_live(False)
        try:
            self.shell.camera.frame.disconnect(self._on_frame)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(e)

    def reject(self):
        self.shell.camera.set_live(False)
        try:
            self.shell.camera.frame.disconnect(self._on_frame)
        except (RuntimeError, TypeError):
            pass
        super().reject()
