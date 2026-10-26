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


class CalibrationDialog(QDialog):
    """Live-view dialog that captures a ColorChecker chart and saves a colour correction matrix."""

    def __init__(self, shell, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.setWindowTitle("Colour calibration")
        self.setMinimumSize(860, 560)
        self.result_matrix = None
        self._frame_img = None

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

    def _calibrate(self):
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
                       "and well-lit inside the red box.")
            return

        M = colourcal.colour_matrix(measured)
        from colourcal.correct import calibration_error, save_calibration
        err = calibration_error(measured, M)
        save_calibration(self._cal_path, M, measured, err)
        self.result_matrix = M

        self.status.setText(f"calibrated — mean error {err['mean']:.1f}, "
                            f"max {err['max']:.1f} (sRGB units)")
        self.status.setStyleSheet(f"color: {style.ACCENT}; font-weight: 600;")
        self.note.setText("Colour calibration saved. Do not change lighting or camera position.\n"
                          "The correction will be applied during processing.")
        self.b_calibrate.setText("DONE")
        self.b_calibrate.setEnabled(True)
        self.b_calibrate.clicked.disconnect()
        self.b_calibrate.clicked.connect(self.accept)
        self.b_skip.setVisible(False)
        self.shell.camera.set_live(True)
        self.shell.camera.frame.connect(self._on_frame)

    def _fail(self, msg):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color: {style.WARN};")
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
