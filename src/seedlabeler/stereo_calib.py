"""Guided stereo calibration for the RX0 pair and calibration-quality verification.

The wizard captures ChArUco board pairs through the live camera service, then
runs the same OpenCV stereo calibration that the CLI ``rx0-calibrate`` does.

``verify_calibration`` checks an existing .npz against a single stereo pair:
it rectifies the images, detects ArUco markers in both, and reports the mean
vertical (epipolar) error on the matched marker corners.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QHBoxLayout, QLabel,
                                QListWidget, QMessageBox, QVBoxLayout)

from . import style
from .widgets import button, caption, label

MIN_PAIRS = 10
TARGET_PAIRS = 20
MIN_CORNERS = 8


# ---------------------------------------------------------------------------
# Verification: check calibration quality using ArUco markers
# ---------------------------------------------------------------------------

def verify_calibration(calib_path, img_l, img_r, sheet_layout=None, bar_scale=1.0):
    """Check stereo calibration by measuring rectification error on ArUco markers.

    Returns (quality, error_px, detail_str) or (None, None, reason) if the
    check could not run (no markers visible, etc.).

    *quality*: ``"good"`` (<1.5 px), ``"poor"`` (<3 px), ``"bad"`` (>=3 px).
    *error_px*: mean absolute vertical disparity of matched marker corners
    after rectification — should be ~0 for a perfect calibration.
    """
    from sonyrx0.stereo.depth import StereoRig

    rig = StereoRig(str(calib_path), scale=0.25)
    small_l = cv2.resize(img_l, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_r = cv2.resize(img_r, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    rect_l, rect_r = rig.rectify(small_l, small_r)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        cv2.aruco.DetectorParameters())
    corners_l, ids_l, _ = detector.detectMarkers(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY))
    corners_r, ids_r, _ = detector.detectMarkers(cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))

    if ids_l is None or ids_r is None:
        return None, None, "no markers detected in one or both images"

    set_l = {int(i): c for i, c in zip(ids_l.ravel(), corners_l)}
    set_r = {int(i): c for i, c in zip(ids_r.ravel(), corners_r)}
    common = sorted(set(set_l) & set(set_r))
    if not common:
        return None, None, "no markers in common between the two views"

    errors = []
    for mid in common:
        cl = set_l[mid][0]  # (4, 2)
        cr = set_r[mid][0]
        errors.extend(np.abs(cl[:, 1] - cr[:, 1]).tolist())

    mean_err = float(np.mean(errors))
    quality = "good" if mean_err < 1.5 else "poor" if mean_err < 3.0 else "bad"
    detail = f"{len(common)} markers, {mean_err:.2f} px mean vertical error"

    if sheet_layout is not None and len(common) >= 2:
        from chickpea.stage import find_markers
        found_l = find_markers(img_l, sheet_layout)
        found_r = find_markers(img_r, sheet_layout)
        if len(found_l) >= 2:
            detail += f" · {len(found_l)} stage markers visible"

    return quality, mean_err, detail


# ---------------------------------------------------------------------------
# Guided stereo calibration dialog
# ---------------------------------------------------------------------------

def _to_pixmap(bgr, max_w, max_h):
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class StereoCalibDialog(QDialog):
    """Capture ChArUco board pairs and compute stereo calibration."""

    def __init__(self, shell, project, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.project = project
        self.setWindowTitle("Stereo calibration")
        self.setMinimumSize(960, 640)
        self.result_path = None
        self._pairs = []
        self._capturing = False
        self._frame_img = None

        from sonyrx0.stereo.common import (DEFAULT_DICT, DEFAULT_SQUARES_X,
                                            DEFAULT_SQUARES_Y)
        self._squares_x = DEFAULT_SQUARES_X
        self._squares_y = DEFAULT_SQUARES_Y
        self._dict_name = DEFAULT_DICT
        self._board = None
        self._detector = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(caption("stereo calibration · charuco board"))
        self.note = label(
            "Hold the printed ChArUco board at varying angles and distances so it covers "
            "the whole frame, especially corners. Take 20+ pairs. Lock focus first (manual).",
            "dim", wrap=True)
        layout.addWidget(self.note)

        body = QHBoxLayout()

        # Left: live view
        lv = QVBoxLayout()
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(400)
        self.view.setStyleSheet(f"background: {style.CANVAS}; border-radius: 4px;")
        lv.addWidget(self.view, 1)
        self.status_label = label("waiting for live view ...", "dim")
        lv.addWidget(self.status_label)
        body.addLayout(lv, 1)

        # Right: controls
        ctrl = QVBoxLayout()
        ctrl.setSpacing(10)

        ctrl.addWidget(caption("printed square size"))
        self.square_mm = QDoubleSpinBox()
        self.square_mm.setRange(5, 100)
        self.square_mm.setValue(23.9)
        self.square_mm.setSingleStep(0.1)
        self.square_mm.setDecimals(1)
        self.square_mm.setSuffix(" mm")
        self.square_mm.setToolTip("Measure a printed square on the ChArUco board with a ruler")
        self.square_mm.valueChanged.connect(self._rebuild_board)
        ctrl.addWidget(self.square_mm)

        self.b_capture = button("capture pair (Enter)", self._capture, primary=True)
        self.b_capture.setMinimumHeight(48)
        ctrl.addWidget(self.b_capture)

        ctrl.addWidget(caption("captured pairs"))
        self.pair_list = QListWidget()
        self.pair_list.setMinimumWidth(280)
        ctrl.addWidget(self.pair_list, 1)

        self.pair_status = label("0 pairs captured", "readout")
        ctrl.addWidget(self.pair_status)

        self.b_calibrate = button("calibrate", self._calibrate, secondary=True)
        self.b_calibrate.setEnabled(False)
        ctrl.addWidget(self.b_calibrate)

        self.result_label = label("", "readout", wrap=True)
        ctrl.addWidget(self.result_label)

        row = QHBoxLayout()
        self.b_cancel = button("cancel", self.reject)
        self.b_done = button("done", self.accept, primary=True)
        self.b_done.setVisible(False)
        row.addStretch(1)
        row.addWidget(self.b_cancel)
        row.addWidget(self.b_done)
        ctrl.addLayout(row)

        body.addLayout(ctrl)
        layout.addLayout(body, 1)

        self._rebuild_board()
        self.shell.camera.frame.connect(self._on_frame)
        self.shell.camera.set_live(True)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and not self._capturing:
            self._capture()
        else:
            super().keyPressEvent(e)

    def _rebuild_board(self):
        from sonyrx0.stereo.common import make_board
        sq = self.square_mm.value()
        self._board = make_board(self._squares_x, self._squares_y, sq, dict_name=self._dict_name)
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def _on_frame(self, img):
        self._frame_img = img
        display = img.copy()
        if self._detector is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = self._detector.detectBoard(gray)
            n = 0 if ids is None else len(ids)
            if corners is not None and len(corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            color = (60, 200, 60) if n >= MIN_CORNERS else (40, 40, 220)
            cv2.putText(display, f"corners: {n}", (16, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        pm = _to_pixmap(display, self.view.width(), self.view.height())
        self.view.setPixmap(pm)

    # ------------------------------------------------------------------ capture
    def _capture(self):
        if self._capturing:
            return
        self._capturing = True
        self.b_capture.setEnabled(False)
        self.status_label.setText("capturing ...")

        n = len(self._pairs)
        out_dir = self.project.root / "calibration" / "_calib_tmp" / f"pair_{n:03d}"

        def on_captured(info):
            try:
                self.shell.camera.captured.disconnect(on_captured)
            except (RuntimeError, TypeError):
                pass
            try:
                self.shell.camera.failed.disconnect(on_failed)
            except (RuntimeError, TypeError):
                pass
            self._capturing = False
            self.b_capture.setEnabled(True)
            self._process_pair(info)

        def on_failed(msg, tag):
            try:
                self.shell.camera.captured.disconnect(on_captured)
            except (RuntimeError, TypeError):
                pass
            try:
                self.shell.camera.failed.disconnect(on_failed)
            except (RuntimeError, TypeError):
                pass
            self._capturing = False
            self.b_capture.setEnabled(True)
            self.status_label.setText(f"capture failed: {msg}")
            self.status_label.setStyleSheet(f"color: {style.WARN};")

        self.shell.camera.captured.connect(on_captured)
        self.shell.camera.failed.connect(on_failed)
        self.shell.camera.capture(out_dir, tag=f"_stereo_calib_{n}")

    def _process_pair(self, info):
        from sonyrx0.stereo.calibrate import detect
        from sonyrx0.stereo.common import read_image

        left_path, right_path = info.get("left"), info.get("right")
        if not left_path or not right_path:
            self.status_label.setText("not a stereo pair — need both cameras")
            self.status_label.setStyleSheet(f"color: {style.WARN};")
            return

        img_l, img_r = read_image(left_path), read_image(right_path)
        dl = detect(self._detector, self._board, img_l)
        dr = detect(self._detector, self._board, img_r)
        shared = len(set(dl) & set(dr))

        self._pairs.append({
            "left": left_path, "right": right_path,
            "det_left": dl, "det_right": dr, "shared": shared,
            "image_size": img_l.shape[1::-1],
        })

        ok = shared >= MIN_CORNERS
        mark = "✓" if ok else "✗"
        self.pair_list.addItem(f"{mark}  pair {len(self._pairs):2d}:  {len(dl)}L  {len(dr)}R  {shared} shared")
        self.pair_list.scrollToBottom()

        usable = sum(1 for p in self._pairs if p["shared"] >= MIN_CORNERS)
        need = max(0, MIN_PAIRS - usable)
        self.pair_status.setText(
            f"{len(self._pairs)} pairs ({usable} usable)"
            + (f" — need {need} more" if need else " — ready to calibrate"))
        self.b_calibrate.setEnabled(usable >= MIN_PAIRS)

        if ok:
            self.status_label.setText(f"pair {len(self._pairs)}: {shared} shared corners ✓")
            self.status_label.setStyleSheet("")
        else:
            self.status_label.setText(f"pair {len(self._pairs)}: only {shared} shared corners — try another angle")
            self.status_label.setStyleSheet(f"color: {style.WARN};")

    # ------------------------------------------------------------------ calibrate
    def _calibrate(self):
        from sonyrx0.stereo.calibrate import to_arrays
        from sonyrx0.stereo.common import save_calibration

        self.status_label.setText("calibrating ...")
        self.b_calibrate.setEnabled(False)

        usable = [p for p in self._pairs if p["shared"] >= MIN_CORNERS]
        image_size = usable[0]["image_size"]

        mono_l, mono_r = ([], []), ([], [])
        stereo_obj, stereo_l, stereo_r = [], [], []
        for p in usable:
            dl, dr = p["det_left"], p["det_right"]
            if dl:
                o, ip = to_arrays(dl, sorted(dl))
                mono_l[0].append(o)
                mono_l[1].append(ip)
            if dr:
                o, ip = to_arrays(dr, sorted(dr))
                mono_r[0].append(o)
                mono_r[1].append(ip)
            common = sorted(set(dl) & set(dr))
            o, pl = to_arrays(dl, common)
            _, pr = to_arrays(dr, common)
            stereo_obj.append(o)
            stereo_l.append(pl)
            stereo_r.append(pr)

        flags = cv2.CALIB_RATIONAL_MODEL
        try:
            rms_l, K1, D1, _, _ = cv2.calibrateCamera(*mono_l, image_size, None, None, flags=flags)
            rms_r, K2, D2, _, _ = cv2.calibrateCamera(*mono_r, image_size, None, None, flags=flags)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
            keep = list(range(len(stereo_obj)))
            while True:
                R, T = np.eye(3), np.zeros((3, 1))
                out = cv2.stereoCalibrateExtended(
                    [stereo_obj[i] for i in keep], [stereo_l[i] for i in keep],
                    [stereo_r[i] for i in keep],
                    K1, D1, K2, D2, image_size, R, T,
                    criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC | flags)
                rms_s, K1, D1, K2, D2, R, T, E, F = out[:9]
                per_view = out[-1]
                errs = per_view.max(axis=1)
                limit = max(1.0, 2.5 * float(np.median(errs)))
                bad = [k for k, e in zip(keep, errs) if e > limit]
                if not bad or len(keep) - len(bad) < 10:
                    break
                keep = [k for k in keep if k not in bad]

            baseline = float(np.linalg.norm(T))
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

            out_path = self.project.root / "calibration" / "calib.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_calibration(str(out_path), K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
                             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                             image_size=np.array(image_size), rms=np.array(rms_s))

            # save rig.json so the same left/right assignment is used next time
            if hasattr(self.shell.camera, '_left_id') and self.shell.camera._left_id:
                rig = out_path.with_name("rig.json")
                rig.write_text(json.dumps({"left": self.shell.camera._left_id}))

            self.result_path = out_path
            quality = "good" if rms_s < 0.5 else "ok" if rms_s < 1.0 else "marginal"
            self.result_label.setText(
                f"Stereo RMS: {rms_s:.3f} px ({quality})\n"
                f"Baseline: {baseline:.1f} mm\n"
                f"Used {len(keep)}/{len(usable)} pairs\n"
                f"Intrinsics: L {rms_l:.3f} px, R {rms_r:.3f} px")
            self.result_label.setStyleSheet(
                f"color: {style.WARN};" if rms_s > 1.0 else f"color: {style.ACCENT};")
            self.status_label.setText(f"calibration saved — {out_path.name}")
            self.status_label.setStyleSheet("")
            self.b_done.setVisible(True)
            self.b_calibrate.setText("recalibrate")
            self.b_calibrate.setEnabled(True)

        except Exception as e:  # noqa: BLE001
            self.status_label.setText(f"calibration failed: {e}")
            self.status_label.setStyleSheet(f"color: {style.WARN};")
            self.b_calibrate.setEnabled(True)

    # ------------------------------------------------------------------ lifecycle
    def _stop_live(self):
        self.shell.camera.set_live(False)
        try:
            self.shell.camera.frame.disconnect(self._on_frame)
        except (RuntimeError, TypeError):
            pass

    def closeEvent(self, e):
        self._stop_live()
        super().closeEvent(e)

    def reject(self):
        self._stop_live()
        super().reject()

    def accept(self):
        self._stop_live()
        super().accept()
