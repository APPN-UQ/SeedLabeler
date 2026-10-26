"""CAPTURE mode: fast, scanner-driven collection. No detection; PROCESS does that later."""
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QListWidget, QVBoxLayout, QWidget

from . import live, style
from .canvas import Canvas, to_qimage
from .widgets import button, caption, card, label, vline


def quality(sample):
    """Quick checks on a fresh capture: [(ok, text)], plus a small preview (BGR)."""
    checks = []
    imgs = []
    files = [("colour", sample.rgb)] if sample.is_rgbd else [("left", sample.left), ("right", sample.right)]
    for side, p in files:
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            checks.append((False, f"{side} image missing"))
            continue
        imgs.append(img)
    if sample.is_rgbd and imgs:
        with np.load(sample.depth_file) as z:
            cov = float((z["depth"] > 0).mean())
        checks.append((cov > 0.5, f"depth on {100 * cov:.0f}% of the view"
                       + ("" if cov > 0.5 else ": too close (under 40 cm)? or a blank background")))
    if len(imgs) == len(files):
        checks.append((True, "images saved" if sample.is_rgbd else "both images saved"))
        sharp = []
        for img in imgs:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            sharp.append(cv2.Laplacian(g[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.CV_64F).var())
        s = min(sharp)
        checks.append((s > 15, f"sharpness {s:.0f}" + ("" if s > 15 else ": blurred? check focus / movement")))
        g = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
        mean, clipped = float(g.mean()), float((g > 250).mean())
        if mean < 40:
            checks.append((False, f"dark (mean {mean:.0f}): raise ISO / light"))
        elif clipped > 0.05:
            checks.append((False, f"{100 * clipped:.0f}% blown highlights: lower exposure"))
        else:
            checks.append((True, f"exposure ok (mean {mean:.0f})"))
    preview = imgs[0] if imgs else None
    return checks, preview


class CapturePage(QWidget):
    def __init__(self, shell):
        super().__init__()
        self.setObjectName("page")
        self.shell = shell
        self.project = None
        self.session = 0
        self._first = True
        body = QHBoxLayout(self)
        body.setContentsMargins(18, 12, 18, 8)
        body.setSpacing(16)

        # live view
        self.canvas = Canvas()
        self.canvas.set_mode("view")
        lv = QVBoxLayout()
        lv.addWidget(caption("live view · keep seeds inside the dashed box"))
        lv.addWidget(self.canvas, 1)
        lw = QWidget()
        lw.setLayout(lv)
        body.addWidget(lw, 1)
        body.addWidget(vline())

        # controls
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(caption("sample id"))
        self.sample_id = QLineEdit()
        self.sample_id.setObjectName("scan")
        self.sample_id.setPlaceholderText("Scan barcode")
        self.sample_id.returnPressed.connect(self._id_entered)
        right.addWidget(self.sample_id)
        row = QHBoxLayout()
        self.b_auto = button("auto capture on scan", checkable=True)
        self.b_auto.setChecked(True)
        row.addWidget(self.b_auto)
        row.addStretch(1)
        right.addLayout(row)
        self.b_capture = button("capture", self.capture, primary=True)
        self.b_capture.setMinimumHeight(48)
        right.addWidget(self.b_capture)
        right.addSpacing(8)

        right.addWidget(caption("last capture"))
        self.thumb = QLabel()
        self.thumb.setFixedHeight(220)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background: #e9e7e1; border-radius: 4px;")
        right.addWidget(self.thumb)
        self.checks = label("—", "readout", wrap=True)
        right.addWidget(self.checks)
        right.addSpacing(8)

        stats_card, sv = card(caption("counts"))
        self.stats = label("—", "readout")
        sv.addWidget(self.stats)
        right.addWidget(stats_card)
        right.addWidget(caption("recent"))
        self.recent = QListWidget()
        right.addWidget(self.recent, 1)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(400)
        body.addWidget(rw)

    # --- lifecycle --------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.session = 0
        self.recent.clear()
        self._update_stats()

    def enter(self):
        self._first = True
        self.shell.camera.frame.connect(self._frame)
        self.shell.camera.set_live(True)
        self.sample_id.setFocus()
        self._update_stats()

    def leave(self):
        self.shell.camera.set_live(False)
        try:
            self.shell.camera.frame.disconnect(self._frame)
        except (RuntimeError, TypeError):
            pass

    def _frame(self, img):
        if self._first:
            self.canvas.clear()
            self.canvas.set_image(img)
            zone = live.depth_zone(self.project, (img.shape[1], img.shape[0]))
            self.canvas.set_guide(None if zone is None else zone.tolist())
            self.canvas.fit()
            self._first = False
        else:
            self.canvas.swap_image(img)

    # --- capture ----------------------------------------------------------------
    def _id_entered(self):
        if self.b_auto.isChecked() and self.sample_id.text().strip():
            self.capture()

    def capture(self):
        sid = self.sample_id.text().strip() or None
        self.sample_id.clear()
        self.b_capture.setEnabled(False)
        t0 = time.time()

        def done(sample):
            self.b_capture.setEnabled(True)
            self.session += 1
            checks, preview = quality(sample)
            ok = all(c[0] for c in checks)
            mark = "✓" if ok else "!"
            self.recent.insertItem(0, f"{mark}  {sample.id}   {time.strftime('%H:%M:%S')}   {time.time() - t0:.1f}s")
            self.checks.setText("\n".join(("✓ " if c[0] else "! ") + c[1] for c in checks))
            self.checks.setStyleSheet("" if ok else f"color: {style.WARN};")
            if preview is not None:
                pm = QPixmap.fromImage(to_qimage(preview)).scaled(self.thumb.width(), self.thumb.height(),
                                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb.setPixmap(pm)
            self._update_stats()
            self.sample_id.setFocus()  # ready for the next scan

        def failed(_msg):
            self.b_capture.setEnabled(True)
            self.sample_id.setFocus()

        self.shell.capture_sample(sid, done, failed)

    def _update_stats(self):
        if not self.project:
            return
        samples = self.project.samples()
        waiting = sum(1 for s in samples if not s.processed)
        self.stats.setText(f"THIS SESSION   {self.session}\nIN PROJECT     {len(samples)}\n"
                           f"TO PROCESS     {waiting}")
