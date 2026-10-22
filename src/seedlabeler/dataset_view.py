"""VIEW DATASET: thumbnails of the crop's dataset samples, each zoomable, removable, openable."""
import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from . import ml, processing
from .canvas import Canvas, to_qimage
from .style import palette_bgr

THUMB = 220


def render_masks(ortho, seeds, alpha=0.45):
    """Top-down image with each seed filled and outlined in its palette colour."""
    out = ortho.copy()
    fill = ortho.copy()
    for i, s in enumerate(seeds):
        pts = np.round(np.asarray(s["polygon"])).astype(np.int32)
        cv2.fillPoly(fill, [pts], palette_bgr(i))
    out = cv2.addWeighted(fill, alpha, out, 1 - alpha, 0)
    for i, s in enumerate(seeds):
        pts = np.round(np.asarray(s["polygon"])).astype(np.int32)
        cv2.polylines(out, [pts], True, palette_bgr(i), max(2, ortho.shape[1] // 800), cv2.LINE_AA)
    return out


def thumbnail(sample):
    """Cached thumbnail (seeds' region, coloured masks); rebuilt when the outlines change."""
    f = sample.path / "thumb.jpg"
    if f.exists() and sample.ann_file.exists() and f.stat().st_mtime >= sample.ann_file.stat().st_mtime:
        return cv2.imread(str(f))
    ortho, _, _ = processing.load_layers(sample)
    seeds = sample.annotations()["seeds"]
    img = render_masks(ortho, seeds)
    if seeds:
        p = np.vstack([np.asarray(s["polygon"]) for s in seeds])
        m = 0.08 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        x0, y0 = np.maximum(p.min(0) - m, 0).astype(int)
        x1, y1 = (p.max(0) + m).astype(int)
        img = img[y0:y1, x0:x1]
    s = 2 * THUMB / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img


def _button(text, slot):
    b = QPushButton(text.upper())
    b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(slot)
    return b


class DatasetDialog(QDialog):
    openSample = Signal(str)  # sample id

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("DATASET")
        self.resize(1200, 820)
        v = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("readout")
        v.addWidget(self.title)
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)

        # page 0: thumbnails
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 24, THUMB + 44))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setSpacing(8)
        self.grid.itemDoubleClicked.connect(self._open)
        self.stack.addWidget(self.grid)

        # page 1: zoomable view of one sample
        page = QWidget()
        pv = QVBoxLayout(page)
        pv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(_button("back", lambda: self.stack.setCurrentIndex(0)))
        self.view_title = QLabel()
        self.view_title.setObjectName("readout")
        row.addWidget(self.view_title)
        row.addStretch(1)
        row.addWidget(_button("open in editor", self._open_in_editor))
        row.addWidget(_button("remove from dataset", self._remove))
        pv.addLayout(row)
        self.canvas = Canvas()
        self.canvas.set_mode("view")
        pv.addWidget(self.canvas, 1)
        self.stack.addWidget(page)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("DOUBLE-CLICK A SAMPLE TO ZOOM IN   ·   WHEEL ZOOM, MIDDLE-DRAG PAN, F FIT"))
        bottom.addStretch(1)
        bottom.addWidget(_button("close", self.accept))
        v.addLayout(bottom)
        self.current = None
        self.refresh()

    def _dataset_samples(self):
        root = ml.dataset_root(self.project, "rgb")
        out = []
        for s in self.project.samples():
            if s.processed and ml._sample_files(root, s.id):
                split = "val" if list(root.glob(f"labels/val/{s.id}*")) else "train"
                out.append((s, split))
        return out

    def refresh(self):
        self.grid.clear()
        items = self._dataset_samples()
        n_seeds = 0
        for s, split in items:
            n = len(s.annotations()["seeds"])
            n_seeds += n
            img = thumbnail(s)
            it = QListWidgetItem(QIcon(QPixmap.fromImage(to_qimage(img))), f"{s.id}\n{n} SEEDS · {split.upper()}")
            it.setData(Qt.UserRole, s.id)
            it.setTextAlignment(Qt.AlignHCenter)
            self.grid.addItem(it)
        self.title.setText(f"{self.project.config.crop.upper()} DATASET   ·   {len(items)} SAMPLES   ·   {n_seeds} SEEDS")

    def _open(self, item):
        sid = item.data(Qt.UserRole)
        s = next((x for x in self.project.samples() if x.id == sid), None)
        if s is None:
            return
        self.current = s
        ortho, _, _ = processing.load_layers(s)
        self.canvas.clear()
        self.canvas.set_image(ortho)
        self.canvas.set_seeds(s.annotations()["seeds"])
        self.canvas.set_mode("view")
        self.view_title.setText(f"{s.id}   ·   {len(s.annotations()['seeds'])} SEEDS")
        self.stack.setCurrentIndex(1)
        self.canvas.fit()

    def _open_in_editor(self):
        if self.current is not None:
            self.openSample.emit(self.current.id)
            self.accept()

    def _remove(self):
        s = self.current
        if s is None:
            return
        if QMessageBox.question(self, "DATASET", f"Remove {s.id} from the {self.project.config.crop} dataset?\n"
                                "(The sample and its outlines stay in the project.)") != QMessageBox.Yes:
            return
        ml.remove_from_dataset(self.project, s)
        for kind in ml.INPUTS:
            ml.write_data_yaml(self.project, kind)
        self.current = None
        self.refresh()
        self.stack.setCurrentIndex(0)
