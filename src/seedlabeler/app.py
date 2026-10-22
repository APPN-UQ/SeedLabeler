"""SeedLabeler main window."""
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QInputDialog, QMenu,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import live, ml, processing, style
from .canvas import Canvas
from .project import CROP_PROFILES, Project

TABLE_COLS = [("#", "seed_id", "{:.0f}"), ("", "status", "{}"), ("LEN", "length_mm", "{:.2f}"),
              ("WID", "width_mm", "{:.2f}"), ("THK", "thickness_mm", "{:.2f}"), ("AREA", "area_mm2", "{:.1f}"),
              ("VOL", "volume_mm3", "{:.0f}"), ("AR", "aspect_ratio", "{:.2f}"), ("CIRC", "circularity", "{:.2f}"),
              ("L*", "L_mean", "{:.1f}"), ("a*", "a_mean", "{:.1f}"), ("b*", "b_mean", "{:.1f}")]
STATUS_GLYPH = {"auto": "·", "edited": "~", "manual": "+"}
SAMPLE_GLYPH = {"new": "○", "detected": "◐", "reviewed": "●", "dataset": "◆"}


class Task(QThread):
    """Run a function off the GUI thread."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kw))
        except Exception as e:  # noqa: BLE001 - surfaced in the GUI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


def caption(text):
    lab = QLabel(text.upper())
    lab.setObjectName("caption")
    return lab


def button(text, slot=None, checkable=False, primary=False, tip=None):
    b = QPushButton(text.upper())
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    if primary:
        b.setObjectName("primary")
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEEDLABELER")
        self.resize(1600, 960)
        self.project = None
        self.sample = None
        self.layers = None         # (ortho, height, valid) of the current sample
        self.height_img = None
        self.ann = None
        self.undo, self.redo = [], []
        self.tasks = set()
        self.proc = None
        self.live = None
        self._remeasure_pending = False
        self._remeasure_timer = QTimer(self, singleShot=True, interval=250, timeout=self._remeasure)
        self._build()
        self._update_enabled()
        last = QSettings("APPF", "SeedLabeler").value("last_project")
        if last and (Path(last) / "project.json").exists():
            self.open_project(last)

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 10)
        outer.setSpacing(10)

        # header
        head = QHBoxLayout()
        self.eye = style.HalEye(22)
        head.addWidget(self.eye)
        title = QLabel("SEEDLABELER")
        title.setObjectName("title")
        head.addWidget(title)
        head.addSpacing(18)
        self.project_label = QLabel("NO PROJECT")
        self.project_label.setObjectName("dim")
        head.addWidget(self.project_label)
        head.addStretch(1)
        for t, s in (("new", self.new_project), ("open", self.open_project_dialog), ("settings", self.edit_settings)):
            b = button(t, s)
            head.addWidget(b)
            if t == "settings":
                self.b_settings = b
        outer.addLayout(head)
        outer.addWidget(self._hairline())

        body = QHBoxLayout()
        body.setSpacing(18)
        outer.addLayout(body, 1)

        # left: samples
        left = QVBoxLayout()
        left.addWidget(caption("samples"))
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._sample_changed)
        self.sample_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sample_list.customContextMenuRequested.connect(self._sample_menu)
        self.sample_list.installEventFilter(self)
        left.addWidget(self.sample_list, 1)
        left.addWidget(caption("sample id"))
        self.sample_id = QLineEdit()
        self.sample_id.setPlaceholderText("SCAN OR TYPE")
        self.sample_id.setToolTip("Barcode scanners type the code and press Enter.\n"
                                  "With AUTO on, Enter captures straight away under this ID.")
        self.sample_id.returnPressed.connect(self._id_entered)
        left.addWidget(self.sample_id)
        idrow = QHBoxLayout()
        self.b_auto = button("auto", checkable=True, tip="Capture as soon as an ID is scanned")
        self.b_auto.setChecked(True)
        idrow.addWidget(self.b_auto)
        idrow.addStretch(1)
        left.addLayout(idrow)
        self.b_import = button("import pair", self.import_pair, tip="Add a left/right image pair")
        self.b_capture = button("capture", self.capture, primary=True, tip="Fire both cameras (PC Remote mode)")
        left.addWidget(self.b_import)
        left.addWidget(self.b_capture)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(230)
        body.addWidget(lw)
        body.addWidget(self._vline())

        # centre: tools + canvas
        centre = QVBoxLayout()
        tools = QHBoxLayout()
        self.b_select = button("select", lambda: self._set_mode("select"), checkable=True)
        self.b_draw = button("draw", lambda: self._set_mode("draw"), checkable=True)
        self.b_select.setChecked(True)
        self.b_delete = button("delete", lambda: self.canvas.delete_selected())
        self.b_undo = button("undo", self.do_undo)
        self.b_redo = button("redo", self.do_redo)
        self.b_view = button("height", self.toggle_view, checkable=True)
        self.b_fit = button("fit", lambda: self.canvas.fit())
        for w in (self.b_select, self.b_draw):
            tools.addWidget(w)
        tools.addSpacing(16)
        for w in (self.b_delete, self.b_undo, self.b_redo):
            tools.addWidget(w)
        tools.addStretch(1)
        self.b_live = button("live", self.toggle_live, checkable=True, tip="Live view from the left camera")
        self.b_live_det = button("detect", self._live_detect_changed, checkable=True,
                                 tip="Outline and count seeds on the live view (framing aid)")
        self.live_count = QLabel("")
        self.live_count.setObjectName("readout")
        tools.addWidget(self.b_live)
        tools.addWidget(self.b_live_det)
        tools.addWidget(self.live_count)
        tools.addSpacing(16)
        tools.addWidget(self.b_view)
        tools.addWidget(self.b_fit)
        centre.addLayout(tools)
        self.canvas = Canvas()
        self.canvas.edited.connect(self._edited)
        self.canvas.seedSelected.connect(self._canvas_selected)
        centre.addWidget(self.canvas, 1)
        cw = QWidget()
        cw.setLayout(centre)
        body.addWidget(cw, 1)
        body.addWidget(self._vline())

        # right: readout, seeds, model
        right = QVBoxLayout()
        right.addWidget(caption("sample"))
        self.readout = QLabel("—")
        self.readout.setObjectName("readout")
        self.readout.setTextFormat(Qt.PlainText)
        right.addWidget(self.readout)
        det = QHBoxLayout()
        self.detector = QComboBox()
        self.detector.activated.connect(self._detector_changed)
        det.addWidget(self.detector, 1)
        self.b_detect = button("detect", self.detect, primary=True)
        det.addWidget(self.b_detect)
        right.addLayout(det)
        rev = QHBoxLayout()
        self.b_reviewed = button("reviewed", self.toggle_reviewed, checkable=True,
                                 tip="Mark this sample's outlines as checked")
        self.b_dataset = button("add to dataset", self.add_to_dataset,
                                tip="Use this sample's outlines to train models for this crop")
        rev.addWidget(self.b_reviewed)
        rev.addWidget(self.b_dataset)
        rev.addStretch(1)
        right.addLayout(rev)
        right.addSpacing(8)
        right.addWidget(caption("seeds"))
        self.table = QTableWidget(0, len(TABLE_COLS))
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self._table_selected)
        right.addWidget(self.table, 1)
        right.addSpacing(8)
        right.addWidget(caption("model"))
        self.model_readout = QLabel("—")
        self.model_readout.setObjectName("readout")
        right.addWidget(self.model_readout)
        mrow = QHBoxLayout()
        self.b_train = button("train", self.train_dialog)
        self.b_export = button("export csv", self.export, primary=True,
                               tip="Write all samples' traits to exports/")
        mrow.addWidget(self.b_train)
        mrow.addStretch(1)
        mrow.addWidget(self.b_export)
        right.addLayout(mrow)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(560)
        body.addWidget(rw)

        outer.addWidget(self._hairline())
        self.status = QLabel("")
        self.status.setObjectName("dim")
        outer.addWidget(self.status)
        self.setCentralWidget(root)

    def _hairline(self):
        w = QWidget()
        w.setFixedHeight(1)
        w.setStyleSheet(f"background: {style.LINE};")
        return w

    def _vline(self):
        w = QWidget()
        w.setFixedWidth(1)
        w.setStyleSheet(f"background: {style.LINE};")
        return w

    def say(self, msg):
        self.status.setText(msg.upper())

    def _update_enabled(self):
        has_p, has_s = self.project is not None, self.sample is not None
        processed = has_s and self.sample.processed
        for w in (self.b_settings, self.b_import, self.b_capture, self.b_export, self.b_train, self.detector,
                  self.b_live, self.sample_id, self.b_auto):
            w.setEnabled(has_p)
        self.b_live_det.setEnabled(has_p)
        if self.live is not None:
            processed = False  # no editing while the canvas shows the live view
        self.b_detect.setEnabled(has_s)
        for w in (self.b_select, self.b_draw, self.b_delete, self.b_view, self.b_fit,
                  self.b_reviewed, self.b_dataset):
            w.setEnabled(processed)
        self.b_undo.setEnabled(processed and bool(self.undo))
        self.b_redo.setEnabled(processed and bool(self.redo))

    # -------------------------------------------------------------- tasks
    def run_task(self, fn, *args, done=None, msg=None, **kw):
        t = Task(fn, *args, **kw)
        self.tasks.add(t)
        self.eye.set_busy(True)
        if msg:
            self.say(msg)

        def finish():
            self.tasks.discard(t)
            self.eye.set_busy(bool(self.tasks) or self.proc is not None)

        t.done.connect(lambda r: (finish(), done(r) if done else None))
        t.failed.connect(lambda e: (finish(), self._error(e)))
        t.start()
        return t

    def _error(self, e):
        self.say(e)
        QMessageBox.warning(self, "SEEDLABELER", e)

    # ------------------------------------------------------------ project
    def new_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new project")
        if not d:
            return
        if (Path(d) / "project.json").exists():
            return self.open_project(d)
        p = Project.create(d, Path(d).name)
        self.open_project(d)
        self.edit_settings()
        return p

    def open_project_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open project folder")
        if d:
            self.open_project(d)

    def open_project(self, d):
        try:
            self.project = Project.open(d)
        except Exception as e:  # noqa: BLE001
            return self._error(f"Not a project: {e}")
        QSettings("APPF", "SeedLabeler").setValue("last_project", str(d))
        self._refresh_header()
        self._refresh_models()
        self._refresh_samples()

    def _refresh_header(self):
        c = self.project.config
        self.project_label.setText(f"{c.name}   ·   {c.crop}".upper())

    def edit_settings(self):
        if not self.project:
            return
        if SettingsDialog(self.project, self).exec():
            self.project.save()
            self._refresh_header()
            self._refresh_models()
            self._update_model_readout()

    # ------------------------------------------------------------ samples
    def _refresh_samples(self, select_id=None):
        cur = select_id or (self.sample.id if self.sample else None)
        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        row = -1
        for i, s in enumerate(self.project.samples()):
            it = QListWidgetItem(f"{SAMPLE_GLYPH[s.status()]}  {s.id}")
            it.setData(Qt.UserRole, str(s.path))
            self.sample_list.addItem(it)
            if s.id == cur:
                row = i
        self.sample_list.blockSignals(False)
        if row < 0 and self.sample_list.count():
            row = 0
        if row >= 0:
            self.sample_list.setCurrentRow(row)
            self._sample_changed(row)
        else:
            self.sample = None
            self._show_sample()

    def _sample_changed(self, row):
        if row < 0 or not self.project:
            return
        self.stop_live()
        path = Path(self.sample_list.item(row).data(Qt.UserRole))
        self.sample = next((s for s in self.project.samples() if s.path == path), None)
        self.undo, self.redo = [], []
        self._show_sample()

    def _show_sample(self):
        self.canvas.clear()
        self.layers = self.height_img = self.ann = None
        self.table.setRowCount(0)
        self.readout.setText("—")
        if self.sample and self.sample.processed:
            self.layers = processing.load_layers(self.sample)
            self.ann = self.sample.annotations()
            self.b_view.setChecked(False)
            self.canvas.set_image(self.layers[0])
            self.canvas.set_seeds(self.ann["seeds"])
            QTimer.singleShot(0, self.canvas.fit)
            self.b_reviewed.setChecked(bool(self.ann.get("reviewed")))
            self._load_table()
        elif self.sample:
            self.readout.setText("NOT DETECTED YET")
        self._update_enabled()
        self._update_model_readout()

    def import_pair(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose the LEFT and RIGHT images (left first)", "",
                                                "Images (*.jpg *.jpeg *.png *.tif)")
        if len(files) != 2:
            if files:
                self._error("Select exactly two images: left and right")
            return
        left, right = files
        if "right" in Path(left).parts[-2].lower() or "left" in Path(right).parts[-2].lower():
            left, right = right, left  # files picked from left/ and right/ folders
        s = self.project.add_pair(left, right, self._take_id())
        self._refresh_samples(select_id=s.id)
        self.detect()

    def _take_id(self):
        sid = self.sample_id.text().strip()
        self.sample_id.clear()
        return sid or None

    def _id_entered(self):
        if self.b_auto.isChecked() and self.sample_id.text().strip():
            self.capture()

    def capture(self):
        if self.proc is not None:
            return
        self.stop_live()  # the cameras must be free for the capture process
        sid = self._take_id()
        tmp = Path(tempfile.mkdtemp(prefix="seedlabeler_"))
        args = ["-m", "sonyrx0.camera.stereo_capture", "--out", str(tmp), "--rig", str(self.project.rig_file)]
        cam = self.project.config.camera_args.strip()

        def after_capture():
            lefts = sorted((tmp / "left").glob("*.JPG")) + sorted((tmp / "left").glob("*.jpg"))
            rights = sorted((tmp / "right").glob("*.JPG")) + sorted((tmp / "right").glob("*.jpg"))
            if not lefts or not rights:
                return self._error("Capture failed: no images came back. Are both cameras in PC Remote mode?")
            s = self.project.add_pair(lefts[-1], rights[-1], sid or self.project.new_sample_id())
            shutil.rmtree(tmp, ignore_errors=True)
            self._refresh_samples(select_id=s.id)
            self.detect()
            self.sample_id.setFocus()  # ready for the next scan

        if cam:
            self._run_process(["-m", "sonyrx0.camera.settings", *cam.split()], "SETTING CAMERAS",
                              lambda: self._run_process(args, "CAPTURING", after_capture))
        else:
            self._run_process(args, "CAPTURING", after_capture)

    def _run_process(self, args, msg, on_ok, log=None):
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(self.project.root))
        out = []

        def read():
            text = bytes(self.proc.readAll()).decode(errors="replace")
            out.append(text)
            if log:
                log(text)

        def finished(code, _status):
            self.proc = None
            self.eye.set_busy(bool(self.tasks))
            if code == 0:
                on_ok()
            else:
                self._error(f"{msg.title()} failed:\n" + "".join(out)[-1500:])

        self.proc.readyRead.connect(read)
        self.proc.finished.connect(finished)
        self.eye.set_busy(True)
        self.say(msg)
        self.proc.start(sys.executable, args)
        return self.proc

    # ------------------------------------------------------------ live view
    def toggle_live(self):
        if self.b_live.isChecked():
            self.start_live()
        else:
            self.stop_live()
            self._show_sample()

    def start_live(self):
        if self.live is not None or self.proc is not None:
            self.b_live.setChecked(self.live is not None)
            return
        st = self.project.config.settings
        self.live = live.LiveThread(live.left_serial(self.project), self.b_live_det.isChecked(),
                                    live.preview_res(self.project), st.seed_width)
        self.live.frame.connect(self._live_frame)
        self.live.failed.connect(self._live_failed)
        self._live_first = True
        self.b_live.setChecked(True)
        self.live.start()
        self.eye.set_busy(True)
        self.say("live")
        self._update_enabled()

    def stop_live(self):
        if self.live is None:
            return
        self.live.stop()
        self.live = None
        self.b_live.setChecked(False)
        self.live_count.setText("")
        self.eye.set_busy(bool(self.tasks) or self.proc is not None)
        self._update_enabled()

    def _live_failed(self, e):
        self.stop_live()
        self._show_sample()
        self._error(e)

    def _live_frame(self, img, n):
        if self.live is None:
            return
        if self._live_first:
            self.canvas.clear()
            self.canvas.set_image(img)
            # dashed box: keep seeds inside it so they fall where stereo gives depth
            zone = live.depth_zone(self.project, (img.shape[1], img.shape[0]))
            self.canvas.set_guide(None if zone is None else zone.tolist())
            self.canvas.fit()
            self._live_first = False
        else:
            self.canvas.swap_image(img)
        self.live_count.setText(f"N {n}" if n >= 0 else "")

    def _live_detect_changed(self):
        if self.live is not None:
            self.live.detect = self.b_live_det.isChecked()

    # ------------------------------------------------------------ sample management
    def _sample_menu(self, pos):
        if self.sample is None:
            return
        m = QMenu(self)
        m.addAction("RENAME", self.rename_sample)
        m.addAction("DELETE", self.delete_sample)
        m.exec(self.sample_list.mapToGlobal(pos))

    def eventFilter(self, obj, e):
        if obj is self.sample_list and e.type() == e.Type.KeyPress and e.key() == Qt.Key_Delete:
            self.delete_sample()
            return True
        return super().eventFilter(obj, e)

    def delete_sample(self):
        s = self.sample
        if s is None:
            return
        if QMessageBox.question(self, "DELETE", f"Delete sample {s.id}, its images and results?") != QMessageBox.Yes:
            return
        self._remeasure_timer.stop()
        self.canvas.clear()
        self.layers = self.ann = None
        self.sample = None
        try:
            self.project.delete_sample(s)
            self.say(f"deleted {s.id}")
        except Exception as e:  # noqa: BLE001
            self._error(str(e))
        finally:
            self._refresh_samples()
            self._update_model_readout()

    def rename_sample(self):
        s = self.sample
        if s is None:
            return
        new, ok = QInputDialog.getText(self, "RENAME", "Sample ID:", text=s.id)
        if not ok or not new.strip() or new.strip() == s.id:
            return
        self.canvas.clear()
        self.layers = None
        r = self.project.rename_sample(s, new.strip())
        self._refresh_samples(select_id=r.id)
        self.say(f"renamed {s.id} -> {r.id}")

    def closeEvent(self, e):
        self.stop_live()
        super().closeEvent(e)

    # ------------------------------------------------------------ detect
    def _refresh_models(self):
        self.detector.blockSignals(True)
        self.detector.clear()
        self.detector.addItem("CLASSIC", "classic")
        if self.project:
            for m in ml.list_models(self.project):
                score = f"  mAP {m['mask_map50_95']:.2f}" if m["mask_map50_95"] is not None else ""
                kind = "  C+H" if m.get("input") == "rgbh" else ""
                rel = Path(m["path"]).relative_to(self.project.root).as_posix()
                self.detector.addItem(f"MODEL {m['name']}{kind}{score}", rel)
            i = self.detector.findData(self.project.config.detector)
            self.detector.setCurrentIndex(max(i, 0))
        self.detector.blockSignals(False)

    def _detector_changed(self, _i):
        self.project.config.detector = self.detector.currentData()
        self.project.save()

    def detect(self):
        if not self.sample:
            return
        s, det = self.sample, self.project.config.detector
        conf = self.project.config.settings.model_conf

        def work():
            fn = None if det == "classic" else ml.model_detector(self.project.path(det), conf)
            return processing.detect(self.project, s, fn)

        def done(_ann):
            n = len(s.annotations()["seeds"])
            self.say(f"{s.id}: {n} seeds detected")
            self._refresh_samples(select_id=s.id)

        self.run_task(work, done=done, msg=f"detecting {s.id}")

    # ------------------------------------------------------------ editing
    def _set_mode(self, mode):
        self.b_select.setChecked(mode == "select")
        self.b_draw.setChecked(mode == "draw")
        self.canvas.set_mode(mode)
        self.canvas.setFocus()

    def _edited(self, seeds):
        self.undo.append(self.ann["seeds"])
        self.redo.clear()
        self._apply_seeds(seeds)

    def _apply_seeds(self, seeds, redraw=False):
        self.ann["seeds"] = seeds
        self.sample.save_annotations(self.ann)
        if redraw:
            self.canvas.set_seeds(seeds)
        self._update_enabled()
        self._remeasure_timer.start()

    def do_undo(self):
        if self.undo:
            self.redo.append(self.ann["seeds"])
            self._apply_seeds(self.undo.pop(), redraw=True)

    def do_redo(self):
        if self.redo:
            self.undo.append(self.ann["seeds"])
            self._apply_seeds(self.redo.pop(), redraw=True)

    def _remeasure(self):
        if any(isinstance(t, Task) and getattr(t, "_kind", "") == "measure" for t in self.tasks):
            self._remeasure_pending = True
            return
        s, ann, layers = self.sample, {**self.ann, "seeds": list(self.ann["seeds"])}, self.layers

        def done(_r):
            if self._remeasure_pending:
                self._remeasure_pending = False
                self._remeasure_timer.start()
            if s is self.sample:
                self._load_table()

        t = self.run_task(processing.remeasure, self.project, s, ann, layers, done=done)
        t._kind = "measure"

    def toggle_view(self):
        if not self.layers:
            return
        if self.b_view.isChecked():
            if self.height_img is None:
                self.height_img = processing.height_preview(self.layers[1])
            self.canvas.swap_image(self.height_img)
            self.b_view.setText("COLOUR")
        else:
            self.canvas.swap_image(self.layers[0])
            self.b_view.setText("HEIGHT")

    def toggle_reviewed(self):
        self.ann["reviewed"] = self.b_reviewed.isChecked()
        self.sample.save_annotations(self.ann)
        self._refresh_list_glyphs()

    def _refresh_list_glyphs(self):
        for i in range(self.sample_list.count()):
            it = self.sample_list.item(i)
            s = next(x for x in self.project.samples() if str(x.path) == it.data(Qt.UserRole))
            it.setText(f"{SAMPLE_GLYPH[s.status()]}  {s.id}")

    # ------------------------------------------------------------ table
    def _load_table(self):
        import pandas as pd
        if not self.sample.seeds_csv.exists():
            return
        df = pd.read_csv(self.sample.seeds_csv)
        self.table.blockSignals(True)
        self.table.setRowCount(len(df))
        for r, row in df.reset_index(drop=True).iterrows():
            for c, (_, key, fmt) in enumerate(TABLE_COLS):
                v = row.get(key, "")
                if key == "status":
                    text = STATUS_GLYPH.get(v, "")
                else:
                    text = fmt.format(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "—"
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self._readout(df)

    def _readout(self, df):
        if not len(df):
            self.readout.setText("N 0")
            return

        def stat(col, unit):
            v = df[col].dropna()
            if not len(v):
                return "—"
            cv = 100 * v.std() / v.mean() if len(v) > 1 and v.mean() else float("nan")
            return f"{v.mean():7.2f} {unit:<4} CV {cv:4.1f}%"

        st = df["status"].value_counts().to_dict() if "status" in df else {}
        lines = [f"N {len(df):<4}  AUTO {st.get('auto', 0)}  EDITED {st.get('edited', 0)}  MANUAL {st.get('manual', 0)}",
                 f"LENGTH    {stat('length_mm', 'mm')}",
                 f"WIDTH     {stat('width_mm', 'mm')}",
                 f"THICKNESS {stat('thickness_mm', 'mm')}",
                 f"VOLUME    {stat('volume_mm3', 'mm3')}"]
        self.readout.setText("\n".join(lines))

    def _table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.canvas.select_index(rows[0].row())

    def _canvas_selected(self, i):
        self.table.blockSignals(True)
        if i >= 0 and i < self.table.rowCount():
            self.table.selectRow(i)
            self.table.scrollToItem(self.table.item(i, 0))
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)

    # ------------------------------------------------------------ dataset / model
    def add_to_dataset(self):
        s = self.sample

        def done(split):
            self.ann = s.annotations()
            self.b_reviewed.setChecked(True)
            self._refresh_list_glyphs()
            self._update_model_readout()
            self.say(f"{s.id} added to the {self.project.config.crop} dataset ({split})")

        self.run_task(ml.add_to_dataset, self.project, s, done=done, msg="adding to dataset")

    def _update_model_readout(self):
        if not self.project:
            self.model_readout.setText("—")
            return
        c = ml.dataset_counts(self.project)
        models = ml.list_models(self.project)
        best = models[0] if models else None
        lines = [f"CROP      {self.project.config.crop.upper()}",
                 f"DATASET   TRAIN {c['train'][0]} SAMPLES / {c['train'][2]} SEEDS   "
                 f"VAL {c['val'][0]} / {c['val'][2]}",
                 f"MODELS    {len(models)}" + (f"   LATEST {best['name']}" if best else "")]
        if best and best["mask_map50_95"] is not None:
            lines.append(f"          MASK mAP50-95 {best['mask_map50_95']:.3f}   mAP50 {best['mask_map50']:.3f}")
        self.model_readout.setText("\n".join(lines))

    def train_dialog(self):
        c = ml.dataset_counts(self.project)
        if c["train"][0] == 0:
            return self._error("The dataset for this crop is empty. Review samples and ADD TO DATASET first.")
        TrainDialog(self).exec()
        self._refresh_models()
        self._update_model_readout()

    def export(self):
        paths = processing.export_project(self.project)
        self.say("exported " + ", ".join(p.name for p in paths) if paths else "nothing to export yet")


class SettingsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("PROJECT SETTINGS")
        c, s = project.config, project.config.settings
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)
        self.name = QLineEdit(c.name)
        self.crop = QComboBox()
        self.crop.setEditable(True)
        self.crop.addItems(list(CROP_PROFILES))
        self.crop.setCurrentText(c.crop)
        self.crop.currentTextChanged.connect(self._profile)
        self.calib = self._path_row(c.calib, "Stereo calibration (*.npz)")
        self.stage = self._path_row(c.stage, "Stage plane (*.npz)")
        self.camera = QLineEdit(c.camera_args)
        self.camera.setPlaceholderText("--mode m --shutter 1/125 --iso 400")
        self.seed_width = self._spin(s.seed_width, 0.5, 50, " mm")
        self.min_thick = self._spin(s.min_thickness, 0, 20, " mm")
        self.min_area = self._spin(s.min_area, 0, 500, " mm²")
        self.min_depth = self._spin(s.min_depth, 50, 3000, " mm")
        self.max_depth = self._spin(s.max_depth, 50, 3000, " mm")
        self.conf = self._spin(s.model_conf, 0.05, 0.95, "", step=0.05)
        for label, w in (("NAME", self.name), ("CROP", self.crop), ("CALIBRATION", self.calib[0]),
                         ("STAGE", self.stage[0]), ("CAMERA", self.camera), ("SEED WIDTH", self.seed_width),
                         ("MIN THICKNESS", self.min_thick), ("MIN AREA", self.min_area),
                         ("DEPTH FROM", self.min_depth), ("DEPTH TO", self.max_depth), ("MODEL CONF", self.conf)):
            form.addRow(caption(label), w)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("cancel", self.reject))
        row.addWidget(button("save", self.accept, primary=True))
        form.addRow(row)
        self.setMinimumWidth(560)

    def _spin(self, v, lo, hi, suffix, step=0.5):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setSuffix(suffix)
        w.setValue(v)
        return w

    def _path_row(self, value, filt):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(value)
        h.addWidget(e, 1)

        def browse():
            f, _ = QFileDialog.getOpenFileName(self, filt, str(self.project.root), filt)
            if f:
                e.setText(f)

        h.addWidget(button("…", browse))
        return w, e

    def _profile(self, crop):
        p = CROP_PROFILES.get(crop)
        if p:
            self.seed_width.setValue(p["seed_width"])
            self.min_thick.setValue(p["min_thickness"])

    def accept(self):
        c, s = self.project.config, self.project.config.settings
        c.name, c.crop = self.name.text().strip() or c.name, self.crop.currentText().strip().lower() or c.crop
        c.calib, c.stage = self.calib[1].text().strip(), self.stage[1].text().strip()
        c.camera_args = self.camera.text().strip()
        s.seed_width, s.min_thickness, s.min_area = self.seed_width.value(), self.min_thick.value(), self.min_area.value()
        s.min_depth, s.max_depth, s.model_conf = self.min_depth.value(), self.max_depth.value(), self.conf.value()
        if c.detector != "classic" and f"/{c.crop}/" not in c.detector:
            c.detector = "classic"  # models belong to a crop
        super().accept()


class TrainDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main, self.project = main, main.project
        self.setWindowTitle("TRAIN")
        self.resize(820, 560)
        v = QVBoxLayout(self)
        c = ml.dataset_counts(self.project)
        info = QLabel(f"{self.project.config.crop.upper()}   ·   TRAIN {c['train'][0]} SAMPLES, {c['train'][1]} TILES, "
                      f"{c['train'][2]} SEEDS   ·   VAL {c['val'][0]} SAMPLES, {c['val'][1]} TILES, {c['val'][2]} SEEDS")
        info.setObjectName("readout")
        v.addWidget(info)
        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(150)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["640", "960"])  # dataset tiles are 640 px
        self.imgsz.setCurrentText("640")
        self.base = QComboBox()
        self.base.addItem(f"PRETRAINED  {ml.BASE_MODEL}", ml.BASE_MODEL)
        self.base.addItem("PRETRAINED  yolo11n-seg.pt (faster)", "yolo11n-seg.pt")
        for m in ml.list_models(self.project):
            self.base.addItem(f"CONTINUE  {m['name']}", str(m["path"]))
        self.input = QComboBox()
        for k, v in ml.INPUTS.items():
            self.input.addItem(v, k)
        self.input.setCurrentIndex(1)
        self.input.setToolTip("COLOUR + HEIGHT also shows the model each pixel's height above the stage,\n"
                              "so it can tell seeds from shadows and stains of the same colour.")
        self.name = QLineEdit(time.strftime("run_%Y%m%d_%H%M%S"))
        for label, w in (("INPUT", self.input), ("EPOCHS", self.epochs), ("IMAGE SIZE", self.imgsz),
                         ("START FROM", self.base), ("RUN NAME", self.name)):
            form.addRow(caption(label), w)
        v.addLayout(form)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        v.addWidget(self.log, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_close = button("close", self.reject)
        self.b_start = button("start", self.start, primary=True)
        row.addWidget(self.b_close)
        row.addWidget(self.b_start)
        v.addLayout(row)

    def start(self):
        if self.main.proc is not None:
            return
        kind = self.input.currentData()
        self.b_start.setEnabled(False)
        self.log.appendPlainText("PREPARING DATASET...")
        QApplication.processEvents()
        ml.sync_datasets(self.project)  # every dataset sample in both input formats
        argv = ml.train_command(self.project, self.name.text().strip(), self.epochs.value(),
                                int(self.imgsz.currentText()), self.base.currentData(), kind)
        self.log.appendPlainText(" ".join(argv[1:]) + "\n")

        def ok():
            self.log.appendPlainText("\nDONE. The model is now available in the detector list.")
            self.b_start.setEnabled(True)
            self.main.say(f"trained {self.name.text()}")

        def log(text):
            # Ultralytics redraws progress lines with \r: keep the last state of each line
            for line in text.replace("\r\n", "\n").split("\n"):
                line = line.split("\r")[-1]
                if line.strip():
                    self.log.appendPlainText(line)

        self.main._run_process(argv[1:], "TRAINING", ok, log=log)

    def reject(self):
        if self.main.proc is not None and not self.b_start.isEnabled():
            if QMessageBox.question(self, "TRAIN", "Training is running. Stop it?") != QMessageBox.Yes:
                return
            self.main.proc.kill()
        super().reject()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SeedLabeler")
    style.apply(app)
    w = MainWindow()
    if len(sys.argv) > 1:
        w.open_project(sys.argv[1])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
