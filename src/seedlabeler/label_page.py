"""LABEL mode: capture (or import), detect + SAM, correct outlines, build the dataset, train."""
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import live, ml, processing, sam, sheet, style
from .canvas import Canvas
from .dataset_view import DatasetDialog
from .detection import run_detection
from .project import CAMERA_ARGS_HINT, CAMERA_SYSTEMS, CROP_PROFILES
from .widgets import button, caption, vline

TABLE_COLS = [("#", "seed_id", "{:.0f}"), ("", "status", "{}"), ("TAG", "label", "{}"),
              ("LEN", "length_mm", "{:.2f}"),
              ("WID", "width_mm", "{:.2f}"), ("THK", "thickness_mm", "{:.2f}"), ("AREA", "area_mm2", "{:.1f}"),
              ("VOL", "volume_mm3", "{:.0f}"), ("AR", "aspect_ratio", "{:.2f}"), ("CIRC", "circularity", "{:.2f}"),
              ("L*", "L_mean", "{:.1f}"), ("a*", "a_mean", "{:.1f}"), ("b*", "b_mean", "{:.1f}")]
STATUS_GLYPH = {"auto": "·", "edited": "~", "manual": "+"}
SAMPLE_GLYPH = {"new": "○", "detected": "◐", "reviewed": "◐", "dataset": "◆"}


class LabelPage(QWidget):
    def __init__(self, shell):
        super().__init__()
        self.setObjectName("page")
        self.shell = shell
        self.project = None
        self.sample = None
        self.layers = None         # (ortho, height, valid) of the current sample
        self.height_img = None
        self.ann = None
        self.undo, self.redo = [], []
        self.live_on = False
        self.sam_active = None     # the outline SAM is currently drawing: box, clicks, index
        self._sam_busy = False
        self._remeasure_pending = False
        self._remeasure_timer = QTimer(self, singleShot=True, interval=250, timeout=self._remeasure)
        self._build()
        self._update_enabled()

    # ------------------------------------------------------------------ UI
    def _build(self):
        body = QHBoxLayout(self)
        body.setContentsMargins(18, 12, 18, 8)
        body.setSpacing(16)

        # left: samples, id, capture
        left = QVBoxLayout()
        left.addWidget(caption("samples"))
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._sample_changed)
        self.sample_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sample_list.customContextMenuRequested.connect(self._sample_menu)
        self.sample_list.installEventFilter(self)
        left.addWidget(self.sample_list, 1)
        left.addSpacing(6)
        left.addWidget(caption("sample id"))
        self.sample_id = QLineEdit()
        self.sample_id.setPlaceholderText("Scan or type")
        self.sample_id.setToolTip("Barcode scanners type the code and press Enter.\n"
                                  "With AUTO on, Enter captures straight away under this ID.")
        self.sample_id.returnPressed.connect(self._id_entered)
        left.addWidget(self.sample_id)
        idrow = QHBoxLayout()
        self.b_auto = button("auto", checkable=True, tip="Capture as soon as an ID is scanned")
        self.b_auto.setChecked(True)
        self.b_import = button("import pair", self.import_pair, tip="Add a left/right image pair from files")
        idrow.addWidget(self.b_auto)
        idrow.addStretch(1)
        idrow.addWidget(self.b_import)
        left.addLayout(idrow)
        self.b_capture = button("capture", self.capture, primary=True, tip="Capture (both RX0s, or the OAK-D), then detect")
        left.addWidget(self.b_capture)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(240)
        body.addWidget(lw)
        body.addWidget(vline())

        # centre: tools + canvas
        centre = QVBoxLayout()
        tools = QHBoxLayout()
        self.b_select = button("select", lambda: self._set_mode("select"), checkable=True)
        self.b_box = button("box", lambda: self._set_mode("box"), checkable=True,
                            tip="SAM: drag a box around a seed, or click a seed.\n"
                                "Then click to include more, Shift/right-click to exclude.")
        self.b_select.setChecked(True)
        self.b_samfix = button("sam fix", self.sam_resegment, tip="Redraw the selected seed with SAM (R)")
        self.b_delete = button("delete", lambda: self.canvas.delete_selected())
        self.b_undo = button("undo", self.do_undo)
        self.b_redo = button("redo", self.do_redo)
        self.b_view = button("depth map", self.toggle_view, checkable=True)
        self.b_fit = button("fit", lambda: self.canvas.fit())
        self.b_live = button("live", self.toggle_live, checkable=True,
                             tip="Live view (RX0: left camera; OAK-D: with the stage distance and corner markers)")
        for w in (self.b_select, self.b_box):
            tools.addWidget(w)
        tools.addSpacing(14)
        for w in (self.b_samfix, self.b_delete, self.b_undo, self.b_redo):
            tools.addWidget(w)
        tools.addStretch(1)
        tools.addWidget(caption("opacity"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(5)
        self.opacity_slider.setFixedWidth(80)
        self.opacity_slider.setToolTip("Seed overlay opacity")
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        tools.addWidget(self.opacity_slider)
        tools.addSpacing(14)
        self.b_focus = button("set focus", self._focus_lock,
                              tip="Autofocus both cameras on the stage, then lock to manual")
        for w in (self.b_focus, self.b_live, self.b_view, self.b_fit):
            tools.addWidget(w)
        centre.addLayout(tools)
        self.canvas = Canvas()
        self.canvas.edited.connect(self._edited)
        self.canvas.seedSelected.connect(self._canvas_selected)
        self.canvas.seedLabelled.connect(self._seed_labelled)
        self.canvas.boxDrawn.connect(self._sam_box)
        self.canvas.pointClicked.connect(self._sam_click)
        self.canvas.resegmentRequested.connect(self.sam_resegment)
        centre.addWidget(self.canvas, 1)
        cw = QWidget()
        cw.setLayout(centre)
        body.addWidget(cw, 1)
        body.addWidget(vline())

        # right: readout, detection, seeds, dataset/model
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
        self.b_detect = button("detect", self.detect, secondary=True,
                               tip="Find the seeds, then SAM redraws every outline (SAM REFINE in settings)")
        self.b_refine = button("refine all", self.sam_refine_all, tip="Redraw every automatic outline with SAM")
        det.addWidget(self.b_detect)
        det.addWidget(self.b_refine)
        right.addLayout(det)
        right.addSpacing(6)
        right.addWidget(caption("seeds"))
        self.table = QTableWidget(0, len(TABLE_COLS))
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(21)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self._table_selected)
        right.addWidget(self.table, 1)
        right.addSpacing(6)
        right.addWidget(caption("dataset & model"))
        self.model_readout = QLabel("—")
        self.model_readout.setObjectName("readout")
        right.addWidget(self.model_readout)
        mrow = QHBoxLayout()
        self.b_dataset = button("add to dataset", self.add_to_dataset, primary=True,
                                tip="Use this sample's outlines to train models for this crop")
        self.b_view_ds = button("view dataset", self.view_dataset)
        self.b_train = button("train", self.train_dialog)
        mrow.addWidget(self.b_dataset)
        mrow.addWidget(self.b_view_ds)
        mrow.addStretch(1)
        mrow.addWidget(self.b_train)
        right.addLayout(mrow)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(560)
        body.addWidget(rw)

    def _update_enabled(self):
        has_p, has_s = self.project is not None, self.sample is not None
        processed = has_s and self.sample.processed and not self.live_on
        for w in (self.b_import, self.b_capture, self.b_train, self.detector, self.b_live, self.sample_id,
                  self.b_auto, self.b_view_ds):
            w.setEnabled(has_p)
        if has_p and self.project.config.camera == "oak":
            self.b_import.setEnabled(False)  # an OAK sample is colour + depth from the camera, not a file pair
        self.b_detect.setEnabled(has_s)
        for w in (self.b_select, self.b_box, self.b_samfix, self.b_refine, self.b_delete,
                  self.b_view, self.b_fit, self.b_dataset):
            w.setEnabled(processed)
        self.b_undo.setEnabled(processed and bool(self.undo))
        self.b_redo.setEnabled(processed and bool(self.redo))

    # ------------------------------------------------------------ page lifecycle
    def set_project(self, project):
        self.project = project
        self.sample = None
        self.canvas.seed_labels = [""] + list(project.config.settings.seed_labels)
        self._refresh_models()
        self._refresh_samples()

    def enter(self):
        self.start_live()

    def leave(self):
        self.stop_live()

    def say(self, msg):
        self.shell.say(msg)

    def run_task(self, *a, **kw):
        return self.shell.run_task(*a, **kw)

    # ------------------------------------------------------------ samples
    def _refresh_samples(self, select_id=None):
        if not self.project:
            return
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

    def open_sample(self, sample_id):
        self._refresh_samples(select_id=sample_id)

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
            self.b_view.setText("DEPTH MAP")
            self.canvas.set_image(self.layers[0])
            self.canvas.set_seeds(self.ann["seeds"])
            QTimer.singleShot(0, self.canvas.fit)
            self._load_table()
        elif self.sample:
            self.readout.setText("Not detected yet: press DETECT.")
        self._update_enabled()
        self._update_model_readout()

    def import_pair(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose the LEFT and RIGHT images (left first)", "",
                                                "Images (*.jpg *.jpeg *.png *.tif)")
        if len(files) != 2:
            if files:
                self.shell.error("Select exactly two images: left and right")
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
        def done(sample):
            self._refresh_samples(select_id=sample.id)
            self.detect()
            self.sample_id.setFocus()  # ready for the next scan

        was_live = self.live_on
        self.stop_live()
        self.shell.capture_sample(self._take_id(), done)
        if was_live:
            self.b_live.setChecked(False)

    def _focus_lock(self):
        self.shell.camera.focus_lock()
        self.shell.say("focusing … hold still")

    # ------------------------------------------------------------ live view
    def toggle_live(self):
        if self.b_live.isChecked():
            self.start_live()
        else:
            self.stop_live()
            self._show_sample()

    def start_live(self):
        if self.live_on:
            return
        self.live_on = True
        self._live_first = True
        self.b_live.setChecked(True)
        self.shell.camera.frame.connect(self._live_frame)
        self.shell.camera.set_live(True)
        self._update_enabled()

    def stop_live(self):
        if not self.live_on:
            return
        self.live_on = False
        self.shell.camera.set_live(False)
        try:
            self.shell.camera.frame.disconnect(self._live_frame)
        except (RuntimeError, TypeError):
            pass
        self.b_live.setChecked(False)
        self._update_enabled()

    def _live_frame(self, img):
        if not self.live_on:
            return
        if self._live_first:
            self.canvas.clear()
            self.canvas.set_image(img)
            zone = live.depth_zone(self.project, (img.shape[1], img.shape[0]))
            self.canvas.set_guide(None if zone is None else zone.tolist())
            self.canvas.fit()
            self._live_first = False
        else:
            self.canvas.swap_image(img)

    # ------------------------------------------------------------ sample management
    def _sample_menu(self, pos):
        if self.sample is None:
            return
        m = QMenu(self)
        m.addAction("Rename", self.rename_sample)
        m.addAction("Delete", self.delete_sample)
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
        if QMessageBox.question(self, "Delete", f"Delete sample {s.id}, its images and results?") != QMessageBox.Yes:
            return
        self._remeasure_timer.stop()
        self.canvas.clear()
        self.layers = self.ann = None
        self.sample = None
        try:
            self.project.delete_sample(s)
            self.say(f"deleted {s.id}")
        except Exception as e:  # noqa: BLE001
            self.shell.error(str(e))
        finally:
            self._refresh_samples()
            self._update_model_readout()

    def rename_sample(self):
        s = self.sample
        if s is None:
            return
        new, ok = QInputDialog.getText(self, "Rename", "Sample ID:", text=s.id)
        if not ok or not new.strip() or new.strip() == s.id:
            return
        self.canvas.clear()
        self.layers = None
        r = self.project.rename_sample(s, new.strip())
        self._refresh_samples(select_id=r.id)
        self.say(f"renamed {s.id} -> {r.id}")

    # ------------------------------------------------------------ detect
    def _refresh_models(self):
        self.detector.blockSignals(True)
        self.detector.clear()
        self.detector.addItem("Classic", "classic")
        if self.project:
            for m in ml.list_models(self.project):
                score = f"  mAP {m['mask_map50_95']:.2f}" if m["mask_map50_95"] is not None else ""
                kind = "  C+H" if m.get("input") == "rgbh" else ""
                rel = Path(m["path"]).relative_to(self.project.root).as_posix()
                self.detector.addItem(f"Model {m['name']}{kind}{score}", rel)
            i = self.detector.findData(self.project.config.detector)
            self.detector.setCurrentIndex(max(i, 0))
        self.detector.blockSignals(False)

    def _detector_changed(self, _i):
        self.project.config.detector = self.detector.currentData()
        self.project.save()

    def detect(self):
        if not self.sample:
            return
        s = self.sample

        def done(ann):
            n = len(ann["seeds"])
            self._refresh_samples(select_id=s.id)  # the first outlines appear now (a fresh Sample object)
            still_here = self.sample is not None and self.sample.path == s.path
            if self.project.config.settings.sam_refine and n and still_here:
                self.say(f"{s.id}: {n} seeds · sam refining the outlines")
                self.sam_refine_all()  # second pass; skipped if the outlines are edited meanwhile
            else:
                self.say(f"{s.id}: {n} seeds")

        # SAM refine is run as a second pass so the user sees (and can start on) outlines within seconds
        self.run_task(run_detection, self.project, s, refine=False, done=done, msg=f"detecting {s.id}")

    # ------------------------------------------------------------ editing
    def _set_mode(self, mode):
        self.b_select.setChecked(mode == "select")
        self.b_box.setChecked(mode == "box")
        self.sam_active = None
        self.canvas.set_mode(mode)
        self.canvas.setFocus()
        if mode == "box":
            self._sam_preload()

    def _seed_labelled(self, index, label):
        if not self.ann or index >= len(self.ann["seeds"]):
            return
        seeds = [dict(s) for s in self.ann["seeds"]]
        if label:
            seeds[index]["label"] = label
        else:
            seeds[index].pop("label", None)
        self._edited(seeds)

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
        if any(getattr(t, "_kind", "") == "measure" for t in self.shell.tasks):
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

    def _opacity_changed(self, value):
        self.canvas.set_seed_opacity(value / 100.0)

    def toggle_view(self):
        if not self.layers:
            return
        if self.b_view.isChecked():  # the button names what a click will show
            if self.height_img is None:
                self.height_img = processing.height_preview(self.layers[1])
            self.canvas.swap_image(self.height_img)
            self.b_view.setText("RGB")
        else:
            self.canvas.swap_image(self.layers[0])
            self.b_view.setText("DEPTH MAP")

    def _refresh_list_glyphs(self):
        for i in range(self.sample_list.count()):
            it = self.sample_list.item(i)
            s = next(x for x in self.project.samples() if str(x.path) == it.data(Qt.UserRole))
            it.setText(f"{SAMPLE_GLYPH[s.status()]}  {s.id}")

    # ------------------------------------------------------------ SAM
    def _sam_preload(self):
        name = self.project.config.settings.sam_model
        if name not in sam._models:
            self.run_task(sam.get_model, name, msg=f"loading {sam.MODELS.get(name, name).lower()}",
                          done=lambda _m: self.say("sam ready"))

    def _sam_run(self, box, points, labels, on_poly):
        """Prompt SAM off the GUI thread; one prompt at a time."""
        if self._sam_busy or not self.layers:
            return
        self._sam_busy = True
        st, res = self.project.config.settings, self.sample.meta()["res"]
        o, h, _ = self.layers

        def done(poly):
            self._sam_busy = False
            if poly is None:
                self.say("sam found nothing there")
            else:
                on_poly(poly.tolist())

        t = self.run_task(sam.segment, st.sam_model, o, box, points, labels, h, res,
                          seed_px=st.seed_width / res, done=done, msg="sam")
        t.failed.connect(lambda _e: setattr(self, "_sam_busy", False))

    def _sam_box(self, x0, y0, x1, y1):
        box = (x0, y0, x1, y1)

        def add(poly):
            seeds = self.ann["seeds"] + [{"polygon": poly, "status": "manual"}]
            self.sam_active = {"index": len(seeds) - 1, "box": box, "points": [], "labels": []}
            self._edited(seeds)
            self.canvas.set_seeds(seeds, keep_selection=len(seeds) - 1)

        self._sam_run(box, None, None, add)

    def _sam_click(self, x, y, positive):
        a = self.sam_active
        if a is None:
            if not positive:
                return

            def add(poly):
                seeds = self.ann["seeds"] + [{"polygon": poly, "status": "manual"}]
                self.sam_active = {"index": len(seeds) - 1, "box": None, "points": [(x, y)], "labels": [1]}
                self._edited(seeds)
                self.canvas.set_seeds(seeds, keep_selection=len(seeds) - 1)

            return self._sam_run(None, [(x, y)], [1], add)
        pts, labs = a["points"] + [(x, y)], a["labels"] + [1 if positive else 0]

        def replace(poly):
            if a["index"] >= len(self.ann["seeds"]):
                return
            seeds = [dict(s) for s in self.ann["seeds"]]
            status = seeds[a["index"]]["status"]
            seeds[a["index"]] = {"polygon": poly, "status": "edited" if status == "auto" else status}
            a["points"], a["labels"] = pts, labs
            self._edited(seeds)
            self.canvas.set_seeds(seeds, keep_selection=a["index"])

        self._sam_run(a["box"], pts, labs, replace)

    def sam_resegment(self):
        sel = [i for i, it in enumerate(self.canvas.items_) if it.isSelected()]
        if len(sel) != 1 or not self.ann:
            return self.say("select one seed first")
        i = sel[0]
        box, centre = sam.core_prompt(self.ann["seeds"][i]["polygon"])

        def replace(poly):
            seeds = [dict(s) for s in self.ann["seeds"]]
            status = seeds[i]["status"]
            seeds[i] = {"polygon": poly, "status": "edited" if status == "auto" else status}
            self.sam_active = {"index": i, "box": box, "points": [centre], "labels": [1]}
            self._edited(seeds)
            self.canvas.set_seeds(seeds, keep_selection=i)

        self._sam_preload()
        self._sam_run(box, [centre], [1], replace)

    def sam_refine_all(self):
        if not self.ann or not self.ann["seeds"]:
            return
        st, res = self.project.config.settings, self.sample.meta()["res"]
        o, h, _ = self.layers
        s, before = self.sample, self.ann["seeds"]

        def done(result):
            seeds, n = result
            if self.sample is None or self.sample.path != s.path or self.ann is None:
                return
            if self.ann["seeds"] != before:
                return self.say("sam refine skipped: the outlines changed while it ran (use REFINE ALL to redo)")
            self._edited(seeds)
            self.canvas.set_seeds(seeds)
            self.say(f"sam redrew {n} of {len(seeds)} outlines")

        self.run_task(sam.refine_all, st.sam_model, o, before, h, res, done=done, msg="sam refining outlines")

    # ------------------------------------------------------------ table
    def _load_table(self):
        import pandas as pd
        if not self.sample or not self.sample.seeds_csv.exists():
            return
        df = pd.read_csv(self.sample.seeds_csv)
        self.table.blockSignals(True)
        self.table.setRowCount(len(df))
        for r, row in df.reset_index(drop=True).iterrows():
            for c, (_, key, fmt) in enumerate(TABLE_COLS):
                v = row.get(key, "")
                if key == "status":
                    text = STATUS_GLYPH.get(v, "")
                elif key == "label":
                    text = str(v) if pd.notna(v) else ""
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
        if 0 <= i < self.table.rowCount():
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
            self._refresh_list_glyphs()
            self._update_model_readout()
            self.shell.refresh_gates()
            self.say(f"{s.id} added to the {self.project.config.crop} dataset ({split})")

        self.run_task(ml.add_to_dataset, self.project, s, done=done, msg="adding to dataset")

    def _update_model_readout(self):
        if not self.project:
            self.model_readout.setText("—")
            return
        c = ml.dataset_counts(self.project)
        n_samples, n_seeds = ml.dataset_summary(self.project)
        models = ml.list_models(self.project)
        best = models[0] if models else None
        lines = [f"DATASET   {n_samples} SAMPLES · {n_seeds} SEEDS ({c['val'][0]} VAL)",
                 f"MODELS    {len(models)}" + (f" · LATEST {best['name']}" if best else "")]
        if best and best["mask_map50_95"] is not None:
            lines.append(f"          MASK mAP50 {best['mask_map50']:.2f} · mAP50-95 {best['mask_map50_95']:.2f}")
        self.model_readout.setText("\n".join(lines))

    def view_dataset(self):
        d = DatasetDialog(self.project, self)
        d.openSample.connect(self.open_sample)
        d.exec()
        self._refresh_list_glyphs()
        self._update_model_readout()

    def train_dialog(self):
        c = ml.dataset_counts(self.project)
        if c["train"][0] == 0:
            return self.shell.error("The dataset for this crop is empty. Correct samples and ADD TO DATASET first.")
        self.shell.train()


class SettingsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Project settings")
        c, s = project.config, project.config.settings
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(10)
        self.name = QLineEdit(c.name)
        self.crop = QComboBox()
        self.crop.setEditable(True)
        self.crop.addItems(list(CROP_PROFILES))
        self.crop.setCurrentText(c.crop)
        self.crop.currentTextChanged.connect(self._profile)
        self.system = QComboBox()
        for key, text in CAMERA_SYSTEMS.items():
            self.system.addItem(text, key)
        self.system.setCurrentIndex(max(0, self.system.findData(c.camera)))
        self.system.setEnabled(False)
        self.system.setToolTip("Camera system is set when the project is created and cannot be changed.")
        self.calib = self._path_row(c.calib, "Stereo calibration (*.npz)")
        self._shell = parent  # Shell — for camera access in calibrate/verify
        self.calib_indicator = QLabel("●")
        self.calib_indicator.setFixedWidth(18)
        self.calib_indicator.setToolTip("Calibration health: click VERIFY to check against the stage markers")
        self.calib_indicator.setStyleSheet(f"color: {style.DIM};")
        self.calib[0].layout().addWidget(self.calib_indicator)
        self.calib[0].layout().addWidget(button("calibrate", self._run_calibration,
                                                 tip="Guided calibration: capture ChArUco board pairs"))
        self.calib[0].layout().addWidget(button("verify", self._verify_calibration,
                                                 tip="Check calibration quality using the stage markers"))
        self.stage = self._path_row(c.stage, "Stage plane (*.npz)")
        self.camera = QLineEdit(c.camera_args)
        self.ground = QComboBox()
        self.ground.addItem("Stage file, else fit to the depth", "plane")
        self.ground.addItem("Corner markers (stage sheet)", "markers")
        self.ground.setToolTip("Corner markers: a printed sheet with an ArUco marker in each corner. They give the\n"
                               "stage plane in every capture and correct the depth's scale; seeds go between them.")
        self.ground.setCurrentIndex(1 if c.markers_sheet else 0)
        self.sheet = self._path_row(c.markers_sheet, "Stage sheet layout (*.json)")
        self.sheet[0].layout().addWidget(button("print sheet", self._print_sheet,
                                                tip="Make a stage sheet (A4 or A3) to print"))
        self.bar = self._spin(c.markers_bar_mm, 20, 200, " mm", step=0.5)
        self.bar.setToolTip("Measure the sheet's 100 mm bar with a ruler after printing: printers often shrink\n"
                            "the page (fit to page), and the markers' size sets every measurement's scale.")
        self.level = button("on", checkable=True,
                            tip="Measure height against the sheet around each seed instead of one plane.\n"
                                "On for the OAK-D, whose depth bends by a few mm across the view.")
        self.level.toggled.connect(lambda on: self.level.setText("ON" if on else "OFF"))
        self.level.setChecked(s.level_height)
        self.level.setText("ON" if s.level_height else "OFF")
        self.seed_width = self._spin(s.seed_width, 0.5, 50, " mm")
        self.min_thick = self._spin(s.min_thickness, 0, 20, " mm")
        self.min_area = self._spin(s.min_area, 0, 500, " mm²")
        self.min_depth = self._spin(s.min_depth, 50, 3000, " mm")
        self.max_depth = self._spin(s.max_depth, 50, 3000, " mm")
        self.conf = self._spin(s.model_conf, 0.05, 0.95, "", step=0.05)
        self.target = QSpinBox()
        self.target.setRange(1, 1000)
        self.target.setValue(s.label_target)
        self.target.setSuffix(" samples")
        self.sam_model = QComboBox()
        for k, v in sam.MODELS.items():
            self.sam_model.addItem(v.title(), k)
        self.sam_model.setCurrentIndex(max(0, self.sam_model.findData(s.sam_model)))
        self.sam_refine = button("on", checkable=True,
                                 tip="DETECT finds the seeds, then SAM redraws every outline precisely.\n"
                                     "In LABEL the first outlines show at once and SAM refines them in a second pass.")
        self.sam_refine.setChecked(s.sam_refine)
        self.sam_refine.toggled.connect(lambda on: self.sam_refine.setText("ON" if on else "OFF"))
        self.sam_refine.setText("ON" if s.sam_refine else "OFF")
        self.seed_labels_edit = QLineEdit(", ".join(s.seed_labels))
        self.seed_labels_edit.setToolTip("Comma-separated labels for tagging individual seeds (right-click a seed to tag it)")
        for lab, w in (("NAME", self.name), ("CROP", self.crop), ("CAMERA SYSTEM", self.system),
                       ("CALIBRATION", self.calib[0]), ("CAMERA", self.camera), ("GROUND PLANE", self.ground),
                       ("STAGE", self.stage[0]), ("SHEET", self.sheet[0]), ("100 MM BAR PRINTED AS", self.bar),
                       ("LEVEL HEIGHT", self.level), ("SEED WIDTH", self.seed_width),
                       ("MIN THICKNESS", self.min_thick), ("MIN AREA", self.min_area),
                       ("DEPTH FROM", self.min_depth), ("DEPTH TO", self.max_depth), ("MODEL CONF", self.conf),
                       ("SAM MODEL", self.sam_model), ("SAM REFINE", self.sam_refine),
                       ("LABEL TARGET", self.target), ("SEED LABELS", self.seed_labels_edit)):
            form.addRow(caption(lab), w)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("cancel", self.reject))
        row.addWidget(button("save", self.accept, primary=True))
        form.addRow(row)
        self.setMinimumWidth(760)
        self.system.currentIndexChanged.connect(self._update_rows)
        self.ground.currentIndexChanged.connect(self._update_rows)
        self._update_rows()

    def _update_rows(self, _i=None):
        oak, markers = self.system.currentData() == "oak", self.ground.currentData() == "markers"
        self.calib[0].setEnabled(not oak)
        self.calib_indicator.setVisible(not oak)
        self.stage[0].setEnabled(not oak and not markers)  # a stage file is in the RX0 rig's frame
        self.sheet[0].setEnabled(markers)
        self.bar.setEnabled(markers)
        self.camera.setPlaceholderText(CAMERA_ARGS_HINT["oak" if oak else "rx0"])

    def _print_sheet(self):
        paper, ok = QInputDialog.getItem(self, "Stage sheet", "Paper size (landscape)", list(sheet.PAPERS), 0, False)
        if not ok:
            return
        default = self.project.root / "calibration" / f"stage_sheet_{paper.lower()}.pdf"
        f, _ = QFileDialog.getSaveFileName(self, "Save the stage sheet", str(default), "PDF (*.pdf)")
        if not f:
            return
        js = sheet.write_sheet(f, paper)
        self.sheet[1].setText(str(js))
        self.bar.setValue(100.0)
        QDesktopServices.openUrl(QUrl.fromLocalFile(f))
        QMessageBox.information(self, "Stage sheet",
                                f"Print the {paper} sheet at 100% (\"Actual size\", not \"Fit\"), landscape, and tape "
                                "it down flat.\n\nThen measure the 100 mm bar along its bottom edge with a ruler and "
                                "enter it as 100 MM BAR PRINTED AS. Keep the seeds between the four markers.")

    def _run_calibration(self):
        if self.project.config.camera != "rx0":
            return QMessageBox.information(self, "Calibration", "The OAK-D uses its factory calibration.")
        from .stereo_calib import StereoCalibDialog
        dlg = StereoCalibDialog(self._shell, self.project, self)
        if dlg.exec() and dlg.result_path:
            rel = dlg.result_path.relative_to(self.project.root).as_posix()
            self.calib[1].setText(rel)

    def _verify_calibration(self):
        calib_path = self.calib[1].text().strip()
        if not calib_path:
            return QMessageBox.warning(self, "Verify", "No calibration file set.")
        full = self.project.path(calib_path)
        if not full.exists():
            return QMessageBox.warning(self, "Verify", f"Calibration file not found:\n{full}")
        self.calib_indicator.setStyleSheet(f"color: {style.DIM};")
        self.calib_indicator.setToolTip("verifying ...")

        def on_captured(info):
            try:
                self._shell.camera.captured.disconnect(on_captured)
            except (RuntimeError, TypeError):
                pass
            try:
                self._shell.camera.failed.disconnect(on_fail)
            except (RuntimeError, TypeError):
                pass
            from sonyrx0.stereo.common import read_image
            from .stereo_calib import verify_calibration
            left, right = info.get("left"), info.get("right")
            if not left or not right:
                self._set_calib_indicator(None, "need a stereo pair to verify")
                return
            img_l, img_r = read_image(left), read_image(right)
            quality, err, detail = verify_calibration(full, img_l, img_r)
            self._set_calib_indicator(quality, detail)
            import shutil
            shutil.rmtree(info["left"].parent, ignore_errors=True)

        def on_fail(msg, tag):
            try:
                self._shell.camera.captured.disconnect(on_captured)
            except (RuntimeError, TypeError):
                pass
            try:
                self._shell.camera.failed.disconnect(on_fail)
            except (RuntimeError, TypeError):
                pass
            self._set_calib_indicator(None, f"capture failed: {msg}")

        out = self.project.root / "calibration" / "_verify_tmp"
        self._shell.camera.captured.connect(on_captured)
        self._shell.camera.failed.connect(on_fail)
        self._shell.camera.capture(out, tag="_verify_calib")

    def _set_calib_indicator(self, quality, detail):
        colors = {"good": style.ACCENT, "poor": "#e8a020", "bad": style.WARN}
        c = colors.get(quality, style.DIM)
        self.calib_indicator.setStyleSheet(f"color: {c}; font-size: 16px;")
        self.calib_indicator.setToolTip(detail or "unknown")

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

        h.addWidget(button("browse", browse))
        return w, e

    def _profile(self, crop):
        p = CROP_PROFILES.get(crop)
        if p:
            self.seed_width.setValue(p["seed_width"])
            self.min_thick.setValue(p["min_thickness"])

    def accept(self):
        markers = self.ground.currentData() == "markers"
        sheet_file = self.sheet[1].text().strip()
        if markers and not (sheet_file and self.project.path(sheet_file).exists()):
            return QMessageBox.warning(self, "Project settings", "Corner markers need the stage sheet's layout "
                                       "(.json, written next to the PDF): choose it, or PRINT SHEET to make one.")
        c, s = self.project.config, self.project.config.settings
        c.name, c.crop = self.name.text().strip() or c.name, self.crop.currentText().strip().lower() or c.crop
        c.calib, c.stage = self.calib[1].text().strip(), self.stage[1].text().strip()
        c.markers_sheet, c.markers_bar_mm = (sheet_file if markers else ""), self.bar.value()
        s.level_height = self.level.isChecked()
        c.camera_args = self.camera.text().strip()
        s.seed_width, s.min_thickness, s.min_area = self.seed_width.value(), self.min_thick.value(), self.min_area.value()
        s.min_depth, s.max_depth, s.model_conf = self.min_depth.value(), self.max_depth.value(), self.conf.value()
        s.sam_model, s.sam_refine = self.sam_model.currentData(), self.sam_refine.isChecked()
        s.label_target = self.target.value()
        s.seed_labels = [x.strip() for x in self.seed_labels_edit.text().split(",") if x.strip()]
        if c.detector != "classic" and f"/{c.crop}/" not in c.detector:
            c.detector = "classic"  # models belong to a crop
        super().accept()


class TrainDialog(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell, self.project = shell, shell.project
        self.setWindowTitle("Train")
        self.resize(820, 560)
        v = QVBoxLayout(self)
        c = ml.dataset_counts(self.project)
        info = QLabel(f"{self.project.config.crop.upper()} · TRAIN {c['train'][0]} SAMPLES, {c['train'][1]} TILES, "
                      f"{c['train'][2]} SEEDS · VAL {c['val'][0]} SAMPLES, {c['val'][1]} TILES, {c['val'][2]} SEEDS")
        info.setObjectName("readout")
        v.addWidget(info)
        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        latest = ml.latest_model(self.project)
        self.epochs.setValue(150 if latest is None else 80)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["640", "960"])  # dataset tiles are 640 px
        self.base = QComboBox()
        # Once there is a model, improve it: continue from its weights on the whole dataset
        # (old + new samples), so it keeps what it learned and adds the new examples.
        for m in ml.list_models(self.project):
            self.base.addItem(f"Improve  {m['name']}  (continue training)", str(m["path"]))
        self.base.addItem(f"Start fresh  {ml.BASE_MODEL}", ml.BASE_MODEL)
        self.base.addItem("Start fresh  yolo11n-seg.pt (faster)", "yolo11n-seg.pt")
        self.base.currentIndexChanged.connect(
            lambda _i: self.epochs.setValue(150 if self.base.currentData() in (ml.BASE_MODEL, "yolo11n-seg.pt") else 80))
        if latest is not None and latest.get("input"):
            self.input_default = latest["input"]
        else:
            self.input_default = "rgbh"
        self.input = QComboBox()
        for k, val in ml.INPUTS.items():
            self.input.addItem(val.title(), k)
        self.input.setCurrentIndex(max(0, self.input.findData(self.input_default)))
        self.input.setToolTip("Colour + height also shows the model each pixel's height above the stage,\n"
                              "so it can tell seeds from shadows and stains of the same colour.")
        self.name = QLineEdit(time.strftime("run_%Y%m%d_%H%M%S"))
        for lab, w in (("INPUT", self.input), ("EPOCHS", self.epochs), ("IMAGE SIZE", self.imgsz),
                       ("START FROM", self.base), ("RUN NAME", self.name)):
            form.addRow(caption(lab), w)
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
        if self.shell.proc is not None:
            return
        kind = self.input.currentData()
        self.b_start.setEnabled(False)
        self.log.appendPlainText("Preparing dataset...")
        QApplication.processEvents()
        ml.sync_datasets(self.project)  # every dataset sample in both input formats
        argv = ml.train_command(self.project, self.name.text().strip(), self.epochs.value(),
                                int(self.imgsz.currentText()), self.base.currentData(), kind)
        self.log.appendPlainText(" ".join(argv[1:]) + "\n")

        run_name = self.name.text().strip()

        def ok():
            weights = self.project.models_dir / run_name / "weights" / "best.pt"
            if weights.exists():
                self.shell.use_model(weights)  # new detections start from the improved model
            self.log.appendPlainText(f"\nDone. {run_name} is now the detector: detect new samples with it, "
                                     "correct them, add them to the dataset and retrain to keep improving it.")
            self.b_start.setEnabled(True)
            self.shell.say(f"trained {run_name}; it is now the detector")

        def log(text):
            for line in text.replace("\r\n", "\n").split("\n"):
                line = line.split("\r")[-1]  # Ultralytics redraws progress lines with \r
                if line.strip():
                    self.log.appendPlainText(line)

        self.shell.run_process(argv[1:], "training", ok, log=log)

    def reject(self):
        if self.shell.proc is not None and not self.b_start.isEnabled():
            if QMessageBox.question(self, "Train", "Training is running. Stop it?") != QMessageBox.Yes:
                return
            self.shell.proc.kill()
        super().reject()
