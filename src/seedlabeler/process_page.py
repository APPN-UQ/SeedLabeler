"""PROCESS mode: batch detection of captured samples, flags worth a look, export."""
import statistics
import time
import traceback

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QMessageBox, QProgressBar,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import ml, processing
from .detection import flags, run_detection
from .widgets import button, caption, label

COLS = ["SAMPLE", "STATUS", "SEEDS", "DEPTH", "FLAGS", "CAPTURED"]
STATUS_TEXT = {"new": "to process", "detected": "detected", "reviewed": "detected", "dataset": "in dataset"}


class BatchWorker(QThread):
    progress = Signal(int, int, str, str)   # done, total, sample id, error ("" if ok)
    finished_all = Signal(int, float)       # processed, seconds

    def __init__(self, project, samples):
        super().__init__()
        self.project, self.samples = project, samples
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        t0, n = time.time(), 0
        for i, s in enumerate(self.samples, start=1):
            if self._stop:
                break
            err = ""
            try:
                run_detection(self.project, s)
                n += 1
            except Exception as e:  # noqa: BLE001 - keep going, report per sample
                traceback.print_exc()
                err = f"{type(e).__name__}: {e}"
            self.progress.emit(i, len(self.samples), s.id, err)
        self.finished_all.emit(n, time.time() - t0)


class ProcessPage(QWidget):
    openInLabel = Signal(str)

    def __init__(self, shell):
        super().__init__()
        self.setObjectName("page")
        self.shell = shell
        self.project = None
        self.worker = None
        self._t0 = None
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 12, 18, 8)
        v.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(caption("detector"))
        self.detector = QComboBox()
        self.detector.setMinimumWidth(260)
        self.detector.activated.connect(self._detector_changed)
        top.addWidget(self.detector)
        self.sam_state = label("", "dim")
        top.addWidget(self.sam_state)
        top.addStretch(1)
        self.b_new = button("process new", self.process_new, primary=True,
                            tip="Detect every sample that hasn't been processed yet")
        self.b_sel = button("process selected", self.process_selected, secondary=True,
                            tip="(Re)detect the selected samples; replaces their outlines")
        self.b_stop = button("stop", self.stop)
        self.b_export = button("export csv", self.export, secondary=True,
                               tip="Write every sample's traits into exports/")
        for b in (self.b_new, self.b_sel, self.b_stop, self.b_export):
            top.addWidget(b)
        v.addLayout(top)

        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setMaximum(1)
        self.progress.setValue(0)
        self.eta = label("", "readout")
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.eta)
        v.addLayout(prow)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        h = self.table.horizontalHeader()
        h.setStretchLastSection(True)
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for i, w in enumerate((260, 120, 80, 80, 220)):
            self.table.setColumnWidth(i, w)
        self.table.cellDoubleClicked.connect(self._open)
        v.addWidget(self.table, 1)
        self.summary = label("", "dim")
        v.addWidget(self.summary)
        self._update_enabled()

    # --- lifecycle --------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.refresh()

    def enter(self):
        self.refresh()

    def leave(self):
        pass

    def _refresh_detectors(self):
        self.detector.clear()
        self.detector.addItem("Classic", "classic")
        for m in ml.list_models(self.project):
            score = f"  mAP {m['mask_map50_95']:.2f}" if m["mask_map50_95"] is not None else ""
            rel = m["path"].relative_to(self.project.root).as_posix()
            self.detector.addItem(f"Model {m['name']}{score}", rel)
        self.detector.setCurrentIndex(max(0, self.detector.findData(self.project.config.detector)))
        st = self.project.config.settings
        self.sam_state.setText("+ SAM refine" if st.sam_refine else "(SAM refine off)")

    def _detector_changed(self, _i):
        self.project.config.detector = self.detector.currentData()
        self.project.save()

    def refresh(self):
        if not self.project:
            return
        self._refresh_detectors()
        samples = self.project.samples()
        counts = [len(s.annotations()["seeds"]) for s in samples if s.processed]
        med = statistics.median(counts) if counts else None
        self.table.setRowCount(len(samples))
        for r, s in enumerate(samples):
            st = s.status()
            meta = s.meta()
            n = len(s.annotations()["seeds"]) if s.processed else None
            fl = flags(s, med)
            cells = [s.id, STATUS_TEXT.get(st, st), "" if n is None else str(n),
                     f"{100 * meta['depth_coverage']:.0f}%" if "depth_coverage" in meta else "",
                     "  ".join(fl), time.strftime("%Y-%m-%d %H:%M", time.localtime(s.image.stat().st_mtime))]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setData(Qt.UserRole, s.id)
                self.table.setItem(r, c, it)
        waiting = sum(1 for s in samples if not s.processed)
        flagged = sum(1 for s in samples if s.processed and flags(s, med))
        self.summary.setText(f"{len(samples)} samples · {waiting} to process · {flagged} flagged · "
                             "double-click a sample to check or correct it in LABEL")
        self._update_enabled()

    def _update_enabled(self):
        running = self.worker is not None
        has = self.project is not None
        self.b_new.setEnabled(has and not running)
        self.b_sel.setEnabled(has and not running)
        self.b_export.setEnabled(has and not running)
        self.b_stop.setEnabled(running)
        self.detector.setEnabled(has and not running)

    # --- batch ------------------------------------------------------------------
    def process_new(self):
        todo = [s for s in self.project.samples() if not s.processed]
        if not todo:
            return self.shell.say("nothing to process")
        self._run(todo)

    def process_selected(self):
        ids = {self.table.item(i.row(), 0).data(Qt.UserRole) for i in self.table.selectionModel().selectedRows()}
        todo = [s for s in self.project.samples() if s.id in ids]
        if not todo:
            return self.shell.say("select samples first")
        edited = [s for s in todo if s.processed and any(x["status"] != "auto" for x in s.annotations()["seeds"])]
        if edited and QMessageBox.question(
                self, "Process", f"{len(edited)} selected sample(s) have corrected outlines, which re-detecting "
                                 "will replace. Continue?") != QMessageBox.Yes:
            return
        self._run(todo)

    def _run(self, samples):
        self.worker = BatchWorker(self.project, samples)
        self.worker.progress.connect(self._progress)
        self.worker.finished_all.connect(self._finished)
        self.progress.setMaximum(len(samples))
        self.progress.setValue(0)
        self._t0 = time.time()
        self.eta.setText(f"0 / {len(samples)}")
        self.shell.set_busy("batch", True)
        self.worker.start()
        self._update_enabled()

    def _progress(self, done, total, sid, err):
        self.progress.setValue(done)
        per = (time.time() - self._t0) / done
        left = per * (total - done)
        self.eta.setText(f"{done} / {total} · {per:.0f} s each · ~{left / 60:.0f} min left" if done < total
                         else f"{done} / {total}")
        if err:
            self.shell.say(f"{sid}: {err}")
        self.refresh()

    def _finished(self, n, seconds):
        self.worker = None
        self.shell.set_busy("batch", False)
        self.shell.say(f"processed {n} samples in {seconds / 60:.1f} min")
        self.refresh()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.shell.say("stopping after the current sample")

    # --- other ------------------------------------------------------------------
    def _open(self, row, _col):
        self.openInLabel.emit(self.table.item(row, 0).data(Qt.UserRole))

    def export(self):
        paths = processing.export_project(self.project)
        self.shell.say("exported " + ", ".join(p.name for p in paths) if paths else "nothing to export yet")
