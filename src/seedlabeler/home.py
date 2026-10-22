"""Home (open / create a project) and the project hub (status, suggested workflow, modes)."""
import os
import re
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from . import ml, style
from .project import CROP_PROFILES, Project
from .widgets import button, caption, label

SETTINGS = ("APPF", os.environ.get("SEEDLABELER_SETTINGS", "SeedLabeler"))  # tests use their own


def recent_projects():
    paths = QSettings(*SETTINGS).value("recent_projects", []) or []
    if isinstance(paths, str):
        paths = [paths]
    return [p for p in paths if (Path(p) / "project.json").exists()]


def remember_project(path):
    s = QSettings(*SETTINGS)
    paths = [str(path)] + [p for p in recent_projects() if Path(p) != Path(path)]
    s.setValue("recent_projects", paths[:10])


# --------------------------------------------------------------------------- home
class HomePage(QWidget):
    openProject = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        outer = QHBoxLayout(self)
        outer.addStretch(1)
        col = QVBoxLayout()
        col.addStretch(1)
        col.addWidget(label("SEEDLABELER", "hero"))
        col.addWidget(label("Stereo seed phenotyping: size, shape, colour and uniformity.", "dim"))
        col.addSpacing(28)
        row = QHBoxLayout()
        row.addWidget(button("new project", self.new_project, primary=True))
        row.addWidget(button("open project", self.open_dialog, secondary=True))
        row.addStretch(1)
        col.addLayout(row)
        col.addSpacing(28)
        col.addWidget(caption("recent projects"))
        self.recent = QListWidget()
        self.recent.setMinimumHeight(260)
        self.recent.itemActivated.connect(lambda it: self.openProject.emit(it.data(Qt.UserRole)))
        self.recent.itemClicked.connect(lambda it: self.openProject.emit(it.data(Qt.UserRole)))
        col.addWidget(self.recent)
        col.addStretch(2)
        box = QWidget()
        box.setLayout(col)
        box.setFixedWidth(620)
        outer.addWidget(box)
        outer.addStretch(1)
        self.refresh()

    def refresh(self):
        self.recent.clear()
        for p in recent_projects():
            try:
                cfg = Project.open(p).config
                text = f"{cfg.name}    ·    {cfg.crop}\n{p}"
            except Exception:  # noqa: BLE001
                continue
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, p)
            self.recent.addItem(it)
        if not self.recent.count():
            it = QListWidgetItem("No projects yet: create one to start.")
            it.setFlags(Qt.NoItemFlags)
            self.recent.addItem(it)

    def open_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open project folder")
        if d:
            self.openProject.emit(d)

    def new_project(self):
        dlg = NewProjectDialog(self)
        if dlg.exec() and dlg.created:
            self.openProject.emit(str(dlg.created))


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.created = None
        s = QSettings(*SETTINGS)
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.addWidget(label("New project", "h2"))
        v.addWidget(label("One project per experiment or batch. Everything can be changed later in settings.",
                          "dim", wrap=True))
        v.addSpacing(10)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. chickpea trial 2026")
        self.location = self._path_row(s.value("last_location", str(Path.home() / "Documents" / "SeedLabeler")),
                                       folder=True)
        self.crop = QComboBox()
        self.crop.setEditable(True)
        self.crop.addItems(list(CROP_PROFILES))
        self.calib = self._path_row(s.value("last_calib", ""), filt="Stereo calibration (*.npz)")
        self.stage = self._path_row("", filt="Stage plane (*.npz)")
        self.camera = QLineEdit(s.value("last_camera_args", ""))
        self.camera.setPlaceholderText("--mode m --shutter 1/125 --iso 400 (optional)")
        for lab, w in (("NAME", self.name), ("LOCATION", self.location[0]), ("CROP", self.crop),
                       ("CALIBRATION", self.calib[0]), ("STAGE (OPTIONAL)", self.stage[0]),
                       ("CAMERA SETTINGS", self.camera)):
            form.addRow(caption(lab), w)
        v.addLayout(form)
        self.error = label("", "warn")
        v.addWidget(self.error)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("cancel", self.reject))
        row.addWidget(button("create", self.create, primary=True))
        v.addLayout(row)
        self.setMinimumWidth(640)

    def _path_row(self, value, folder=False, filt=""):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(value)
        h.addWidget(e, 1)

        def browse():
            if folder:
                f = QFileDialog.getExistingDirectory(self, "Location", e.text())
            else:
                f, _ = QFileDialog.getOpenFileName(self, filt, e.text(), filt)
            if f:
                e.setText(f)

        h.addWidget(button("browse", browse))
        return w, e

    def create(self):
        name = self.name.text().strip()
        if not name:
            return self.error.setText("Give the project a name.")
        folder = Path(self.location[1].text().strip()) / re.sub(r"[^A-Za-z0-9 _-]+", "_", name)
        if (folder / "project.json").exists():
            self.created = folder  # already a project: just open it
            return self.accept()
        calib = self.calib[1].text().strip()
        if calib and not Path(calib).exists():
            return self.error.setText("The calibration file doesn't exist.")
        crop = self.crop.currentText().strip().lower() or "custom"
        p = Project.create(folder, name, crop=crop, calib=calib, stage=self.stage[1].text().strip())
        p.config.camera_args = self.camera.text().strip()
        p.save()
        s = QSettings(*SETTINGS)
        s.setValue("last_location", str(folder.parent))
        if calib:
            s.setValue("last_calib", calib)
        s.setValue("last_camera_args", p.config.camera_args)
        self.created = folder
        self.accept()


# --------------------------------------------------------------------------- hub
STEPS = [
    ("LABEL", "Capture and correct samples. Once there is a model, it does the first draft and you correct it."),
    ("TRAIN", "Train the crop's model, then retrain as you add corrected samples: each round starts from the "
              "last model and learns from the whole dataset."),
    ("CAPTURE", "Collect the rest quickly: scan, capture, next. Unlocks once the crop has a model."),
    ("PROCESS", "Detect every captured sample in one batch, check the flagged ones, export traits."),
]
MODES = {
    "label": ("LABEL & TRAIN", "Capture, detect with SAM, correct outlines, add to the dataset, train models. "
                               "Slower and deliberate."),
    "capture": ("RAPID CAPTURE", "Barcode-driven collection with live view and instant quality checks. "
                                 "Built for throughput. Needs a trained model."),
    "process": ("PROCESS & EXPORT", "Batch detection of everything captured, flags worth a look, "
                                    "traits CSVs."),
}


def workflow_state(project):
    """Per-step (state, text) with state in done/next/todo, and the recommended mode."""
    samples = project.samples()
    waiting = sum(1 for s in samples if not s.processed)
    ds_samples, ds_seeds = ml.dataset_summary(project)
    target = project.config.settings.label_target
    models = ml.list_models(project)
    model = models[0] if models else None
    new = ml.samples_since_model(project, model)
    # retrain once a meaningful share of the dataset is new
    retrain_due = model is not None and new >= max(5, 0.25 * max(ds_samples, 1))

    done1 = ds_samples >= target or model is not None
    done2 = model is not None and not retrain_due
    if model is None:
        train_text = "no model yet" + (f" · ready to train on {ds_samples} samples" if done1 else "")
    else:
        score = f" · mask mAP50 {model['mask_map50']:.2f}" if model.get("mask_map50") is not None else ""
        train_text = f"{model['name']}{score} · {new} sample(s) added since" + (" · retrain" if retrain_due else "")
    texts = [
        f"{ds_samples} / {target} samples · {ds_seeds} seeds in the dataset" if model is None
        else f"{ds_samples} samples · {ds_seeds} seeds in the dataset",
        train_text,
        f"{len(samples)} captured" if model is not None else "locked until there is a model",
        f"{waiting} waiting to be processed" if waiting else "everything processed",
    ]
    if model is None:
        nxt = 1 if done1 else 0
    elif retrain_due:
        nxt = 1
    elif waiting:
        nxt = 3
    else:
        nxt = 2
    states = ["done" if (i == 0 and done1) or (i == 1 and done2) else "todo" for i in range(4)]
    states[nxt] = "next"
    mode = {0: "label", 1: "label", 2: "capture", 3: "process"}[nxt]
    return list(zip(states, texts)), mode, {"samples": len(samples), "waiting": waiting, "ds_samples": ds_samples,
                                            "ds_seeds": ds_seeds, "models": len(models), "has_model": model is not None,
                                            "new_since_model": new}


class StepWidget(QFrame):
    def __init__(self, n, title, desc):
        super().__init__()
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        top = QHBoxLayout()
        self.badge = QLabel(str(n))
        self.badge.setFixedSize(26, 26)
        self.badge.setAlignment(Qt.AlignCenter)
        top.addWidget(self.badge)
        self.title = label(title)
        top.addWidget(self.title)
        top.addStretch(1)
        v.addLayout(top)
        v.addWidget(label(desc, "dim", wrap=True))
        self.status = label("", "readout", wrap=True)
        v.addWidget(self.status)
        v.addStretch(1)
        self.n = n

    def set_state(self, state, text):
        self.status.setText(text)
        if state == "done":
            self.badge.setText("✓")
            self.badge.setStyleSheet(f"background:{style.ACCENT_SOFT}; color:{style.ACCENT}; border-radius:13px;")
            self.title.setStyleSheet(f"color:{style.TEXT}; letter-spacing:2px;")
            self.setStyleSheet("")
        elif state == "next":
            self.badge.setText(str(self.n))
            self.badge.setStyleSheet(f"background:{style.ACCENT}; color:white; border-radius:13px; font-weight:600;")
            self.title.setStyleSheet(f"color:{style.ACCENT}; font-weight:600; letter-spacing:2px;")
            self.setStyleSheet(f"QFrame#card {{ border: 2px solid {style.ACCENT}; }}")
        else:
            self.badge.setText(str(self.n))
            self.badge.setStyleSheet(f"background:{style.LINE}; color:{style.DIM}; border-radius:13px;")
            self.title.setStyleSheet(f"color:{style.DIM}; letter-spacing:2px;")
            self.setStyleSheet("")


class HubPage(QWidget):
    goMode = Signal(str)
    editSettings = Signal()
    train = Signal()
    loadModel = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        self.project = None
        v = QVBoxLayout(self)
        v.setContentsMargins(36, 26, 36, 20)
        v.setSpacing(14)
        head = QHBoxLayout()
        col = QVBoxLayout()
        self.title = label("", "h2")
        self.sub = label("", "dim")
        col.addWidget(self.title)
        col.addWidget(self.sub)
        head.addLayout(col, 1)
        head.addWidget(button("load model", self.loadModel.emit, secondary=True,
                              tip="Use a model trained elsewhere (e.g. another project with the same crop)"))
        head.addWidget(button("settings", self.editSettings.emit, secondary=True))
        v.addLayout(head)

        stats = QHBoxLayout()
        self.stat_labels = {}
        for key, cap in (("samples", "captured"), ("waiting", "to process"), ("ds_samples", "labelled samples"),
                         ("ds_seeds", "labelled seeds"), ("models", "models")):
            box = QVBoxLayout()
            box.addWidget(caption(cap))
            lab = label("—", "big")
            box.addWidget(lab)
            self.stat_labels[key] = lab
            stats.addLayout(box)
            stats.addSpacing(28)
        stats.addStretch(1)
        v.addLayout(stats)

        v.addWidget(caption("suggested workflow"))
        steps = QHBoxLayout()
        self.steps = [StepWidget(i + 1, t, d) for i, (t, d) in enumerate(STEPS)]
        for i, s in enumerate(self.steps):
            steps.addWidget(s, 1)
            if i < 3:
                steps.addWidget(label("→", "dim"))
        v.addLayout(steps)
        self.b_train = button("train now", self.train.emit, primary=True)
        trow = QHBoxLayout()
        trow.addStretch(1)
        trow.addWidget(self.b_train)
        v.addLayout(trow)

        v.addWidget(caption("modes"))
        tiles = QGridLayout()
        tiles.setHorizontalSpacing(14)
        self.tiles = {}
        for i, (key, (t, d)) in enumerate(MODES.items()):
            b = QPushButton(f"{t}\n\n{d}".replace("&", "&&"))  # a single & would be a shortcut marker
            b.setObjectName("tile")
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(130)
            b.clicked.connect(lambda _=False, k=key: self.goMode.emit(k))
            tiles.addWidget(b, 0, i)
            self.tiles[key] = b
        v.addLayout(tiles)
        v.addStretch(1)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def enter(self):
        self.refresh()

    def leave(self):
        pass

    def refresh(self):
        if not self.project:
            return
        c = self.project.config
        self.title.setText(c.name)
        self.sub.setText(f"{c.crop}  ·  {self.project.root}")
        steps, mode, stats = workflow_state(self.project)
        for w, (state, text) in zip(self.steps, steps):
            w.set_state(state, text)
        for k, lab in self.stat_labels.items():
            lab.setText(str(stats[k]))
        self.b_train.setVisible(steps[1][0] == "next")
        self.b_train.setText("RETRAIN NOW" if stats["has_model"] else "TRAIN NOW")
        self.tiles["capture"].setEnabled(stats["has_model"])
        self.tiles["capture"].setToolTip("" if stats["has_model"] else
                                         "Train a model in LABEL (or LOAD MODEL) to unlock rapid capture.")
        for k, b in self.tiles.items():
            b.setProperty("recommended", "true" if k == mode else "false")
            b.style().unpolish(b)
            b.style().polish(b)
