"""Home (open / create a project) and the project hub (status, suggested workflow, modes)."""
import os
import re
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from . import ml, style
from .project import CAMERA_ARGS_HINT, CAMERA_SYSTEMS, CROP_PROFILES, Project
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
        self.system = QComboBox()
        for key, text in CAMERA_SYSTEMS.items():
            self.system.addItem(text, key)
        self.system.setCurrentIndex(max(0, self.system.findData(s.value("last_camera_system", "rx0"))))
        self.calib = self._path_row(s.value("last_calib", ""), filt="Stereo calibration (*.npz)")
        self.stage = self._path_row("", filt="Stage plane (*.npz)")
        self.camera = QLineEdit(s.value("last_camera_args", ""))
        for lab, w in (("NAME", self.name), ("LOCATION", self.location[0]), ("CROP", self.crop),
                       ("CAMERA SYSTEM", self.system), ("CALIBRATION", self.calib[0]),
                       ("STAGE (OPTIONAL)", self.stage[0]), ("CAMERA SETTINGS", self.camera)):
            form.addRow(caption(lab), w)
        v.addLayout(form)
        self.hint = label("", "dim", wrap=True)
        v.addWidget(self.hint)
        self.system.currentIndexChanged.connect(self._system_changed)
        self._system_changed()
        self.error = label("", "warn")
        v.addWidget(self.error)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("cancel", self.reject))
        row.addWidget(button("create", self.create, primary=True))
        v.addLayout(row)
        self.setMinimumWidth(640)

    def _system_changed(self, _i=None):
        oak = self.system.currentData() == "oak"
        self.calib[0].setEnabled(not oak)
        self.stage[0].setEnabled(not oak)
        self.camera.setPlaceholderText(CAMERA_ARGS_HINT["oak" if oak else "rx0"])
        self.hint.setText("OAK-D: no calibration needed (it carries its factory calibration). Keep the stage "
                          "40 cm or more from the camera. For a stage plane from corner markers, print the "
                          "stage sheet in SETTINGS after creating the project." if oak else "")

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
        system = self.system.currentData()
        calib = self.calib[1].text().strip() if system == "rx0" else ""
        stage = self.stage[1].text().strip() if system == "rx0" else ""
        if calib and not Path(calib).exists():
            return self.error.setText("The calibration file doesn't exist.")
        crop = self.crop.currentText().strip().lower() or "custom"
        p = Project.create(folder, name, crop=crop, calib=calib, stage=stage, camera=system)
        p.config.camera_args = self.camera.text().strip()
        p.save()
        s = QSettings(*SETTINGS)
        s.setValue("last_location", str(folder.parent))
        if calib:
            s.setValue("last_calib", calib)
        s.setValue("last_camera_args", p.config.camera_args)
        s.setValue("last_camera_system", system)
        self.created = folder
        self.accept()


# --------------------------------------------------------------------------- hub
STEPS = [
    ("CAPTURE", "Scan a barcode or type a sample id, capture, next. No detection here: just collect good images. "
                "Start with a first batch so there is something to label."),
    ("LABEL", "Detect seeds in the captured samples, correct the outlines, add each to the dataset. "
              "Once there is a model, it does the first draft and you correct it."),
    ("TRAIN", "Train the crop's model on the labelled samples, then retrain as you correct more: each round "
              "starts from the last model and learns from the whole dataset."),
    ("PROCESS", "Detect every captured sample in one batch, check the flagged ones, export traits."),
]
MODES = {
    "capture": ("CAPTURE", "Barcode-driven collection with live view and instant quality checks. "
                           "Built for throughput."),
    "label": ("LABEL & TRAIN", "Detect with SAM, correct outlines, add to the dataset, train models. "
                               "Slower and deliberate."),
    "process": ("PROCESS & EXPORT", "Batch detection of everything captured, flags worth a look, "
                                    "traits CSVs."),
}


def workflow_state(project):
    """Per-step (state, text) with state in done/next/todo, the recommended mode, and counts.

    The flow is capture -> label -> train -> process. The same target (settings
    .label_target) says how many samples to capture before labelling is worth
    starting and how many to label before training. Both are soft: the counts
    guide, nothing is locked.
    """
    samples = project.samples()
    captured = len(samples)
    waiting = sum(1 for s in samples if not s.processed)
    ds_samples, ds_seeds = ml.dataset_summary(project)
    target = project.config.settings.label_target
    models = ml.list_models(project)
    model = models[0] if models else None
    new = ml.samples_since_model(project, model)
    # retrain once a meaningful share of the dataset is new
    retrain_due = model is not None and new >= max(5, 0.25 * max(ds_samples, 1))

    has_model = model is not None
    done_capture = captured >= target or has_model
    done_label = ds_samples >= target or has_model
    done_train = has_model and not retrain_due
    if not has_model:
        train_text = "no model yet" + (f" · ready to train on {ds_samples} samples" if done_label else "")
    else:
        score = f" · mask mAP50 {model['mask_map50']:.2f}" if model.get("mask_map50") is not None else ""
        train_text = f"{model['name']}{score} · {new} sample(s) added since" + (" · retrain" if retrain_due else "")
    texts = [
        f"{captured} captured" if has_model else f"{captured} / {target} samples captured",
        (f"{ds_samples} samples · {ds_seeds} seeds in the dataset" if has_model
         else f"{ds_samples} / {target} samples labelled · {ds_seeds} seeds"),
        train_text,
        f"{waiting} waiting to be processed" if waiting else "everything processed",
    ]
    if not has_model:
        nxt = 0 if not done_capture else (1 if not done_label else 2)
    elif retrain_due:
        nxt = 2
    elif waiting:
        nxt = 3
    else:
        nxt = 0
    states = ["done" if d else "todo" for d in (done_capture, done_label, done_train, False)]
    states[nxt] = "next"
    mode = {0: "capture", 1: "label", 2: "label", 3: "process"}[nxt]

    # the top-bar readiness pill: one line on what to do next; "ok" once the current step's door is open
    if not has_model:
        if not done_capture:
            pill = (f"capture {captured} / {target}", "warn")
        elif not done_label:
            pill = (f"ready to label · {ds_samples} / {target} labelled", "ok")
        else:
            pill = (f"ready to train · {ds_samples} labelled", "ok")
    elif retrain_due:
        pill = (f"{model['name']} · {new} new samples · retrain", "warn")
    else:
        pill = (f"{model['name']} · {waiting} to process" if waiting else f"{model['name']} · ready", "ok")
    return list(zip(states, texts)), mode, {
        "samples": captured, "waiting": waiting, "ds_samples": ds_samples, "ds_seeds": ds_seeds,
        "models": len(models), "has_model": has_model, "new_since_model": new, "target": target,
        "label_open": done_capture, "process_open": has_model, "pill": pill}


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
        self.b_train.setVisible(steps[2][0] == "next")
        self.b_train.setText("RETRAIN NOW" if stats["has_model"] else "TRAIN NOW")
        for k, b in self.tiles.items():
            b.setProperty("recommended", "true" if k == mode else "false")
            b.style().unpolish(b)
            b.style().polish(b)
