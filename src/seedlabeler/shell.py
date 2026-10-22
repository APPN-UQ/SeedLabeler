"""The main window: top bar with modes, shared services (tasks, camera, training), page switching."""
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from . import ml, style
from .camera import CameraService
from .capture_page import CapturePage
from .home import HomePage, HubPage, remember_project
from .label_page import LabelPage, SettingsDialog, TrainDialog
from .process_page import ProcessPage
from .project import Project, Sample
from .widgets import Task, button, hline, label

MODES = [("hub", "HUB"), ("label", "LABEL"), ("capture", "CAPTURE"), ("process", "PROCESS")]


class Shell(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeedLabeler")
        self.resize(1600, 980)
        self.project = None
        self.tasks = set()
        self.proc = None
        self._busy = set()
        self._pending = {}
        self._camera_args_applied = None
        self.camera = CameraService()
        self.camera.captured.connect(self._captured)
        self.camera.failed.connect(self._camera_failed)
        self.camera.state.connect(self._camera_state)
        self.camera.start()

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(52)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(20, 0, 20, 0)
        self.wordmark = button("seedlabeler", self.go_home, tip="Home: open or create a project")
        self.wordmark.setStyleSheet(f"color:{style.TEXT}; letter-spacing:5px; font-weight:600; font-size:12px;")
        bl.addWidget(self.wordmark)
        self.project_label = label("", "dim")
        bl.addWidget(self.project_label)
        bl.addSpacing(24)
        self.tabs = {}
        for key, text in MODES:
            b = button(text, lambda _=False, k=key: self.go(k), checkable=True, name="tab")
            bl.addWidget(b)
            self.tabs[key] = b
        bl.addStretch(1)
        self.cam_label = label("cameras · not connected", "dim")
        bl.addWidget(self.cam_label)
        bl.addSpacing(12)
        self.b_settings = button("settings", self.edit_settings)
        self.b_close = button("close project", self.go_home)
        bl.addWidget(self.b_settings)
        bl.addWidget(self.b_close)
        outer.addWidget(bar)
        self.busy_line = style.BusyLine()
        outer.addWidget(self.busy_line)

        self.stack = QStackedWidget()
        self.home = HomePage()
        self.hub = HubPage()
        self.label = LabelPage(self)
        self.capture = CapturePage(self)
        self.process = ProcessPage(self)
        self.pages = {"home": self.home, "hub": self.hub, "label": self.label, "capture": self.capture,
                      "process": self.process}
        for p in self.pages.values():
            self.stack.addWidget(p)
        outer.addWidget(self.stack, 1)

        outer.addWidget(hline())
        foot = QHBoxLayout()
        foot.setContentsMargins(20, 6, 20, 6)
        self.status = label("", "dim")
        foot.addWidget(self.status)
        fw = QWidget()
        fw.setLayout(foot)
        outer.addWidget(fw)
        self.setCentralWidget(root)

        self.home.openProject.connect(self.open_project)
        self.hub.goMode.connect(self.go)
        self.hub.editSettings.connect(self.edit_settings)
        self.hub.train.connect(self.train)
        self.hub.loadModel.connect(self.load_model)
        self.process.openInLabel.connect(self.open_in_label)
        self.current = None
        self.go_home()

    # ------------------------------------------------------------ navigation
    def has_model(self):
        return self.project is not None and ml.latest_model(self.project) is not None

    def go(self, key):
        if key != "home" and self.project is None:
            return
        if key == "capture" and not self.has_model():
            self.tabs["capture"].setChecked(False)
            return self.say("Rapid capture unlocks once this crop has a model: train one in LABEL or LOAD MODEL "
                            "on the hub.")
        if self.current and hasattr(self.pages[self.current], "leave"):
            self.pages[self.current].leave()
        self.current = key
        self.stack.setCurrentWidget(self.pages[key])
        if hasattr(self.pages[key], "enter"):
            self.pages[key].enter()
        for k, b in self.tabs.items():
            b.setChecked(k == key)
            b.setVisible(self.project is not None)
        self.tabs["capture"].setEnabled(self.has_model())
        self.tabs["capture"].setToolTip("" if self.has_model() else "Needs a trained model for this crop")
        self.b_settings.setVisible(self.project is not None)
        self.b_close.setVisible(self.project is not None)

    def go_home(self):
        if self.project is not None and (self.tasks or self._pending):
            return self.say("wait for the running work to finish")
        self.project = None
        self.project_label.setText("")
        self.home.refresh()
        self.go("home")

    def open_project(self, path):
        try:
            self.project = Project.open(path)
        except Exception as e:  # noqa: BLE001
            return self.error(f"Not a project: {e}")
        remember_project(self.project.root)
        self.camera.set_rig_file(self.project.rig_file)
        self._camera_args_applied = None
        c = self.project.config
        self.project_label.setText(f"·   {c.name}   ·   {c.crop}")
        for k in ("hub", "label", "capture", "process"):
            self.pages[k].set_project(self.project)
        self.go("hub")

    def open_in_label(self, sample_id):
        self.go("label")
        self.label.open_sample(sample_id)

    def edit_settings(self):
        if not self.project:
            return
        if SettingsDialog(self.project, self).exec():
            self.project.adopt_files()  # copy a newly chosen calibration/stage into the project
            self.project.save()
            self.camera.set_rig_file(self.project.rig_file)
            self._camera_args_applied = None
            c = self.project.config
            self.project_label.setText(f"·   {c.name}   ·   {c.crop}")
            for k in ("hub", "label", "capture", "process"):
                self.pages[k].set_project(self.project)
            self.go(self.current)

    def train(self):
        TrainDialog(self).exec()
        self.models_changed()

    def models_changed(self):
        """After training or loading a model: refresh everything that lists models or gates on them."""
        self.label._refresh_models()  # keeps the sample being edited
        self.label._update_model_readout()
        self.process.refresh()
        self.hub.refresh()
        self.go(self.current)

    def use_model(self, weights):
        """Make a model the project's detector, so new detections (and labels) start from it."""
        self.project.config.detector = Path(weights).relative_to(self.project.root).as_posix()
        self.project.save()

    def load_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load a trained model (best.pt from a SeedLabeler run folder)",
                                           str(self.project.root), "Model (*.pt)")
        if not f:
            return
        weights, assumed = ml.import_model(self.project, f)
        self.use_model(weights)
        self.models_changed()
        self.say(f"loaded model {weights.parent.parent.name} and made it the detector"
                 + (" (no settings found with it: assumed colour input and this project's tiling)" if assumed else ""))

    # ------------------------------------------------------------ shared services
    def say(self, msg):
        self.status.setText(msg)

    def error(self, msg):
        self.say(msg)
        QMessageBox.warning(self, "SeedLabeler", msg)

    def set_busy(self, key, on):
        (self._busy.add if on else self._busy.discard)(key)
        self.busy_line.set_busy(bool(self._busy or self.tasks or self.proc is not None))

    def run_task(self, fn, *args, done=None, msg=None, **kw):
        t = Task(fn, *args, **kw)
        self.tasks.add(t)
        self.set_busy("tasks", True)
        if msg:
            self.say(msg)

        def finish():
            self.tasks.discard(t)
            self.set_busy("tasks", bool(self.tasks))

        t.done.connect(lambda r: (finish(), done(r) if done else None))
        t.failed.connect(lambda e: (finish(), self.error(e)))
        t.start()
        return t

    def run_process(self, args, msg, on_ok, log=None):
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
            self.set_busy("proc", False)
            if code == 0:
                on_ok()
            else:
                self.error(f"{msg.capitalize()} failed:\n" + "".join(out)[-1500:])

        self.proc.readyRead.connect(read)
        self.proc.finished.connect(finished)
        self.set_busy("proc", True)
        self.say(msg)
        self.proc.start(sys.executable, args)
        return self.proc

    # ------------------------------------------------------------ cameras
    def capture_sample(self, sample_id, done, failed=None):
        """Fire both cameras into a new sample folder; done(Sample) when the images are in."""
        args = self.project.config.camera_args.strip()
        if args and self._camera_args_applied != args:
            self.camera.apply_settings(args)  # queued before the capture
            self._camera_args_applied = args
        sid = self.project.new_sample_id(sample_id)
        raw = self.project.root / "samples" / sid / "raw"
        self._pending[sid] = (done, failed, raw)
        self.set_busy("capture", True)
        self.say(f"capturing {sid}")
        self.camera.capture(raw, tag=sid)

    def _captured(self, info):
        sid = info["tag"]
        entry = self._pending.pop(sid, None)
        self.set_busy("capture", bool(self._pending))
        if entry is None:
            return
        done, _failed, raw = entry
        self.say(f"captured {sid} in {info['seconds']:.1f} s")
        done(Sample(raw.parent))

    def _camera_failed(self, msg, tag):
        entry = self._pending.pop(tag, None) if tag is not None else None
        self.set_busy("capture", bool(self._pending))
        if entry is not None:
            _done, failed, raw = entry
            shutil.rmtree(raw.parent, ignore_errors=True)
            if failed:
                failed(msg)
        self.error(msg)

    def _camera_state(self, state):
        self.cam_label.setText({"connected": "cameras · connected", "capturing": "cameras · capturing",
                                "disconnected": "cameras · not connected"}.get(state, state))

    def closeEvent(self, e):
        if self.current and hasattr(self.pages[self.current], "leave"):
            self.pages[self.current].leave()
        if self.proc is not None:
            self.proc.kill()
        self.camera.stop()
        super().closeEvent(e)
