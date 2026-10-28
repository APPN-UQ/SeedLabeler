"""Auto-update: check GitHub Releases for a newer version, download and swap."""
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QVBoxLayout

from . import __version__, style
from .widgets import button, caption

REPO = "APPN-UQ/SeedLabeler"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases/latest"


def _parse_version(tag):
    return tuple(int(x) for x in tag.lstrip("v").split(".")[:3])


def _is_frozen():
    return getattr(sys, "frozen", False)


class _CheckThread(QThread):
    result = Signal(object)

    def run(self):
        try:
            req = Request(RELEASES_URL, headers={"Accept": "application/vnd.github+json"})
            with urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            tag = data.get("tag_name", "")
            remote = _parse_version(tag)
            local = _parse_version(__version__)
            if remote > local:
                asset = None
                for a in data.get("assets", []):
                    if a["name"].endswith(".zip") and "win" in a["name"].lower():
                        asset = a
                        break
                self.result.emit({"tag": tag, "notes": data.get("body", ""), "asset": asset})
            else:
                self.result.emit(None)
        except Exception:
            self.result.emit(None)


class _DownloadThread(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, url, dest):
        super().__init__()
        self._url, self._dest = url, dest

    def run(self):
        try:
            req = Request(self._url, headers={"Accept": "application/octet-stream"})
            with urlopen(req, timeout=120) as r:
                total = int(r.headers.get("Content-Length", 0))
                buf = bytearray()
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    self.progress.emit(len(buf), total)
            Path(self._dest).write_bytes(buf)
            self.finished.emit(self._dest)
        except Exception as e:
            self.failed.emit(str(e))


class UpdateDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update available")
        self.setMinimumWidth(480)
        self.info = info
        self._dl_thread = None
        v = QVBoxLayout(self)
        v.setSpacing(12)
        v.addWidget(caption(f"SeedLabeler {info['tag']} is available  (you have {__version__})"))
        if info["notes"]:
            notes = QLabel(info["notes"][:500])
            notes.setWordWrap(True)
            notes.setStyleSheet(f"color: {style.DIM}; padding: 8px;")
            v.addWidget(notes)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {style.DIM};")
        v.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_skip = button("skip", self.reject)
        self.b_update = button("download && install", self._download, primary=True)
        if not info["asset"]:
            self.b_update.setEnabled(False)
            self.b_update.setToolTip("No Windows build attached to this release")
        if not _is_frozen():
            self.b_update.setEnabled(False)
            self.b_update.setToolTip("Auto-update only works in the packaged app — pull from git instead")
        row.addWidget(self.b_skip)
        row.addWidget(self.b_update)
        v.addLayout(row)

    def _download(self):
        asset = self.info["asset"]
        self.b_update.setEnabled(False)
        self.b_skip.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.status.setText("downloading ...")
        dest = os.path.join(tempfile.gettempdir(), asset["name"])
        self._dl_thread = _DownloadThread(asset["browser_download_url"], dest)
        self._dl_thread.progress.connect(self._on_progress)
        self._dl_thread.finished.connect(self._on_downloaded)
        self._dl_thread.failed.connect(self._on_failed)
        self._dl_thread.start()

    def _on_progress(self, got, total):
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(got)
            self.status.setText(f"downloading ... {got // 1024} / {total // 1024} KB")

    def _on_downloaded(self, zip_path):
        self.status.setText("extracting ...")
        try:
            app_dir = Path(sys.executable).parent
            extract_dir = Path(tempfile.mkdtemp(prefix="seedlabeler_update_"))
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            # find the actual app folder inside the zip (may be nested one level)
            contents = list(extract_dir.iterdir())
            src = contents[0] if len(contents) == 1 and contents[0].is_dir() else extract_dir
            self._write_swap_script(src, app_dir)
            self.status.setText("update ready — restarting ...")
            self.status.setStyleSheet(f"color: {style.ACCENT}; font-weight: 600;")
            QMessageBox.information(self, "Update ready",
                                   "SeedLabeler will close and restart with the new version.")
            self._launch_swap_and_exit()
        except Exception as e:
            self._on_failed(str(e))

    def _on_failed(self, msg):
        self.status.setText(f"update failed: {msg}")
        self.status.setStyleSheet(f"color: {style.WARN};")
        self.b_skip.setEnabled(True)

    def _write_swap_script(self, src, dest):
        self._script = Path(tempfile.gettempdir()) / "seedlabeler_update.bat"
        # wait for the app to exit, xcopy new files over old, relaunch, clean up
        self._script.write_text(
            f'@echo off\r\n'
            f'echo Updating SeedLabeler ...\r\n'
            f'timeout /t 2 /nobreak >nul\r\n'
            f'xcopy /s /e /y "{src}\\*" "{dest}\\" >nul\r\n'
            f'start "" "{dest}\\SeedLabeler.exe"\r\n'
            f'rmdir /s /q "{src.parent}"\r\n'
            f'del "%~f0"\r\n',
            encoding="utf-8",
        )

    def _launch_swap_and_exit(self):
        import subprocess
        subprocess.Popen(
            ["cmd", "/c", str(self._script)],
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        )
        sys.exit(0)


def check_for_update(parent):
    """Call once at startup. Runs the check in a background thread, shows the dialog only if an update exists."""
    checker = _CheckThread(parent)
    checker.result.connect(lambda info: _show_if_available(info, parent))
    checker.start()
    parent._update_checker = checker  # prevent GC


def _show_if_available(info, parent):
    if info is not None:
        UpdateDialog(info, parent).exec()
