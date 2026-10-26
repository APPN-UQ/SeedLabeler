"""The camera service: both RX0s stay connected for the whole session.

One thread owns both cameras (COM objects must stay on the thread that
created them) and runs live view, stereo capture and settings changes from a
command queue. Not reconnecting for every shot is what makes rapid capture
take a couple of seconds per sample. Commands queue up while a capture or a
settings change runs; live view pauses meanwhile and then resumes.
"""
import json
import queue
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal


class CameraService(QThread):
    frame = Signal(object)          # live-view BGR frame (left camera)
    captured = Signal(object)       # {"left": path, "right": path, "tag": any, "seconds": float}
    failed = Signal(str, object)    # message, tag of the failed request (or None)
    state = Signal(str)             # "disconnected" | "connected" | "capturing"

    def __init__(self, rig_file=None):
        super().__init__()
        self.rig_file = rig_file
        self._q = queue.Queue()
        self._live = False
        self._stop = False
        self._markers = None
        self._left_id = None
        self.cams = None  # {"left": RX0, "right": RX0}

    # --- public, thread-safe ------------------------------------------------
    def set_rig_file(self, path):
        self._q.put(("rig", path))

    def set_markers(self, layout):
        self._markers = layout

    def set_live(self, on):
        self._live = bool(on)

    def capture(self, out_dir, tag=None):
        """Fire both cameras and save <out_dir>/left.jpg and right.jpg."""
        self._q.put(("capture", (Path(out_dir), tag)))

    def apply_settings(self, args, tag=None):
        """Camera settings as rx0-settings-style arguments, e.g. '--mode m --shutter 1/125 --iso 400'."""
        self._q.put(("settings", (args, tag)))

    def focus_lock(self, tag=None):
        """Autofocus on what's in front of both cameras, then lock to manual."""
        self._q.put(("focus_lock", (None, tag)))

    def stop(self):
        self._stop = True
        self.wait(8000)

    # --- thread ---------------------------------------------------------------
    def _connect(self):
        from sonyrx0.camera.sony import RX0
        from sonyrx0.camera.wpd_mtp import find_devices

        paths = find_devices()
        if len(paths) < 2:
            raise RuntimeError(f"Found {len(paths)} camera(s); need both cameras on, connected and in PC Remote mode.")
        left_serial = None
        if self.rig_file and Path(self.rig_file).exists():
            left_serial = json.loads(Path(self.rig_file).read_text()).get("left")
        paths = sorted(paths, key=lambda p: 0 if left_serial and left_serial in p else 1)
        self._left_id = paths[0]
        cams = {"left": RX0(paths[0]), "right": RX0(paths[1])}
        for c in cams.values():
            c.discard_pending()
        self.cams = cams
        self.state.emit("connected")

    def _disconnect(self):
        if self.cams:
            for c in self.cams.values():
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
        self.cams = None
        self.state.emit("disconnected")

    def _ensure(self):
        if self.cams is None:
            self._connect()

    def _capture(self, out_dir, tag):
        t0 = time.time()
        self._ensure()
        self.state.emit("capturing")
        left, right = self.cams["left"], self.cams["right"]
        # half-press both, let focus/metering settle once, then full-press both back to back
        for c in (left, right):
            c.trigger_prepare(focus_wait=0)
        time.sleep(0.3)
        left.trigger_release()
        right.trigger_release()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for side, c in (("left", left), ("right", right)):
            c.wait_for_image()
            files = [c.download_image()]
            time.sleep(0.15)
            while c.pending_images():  # e.g. RAW+JPEG: keep the JPEG
                files.append(c.download_image())
                time.sleep(0.15)
            name, data = next(((n, d) for n, d in files if n.lower().endswith((".jpg", ".jpeg"))), files[0])
            p = out_dir / f"{side}.jpg"
            p.write_bytes(data)
            paths[side] = p
        self.state.emit("connected")
        self.captured.emit({**paths, "tag": tag, "seconds": time.time() - t0})

    def _settings(self, args):
        from sonyrx0.camera.settings import ORDER
        self._ensure()
        toks = args.split()
        wanted = {}
        for i, t in enumerate(toks):
            if t.startswith("--") and i + 1 < len(toks):
                wanted[t[2:].replace("-", "_")] = toks[i + 1]
        for c in self.cams.values():
            for name in ORDER:
                if name in wanted:
                    c.apply(name, wanted[name])

    def _focus_lock(self):
        self._ensure()
        for c in self.cams.values():
            c.focus_lock()

    def run(self):
        import comtypes
        comtypes.CoInitialize()
        try:
            while not self._stop:
                try:
                    cmd, arg = self._q.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd is not None:
                    tag = arg[1] if isinstance(arg, tuple) else None
                    try:
                        if cmd == "rig":
                            if arg != self.rig_file:
                                self.rig_file = arg
                                self._disconnect()
                        elif cmd == "capture":
                            self._capture(*arg)
                        elif cmd == "settings":
                            self._settings(arg[0])
                        elif cmd == "focus_lock":
                            self._focus_lock()
                    except Exception as e:  # noqa: BLE001 - report, reconnect next time
                        self._disconnect()
                        self.failed.emit(f"{type(e).__name__}: {e}", tag)
                    continue
                if self._live:
                    try:
                        self._ensure()
                        img = self.cams["left"].liveview_frame(timeout=0.5)
                        if img is not None:
                            if self._markers is not None:
                                import cv2
                                from chickpea.stage import find_markers
                                found = find_markers(img, self._markers)
                                for corners in found.values():
                                    cv2.polylines(img, [corners.astype(np.int32)], True, (60, 200, 60), 3)
                                cv2.putText(img, f"markers {len(found)}/4", (16, img.shape[0] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (60, 200, 60) if len(found) >= 2 else (40, 40, 220), 2, cv2.LINE_AA)
                            self.frame.emit(img)
                    except Exception as e:  # noqa: BLE001
                        self._live = False
                        self._disconnect()
                        self.failed.emit(f"Live view: {type(e).__name__}: {e}", None)
                else:
                    time.sleep(0.02)
        finally:
            self._disconnect()
