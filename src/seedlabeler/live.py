"""Live camera view with an optional fast seed overlay.

The thread owns the left camera for as long as live view runs (COM objects
must stay on the thread that created them) and closes it cleanly when
stopped, so a capture can open both cameras straight afterwards.

The overlay is a 2D-only version of the classic detector run on the ~1 MP
preview: stage region, colour threshold, distance-transform splitting. It is
for framing and spacing seeds before capturing, not for measuring.
"""
import json
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from scipy import ndimage as ndi
from skimage import morphology, segmentation

from chickpea.segment import _seed_threshold, _watershed_split, find_stage_region, to_lab


def fast_detect(bgr, res, seed_width_mm, cache=None, lightness_weight=0.3, work_width=512):
    """Seed outlines on a preview frame. res: approx. mm per preview pixel. Returns contours."""
    f = min(1.0, work_width / bgr.shape[1])
    if f < 1.0:
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        res = res / f
    lab = to_lab(bgr)
    valid = np.ones(lab.shape[:2], bool)
    if cache is None or time.time() - cache.get("t", 0) > 2.0:  # stage region changes slowly
        bg, region, contrast = find_stage_region(lab, valid, lightness_weight, work_side=400)
        cache = {"t": time.time(), "bg": bg, "region": region, "contrast": contrast}
    bg, region = cache["bg"], cache["region"]
    diff = lab - bg
    diff[..., 0] *= lightness_weight
    dist = cv2.GaussianBlur(np.linalg.norm(diff, axis=2), (0, 0), 1.0)
    mask = (dist > _seed_threshold(dist, region, cache["contrast"])) & region
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_fill_holes(mask)
    lab_ids, _ = ndi.label(mask)
    sizes = np.bincount(lab_ids.ravel())
    min_px = 0.25 * (seed_width_mm / res) ** 2
    mask = (sizes >= min_px)[lab_ids] & mask
    labels = _watershed_split(mask, res, seed_width_mm) if mask.any() else np.zeros(mask.shape, int)
    # objects touching the edge of the stage region are surroundings, not seeds
    rim = region & ~ndi.binary_erosion(region, iterations=3)
    for i in np.unique(labels[rim]):
        if i:
            labels[labels == i] = 0
    contours = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            contours.append((max(cs, key=cv2.contourArea) / f).astype(np.int32))
    return contours, cache


class LiveThread(QThread):
    frame = Signal(object, int)  # BGR image with overlay, seed count (-1 = overlay off)
    failed = Signal(str)

    def __init__(self, serial=None, detect=False, res=0.4, seed_width=6.0):
        super().__init__()
        self.serial, self.detect, self.res, self.seed_width = serial, detect, res, seed_width
        self._stop = False

    def _detect_loop(self, shared):
        cache = None
        while not self._stop:
            img, shared["frame"] = shared["frame"], None
            if img is None or not self.detect:
                time.sleep(0.02)
                continue
            try:
                shared["contours"], cache = fast_detect(img, self.res, self.seed_width, cache)
            except Exception:  # noqa: BLE001 - an odd frame must not stop the overlay
                shared["contours"] = []

    def stop(self):
        self._stop = True
        self.wait(5000)

    def run(self):
        import comtypes
        from sonyrx0.camera.sony import RX0
        from sonyrx0.camera.wpd_mtp import find_devices

        comtypes.CoInitialize()
        cam = None
        try:
            paths = find_devices()
            if not paths:
                raise RuntimeError("No camera found. Is it on, connected, and in PC Remote mode?")
            path = next((p for p in paths if self.serial and self.serial in p), paths[0])
            cam = RX0(path)
            shared = {"frame": None, "contours": [], "busy": False}
            detector = threading.Thread(target=self._detect_loop, args=(shared,), daemon=True)
            detector.start()
            while not self._stop:
                img = cam.liveview_frame()
                if img is None:
                    continue
                n = -1
                if self.detect:
                    shared["frame"] = img.copy()
                    cv2.drawContours(img, shared["contours"], -1, (255, 255, 255), 1, cv2.LINE_AA)
                    n = len(shared["contours"])
                self.frame.emit(img, n)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            if cam is not None:
                cam.close()


def preview_res(project, preview_width=1024):
    """Approximate mm per live-preview pixel from the calibration and the stage distance."""
    try:
        K = np.load(project.path(project.config.calib))["K1"]
        size = np.load(project.path(project.config.calib))["image_size"]
        f_prev = K[0, 0] * preview_width / float(size[0])
    except Exception:  # noqa: BLE001
        f_prev = 3460 * preview_width / 4800  # RX0 at full resolution
    d = None
    if project.config.stage:
        try:
            d = float(np.load(project.path(project.config.stage))["d"])
        except Exception:  # noqa: BLE001
            d = None
    if d is None:  # last processed sample, else the middle of the depth range
        for s in reversed(project.samples()):
            if s.meta_file.exists():
                d = json.loads(s.meta_file.read_text()).get("stage_distance")
                if d:
                    break
    if not d:
        st = project.config.settings
        d = (st.min_depth + st.max_depth) / 2
    return d / f_prev


def left_serial(project):
    try:
        return json.loads(project.rig_file.read_text()).get("left")
    except Exception:  # noqa: BLE001
        return None
