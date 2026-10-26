"""The OAK-D camera service: same interface as camera.CameraService (live view, capture, settings).

One thread owns the device and keeps one pipeline running: the 12 MP colour
image for captures, a half-size undistorted colour stream that the depth is
aligned to (live view shows it too), and close-range subpixel stereo depth. A
capture saves raw/rgb.jpg (full resolution) and raw/depth.npz: the per-pixel
median of several depth frames (mm, 0 = unknown) aligned to the half-size
colour, with the intrinsics of both images. The device's factory calibration
is all it needs: no calibration session.
"""
import queue
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

FULL = (4000, 3000)    # colour image saved per capture
ALIGN = (2000, 1500)   # depth is aligned to (and live view shows) this
DEPTH_FRAMES = 10      # averaged per capture: ~1 mm noise on seeds instead of ~2-4; ~4 s
MIN_RANGE_MM = 333     # closest depth with extended disparity (OAK-D: 7.5 cm baseline)
SAFE_RANGE_MM = 400    # live view warns below this: seed tops must stay beyond MIN_RANGE_MM


def parse_args(args):
    """'--exposure 8000 --iso 400 --focus 130 --wb 5000' -> dict (values may be 'auto')."""
    toks = args.split()
    return {t[2:]: toks[i + 1] for i, t in enumerate(toks) if t.startswith("--") and i + 1 < len(toks)}


def median_depth(frames):
    """Per-pixel median of uint16 depth frames, ignoring 0 (unknown); 0 where under half the frames had depth."""
    d = np.stack(frames).astype(np.float32)
    d[d == 0] = np.nan
    n = np.isfinite(d).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(d, axis=0)
    m[(n < len(frames) / 2) | ~np.isfinite(m)] = 0
    return np.round(m).astype(np.uint16)


class OakService(QThread):
    frame = Signal(object)          # live-view BGR frame
    captured = Signal(object)       # {"rgb": path, "depth": path, "tag": any, "seconds": float}
    failed = Signal(str, object)    # message, tag of the failed request (or None)
    state = Signal(str)             # "disconnected" | "connected" | "capturing"

    def __init__(self):
        super().__init__()
        self._q = queue.Queue()
        self._live = False
        self._stop = False
        self._markers = None  # sheet layout: live view outlines the corner markers it sees
        self.pipe = None

    # --- public, thread-safe ------------------------------------------------
    def set_rig_file(self, path):
        pass  # one device: no left/right assignment

    def set_markers(self, layout):
        self._markers = layout

    def set_live(self, on):
        self._live = bool(on)

    def capture(self, out_dir, tag=None):
        """Save <out_dir>/rgb.jpg and depth.npz."""
        self._q.put(("capture", (Path(out_dir), tag)))

    def apply_settings(self, args, tag=None):
        """'--exposure 8000 --iso 400 --wb 5000' (exposure in µs, wb in K; or 'auto').

        --focus: 'calib' (default: the factory calibration's lens position, which keeps
        depth aligned with colour), 'auto', or a lens position 0-255.
        """
        self._q.put(("settings", (args, tag)))

    def stop(self):
        self._stop = True
        self.wait(8000)

    # --- thread ---------------------------------------------------------------
    def _connect(self):
        try:
            import depthai as dai
        except ImportError as e:
            raise RuntimeError("The OAK needs the depthai package: run setup.bat again") from e
        if not dai.Device.getAllAvailableDevices():
            raise RuntimeError("No OAK camera found: plug it into a USB 3 port (blue) and try again.")
        S = dai.CameraBoardSocket
        p = dai.Pipeline()
        rgb = p.create(dai.node.Camera).build(S.CAM_A)
        left = p.create(dai.node.Camera).build(S.CAM_B)
        right = p.create(dai.node.Camera).build(S.CAM_C)
        stereo = p.create(dai.node.StereoDepth)
        stereo.setExtendedDisparity(True)  # halves the closest range: ~33 cm
        stereo.setSubpixel(True)
        stereo.setSubpixelFractionalBits(5)
        stereo.setLeftRightCheck(True)
        left.requestOutput((1280, 800)).link(stereo.left)
        right.requestOutput((1280, 800)).link(stereo.right)
        full = rgb.requestOutput(FULL, dai.ImgFrame.Type.NV12, enableUndistortion=True)
        half = rgb.requestOutput(ALIGN, dai.ImgFrame.Type.NV12, enableUndistortion=True)
        half.link(stereo.inputAlignTo)
        self.q_full = full.createOutputQueue(maxSize=1, blocking=False)
        self.q_half = half.createOutputQueue(maxSize=1, blocking=False)
        self.q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        self.q_ctrl = rgb.inputControl.createInputQueue()
        p.start()
        dev = p.getDefaultDevice()
        cal = dev.readCalibration()
        self.K_full = np.array(cal.getCameraIntrinsics(S.CAM_A, *FULL))
        self.K_depth = np.array(cal.getCameraIntrinsics(S.CAM_A, *ALIGN))
        self.lens_calib = int(cal.getLensPosition(S.CAM_A))
        self.device_id = dev.getDeviceId()
        self.pipe = p
        self._depth_mm = None
        # the colour calibration (hence depth-to-colour alignment) holds at the lens position it was made
        # at: autofocus on a close stage moves the lens and shifts the depth ~4 mm against the colour
        c = dai.CameraControl()
        c.setManualFocus(self.lens_calib)
        self.q_ctrl.send(c)
        self.state.emit("connected")

    def _disconnect(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        self.pipe = None
        self.state.emit("disconnected")

    def _ensure(self):
        if self.pipe is None:
            self._connect()

    def _capture(self, out_dir, tag):
        t0 = time.time()
        self._ensure()
        self.state.emit("capturing")
        while self.q_depth.tryGet() is not None:  # only frames taken after the request
            pass
        depths = [self.q_depth.get().getFrame() for _ in range(DEPTH_FRAMES)]
        img = self.q_full.get().getCvFrame()
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb_path, depth_path = out_dir / "rgb.jpg", out_dir / "depth.npz"
        cv2.imwrite(str(rgb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savez_compressed(depth_path, depth=median_depth(depths), K_rgb=self.K_full, K_depth=self.K_depth,
                            device=self.device_id, frames=len(depths))
        self.state.emit("connected")
        self.captured.emit({"rgb": rgb_path, "depth": depth_path, "tag": tag, "seconds": time.time() - t0})

    def _settings(self, args):
        import depthai as dai
        self._ensure()
        a = parse_args(args)
        c = dai.CameraControl()
        if a.get("exposure", "auto") == "auto" and a.get("iso", "auto") == "auto":
            c.setAutoExposureEnable()
        else:
            c.setManualExposure(int(float(a.get("exposure", 8000))), int(float(a.get("iso", 400))))
        focus = a.get("focus", "calib")
        if focus == "auto":  # sharper close up, but depth no longer lines up with the colour
            c.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        else:
            c.setManualFocus(self.lens_calib if focus == "calib" else int(float(focus)))
        if a.get("wb", "auto") == "auto":
            c.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        else:
            c.setManualWhiteBalance(int(float(a["wb"])))
        self.q_ctrl.send(c)

    def _live_frame(self):
        f = self.q_half.tryGet()
        if f is None:
            return None
        img = f.getCvFrame()
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        d = self.q_depth.tryGet()
        if d is not None:
            dm = d.getFrame()
            h, w = dm.shape
            c = dm[h // 3:2 * h // 3, w // 3:2 * w // 3]
            c = c[c > 0]
            self._depth_mm = float(np.median(c)) if c.size else None
        if self._markers is not None:
            from chickpea.stage import find_markers
            found = find_markers(img, self._markers)
            for corners in found.values():
                cv2.polylines(img, [corners.astype(np.int32)], True, (60, 200, 60), 3)
            cv2.putText(img, f"markers {len(found)}/4", (16, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (60, 200, 60) if len(found) >= 2 else (40, 40, 220), 2, cv2.LINE_AA)
        if self._depth_mm:
            close = self._depth_mm < SAFE_RANGE_MM
            text = f"stage {self._depth_mm / 10:.1f} cm" + ("  TOO CLOSE: raise the camera above 40 cm" if close else "")
            cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (40, 40, 220) if close else (60, 200, 60), 2, cv2.LINE_AA)
        return img

    def run(self):
        try:
            while not self._stop:
                try:
                    cmd, arg = self._q.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd is not None:
                    tag = arg[1] if isinstance(arg, tuple) else None
                    try:
                        if cmd == "capture":
                            self._capture(*arg)
                        elif cmd == "settings":
                            self._settings(arg[0])
                    except Exception as e:  # noqa: BLE001 - report, reconnect next time
                        self._disconnect()
                        self.failed.emit(f"{type(e).__name__}: {e}", tag)
                    continue
                if self._live:
                    try:
                        self._ensure()
                        img = self._live_frame()
                        if img is not None:
                            self.frame.emit(img)
                        else:
                            time.sleep(0.01)
                    except Exception as e:  # noqa: BLE001
                        self._live = False
                        self._disconnect()
                        self.failed.emit(f"Live view: {type(e).__name__}: {e}", None)
                else:
                    time.sleep(0.02)
        finally:
            self._disconnect()
