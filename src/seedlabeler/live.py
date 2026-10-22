"""Live-view helpers. The camera service (camera.py) streams the frames;
depth_zone() gives the outline drawn over them: where stereo gives depth.
"""
import json
import time

import numpy as np


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


def depth_zone(project, preview_size, margin=0.02, samples=40):
    """Outline (in live-preview pixels) of the area where the stereo pair gives depth.

    Depth exists only where both cameras see the stage and the matcher can search:
    on the left rectified image that excludes a band minDisparity+numDisparities
    wide on the left. That band's width follows from the calibration and the
    depth range (from the stage file if set, else the project settings), exactly
    as detection will use them. The rectified rectangle is mapped back to the raw
    left camera, which is what live view shows. Returns an (N, 2) array or None.
    """
    import cv2 as _cv2
    from sonyrx0.stereo.depth import StereoRig

    cfg, st = project.config, project.config.settings
    try:
        calib = np.load(project.path(cfg.calib))
        rig = StereoRig(project.path(cfg.calib), st.scale)
    except Exception:  # noqa: BLE001
        return None
    lo, hi = st.min_depth, st.max_depth
    if cfg.stage:
        try:
            d = float(np.load(project.path(cfg.stage))["d"])
            lo, hi = d - 40.0, d + 10.0  # same range detection uses with a stage file
        except Exception:  # noqa: BLE001
            pass
    min_disp, num_disp = rig.disparity_range(lo, hi)
    w, h = (int(v) for v in calib["image_size"])
    x0 = (min_disp + num_disp) / st.scale
    x0, x1 = x0 + margin * w, w * (1 - margin)
    y0, y1 = margin * h, h * (1 - margin)
    t = np.linspace(0, 1, samples)
    edge = np.concatenate([
        np.stack([x0 + (x1 - x0) * t, np.full_like(t, y0)], 1),
        np.stack([np.full_like(t, x1), y0 + (y1 - y0) * t], 1),
        np.stack([x1 - (x1 - x0) * t, np.full_like(t, y1)], 1),
        np.stack([np.full_like(t, x0), y1 - (y1 - y0) * t], 1)])
    # rectified pixel -> ray in the rectified frame -> raw left camera -> distorted pixel
    P1, R1, K1, D1 = calib["P1"], calib["R1"], calib["K1"], calib["D1"]
    rays = np.linalg.inv(P1[:3, :3]) @ np.vstack([edge.T, np.ones(len(edge))])
    cam = (R1.T @ rays).T
    pts, _ = _cv2.projectPoints(cam.astype(np.float64), np.zeros(3), np.zeros(3), K1, D1)
    pts = pts.reshape(-1, 2) * (preview_size[0] / w)
    return pts
