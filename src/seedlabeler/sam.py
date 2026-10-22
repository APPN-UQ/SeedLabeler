"""SAM (Segment Anything) prompts on the top-down stage image.

Prompts run on a full-resolution crop around the box, not the whole image:
SAM resizes its input to 1024 px, which would leave a seed 20-50 px wide in a
16 MP top-down image. Weights are downloaded once to ~/.seedlabeler/weights.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_DIR = Path.home() / ".seedlabeler" / "weights"
MODELS = {"mobile_sam.pt": "MOBILESAM (FAST)", "sam2.1_b.pt": "SAM 2.1 BASE (ACCURATE)"}
_lock = threading.Lock()
_models = {}


def get_model(name):
    """Load (and on first use download) a SAM model; cached and shared."""
    from ultralytics import SAM
    with _lock:
        if name not in _models:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            _models[name] = SAM(str(WEIGHTS_DIR / name))
        return _models[name]


def _polygon(mask, epsilon):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), epsilon, True).reshape(-1, 2)
    return c.astype(float) if len(c) >= 3 else None


def segment(model_name, ortho, box, points=None, point_labels=None, height=None, res=None, margin=0.4,
            seed_px=None):
    """One seed's outline from a box (x0, y0, x1, y1) and/or clicks, in ortho pixels.

    points: [(x, y), ...] in ortho pixels; point_labels: 1 = seed, 0 = not seed.
    box may be None for a click-only prompt: the crop is then centred on the
    first click and seed_px (a seed width in pixels) wide on each side.
    Returns a polygon (N, 2) in ortho pixels, or None.
    """
    use_box = box is not None
    if not use_box:
        px, py = points[0]
        r = 1.2 * (seed_px or 100)
        box = (px - r, py - r, px + r, py + r)
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    pad = margin * max(x1 - x0, y1 - y0, 10)
    h, w = ortho.shape[:2]
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    crop = np.ascontiguousarray(ortho[cy0:cy1, cx0:cx1])
    kw = {"bboxes": [[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]]} if use_box else {}
    if points:
        kw["points"] = [[[px - cx0, py - cy0] for px, py in points]]
        kw["labels"] = [list(point_labels or [1] * len(points))]
    model = get_model(model_name)
    with _lock:  # one prompt at a time on the GPU
        r = model(crop, verbose=False, **kw)[0]
    if r.masks is None or not len(r.masks.data):
        return None
    m = r.masks.data[0].cpu().numpy() > 0.5
    # No height-based shadow trim here: SAM separates objects by shape and
    # texture and already leaves shadows out; stereo depth is unreliable along
    # seed edges, so trimming on it only cut notches into good outlines.
    # remove thin attached slivers (a strip of shadow along an edge): open with a
    # disk ~0.3 mm in radius, which leaves the seed's own outline essentially intact
    if res:
        r = max(1, int(round(0.3 / res)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        if opened.sum() > 0.5 * m.sum():
            m = opened
    # keep the piece that overlaps the box (or sits under the click) most
    n, comp = cv2.connectedComponents(m.astype(np.uint8))
    if n > 2 and not use_box:
        px, py = int(points[0][0] - cx0), int(points[0][1] - cy0)
        k = comp[min(max(py, 0), comp.shape[0] - 1), min(max(px, 0), comp.shape[1] - 1)]
        if k:
            m = comp == k
            n = 0
    if n > 2:
        inside = comp[int(y0 - cy0):int(y1 - cy0) + 1, int(x0 - cx0):int(x1 - cx0) + 1]
        counts = np.bincount(inside.ravel(), minlength=n)
        counts[0] = 0
        m = comp == int(np.argmax(counts))
    eps = max(1.0, 0.2 / res) if res else 1.0
    poly = _polygon(m, eps)
    return None if poly is None else poly + [cx0, cy0]


def bbox(polygon, pad=0.08):
    p = np.asarray(polygon, float)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    d = pad * max(x1 - x0, y1 - y0)
    return x0 - d, y0 - d, x1 + d, y1 + d


def _iou(a, b, shape):
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a)).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b)).astype(np.int32)], 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / union if union else 0.0


def core_prompt(polygon, shave=0.15):
    """A clean prompt from an existing outline: (box, centre point).

    Outlines from colour segmentation can carry a shadow skirt; its bounding box
    then invites SAM to include the shadow. Opening the outline with a disk
    ~shave of its smaller dimension removes thin appendages, and the box and
    centre of what remains make a tight, unambiguous prompt.
    """
    p = np.asarray(polygon, float)
    x0, y0 = np.floor(p.min(0)).astype(int) - 2
    x1, y1 = np.ceil(p.max(0)).astype(int) + 2
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32)], 1)
    r = max(1, int(shave * min(x1 - x0, y1 - y0) / 2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if core.sum() < 0.5 * m.sum():
        core = m
    ys, xs = np.nonzero(core)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    d = 0.06 * max(bx1 - bx0, by1 - by0)
    box = (bx0 - d + x0, by0 - d + y0, bx1 + d + x0, by1 + d + y0)
    # centre: the core pixel farthest from its edge (inside even for curved seeds like cashews)
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    return box, (float(cx + x0), float(cy + y0))


def refine_all(model_name, ortho, seeds, height=None, res=None, min_iou=0.6):
    """Redraw every seed with SAM, prompted by its current outline (tight box + centre click).

    A SAM outline replaces the old one only if they broadly agree (IoU >=
    min_iou): SAM can occasionally grab a neighbour or a fragment, and then the
    original is the safer answer. Hand-drawn outlines are left alone.
    Returns (new seeds, number replaced).
    """
    out, replaced = [], 0
    for s in seeds:
        if s["status"] == "manual":
            out.append(s)
            continue
        box, centre = core_prompt(s["polygon"])
        poly = segment(model_name, ortho, box, [centre], [1], height=height, res=res)
        if poly is not None:
            x0, y0, x1, y1 = (int(v) for v in bbox(s["polygon"], pad=0.5))
            x0, y0 = max(0, x0), max(0, y0)
            local = (lambda q: np.asarray(q) - [x0, y0])
            shape = (max(1, y1 - y0), max(1, x1 - x0))
            if _iou(local(poly), local(s["polygon"]), shape) >= min_iou:
                out.append({**s, "polygon": poly.tolist(), "sam": True})
                replaced += 1
                continue
        out.append(s)
    return out, replaced
