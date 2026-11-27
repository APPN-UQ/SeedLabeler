"""Detection, polygon <-> mask conversion and (re-)measurement of a sample.

All geometry lives on the sample's top-down stage grid (ortho pixels, with
`res` mm per pixel), the same space the traits are measured in.
"""
import json
import shutil

import cv2
import numpy as np
import pandas as pd

from chickpea.measure import measure_seeds
from chickpea.pipeline import measure_pair, measure_rgbd
from chickpea.stage import Stage
from chickpea.summary import summarise
from sonyrx0.stereo.common import read_image
from sonyrx0.stereo.depth import StereoRig, colourise

_rigs = {}


def check_swapped(calib, img_l, img_r):
    """True if the pair looks left/right swapped relative to the calibration.

    In a correct pair, matching features sit on the same row after rectification
    and lie further right in the left image (positive disparity). A swapped
    pair breaks both. Checked on quarter-size images with ORB features (~0.5 s).
    """
    rig = _rig(calib, 0.25)
    rl, rr = rig.rectify(img_l, img_r)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return False
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 30:
        return False
    p1 = np.array([k1[x.queryIdx].pt for x in m])
    p2 = np.array([k2[x.trainIdx].pt for x in m])
    dy, dx = p1[:, 1] - p2[:, 1], p1[:, 0] - p2[:, 0]
    consistent = np.abs(dy - np.median(dy)) < 3
    return bool(np.median(dx[consistent]) < 0)


def _rig(calib, scale):
    key = (str(calib), scale)
    if key not in _rigs:
        _rigs[key] = StereoRig(calib, scale)
    return _rigs[key]


# --- polygons <-> labels -------------------------------------------------
def labels_to_polygons(labels, epsilon=1.0):
    polys = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        if not m.any():
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append(c.astype(float).tolist())
    return polys


def polygons_to_labels(polys, shape):
    labels = np.zeros(shape, np.int32)
    for i, p in enumerate(polys, start=1):
        cv2.fillPoly(labels, [np.round(np.asarray(p)).astype(np.int32)], i)
    return labels


_RANK = {"manual": 2, "edited": 1}  # whose outline wins a shared pixel (auto: 0)


def resolve_overlaps(seeds, shape, epsilon=1.0, protect=None):
    """Make outlines disjoint: seen from above, two seeds can't share a pixel.

    A shared pixel goes to the protected seed (the one being edited, if any),
    then to hand-drawn over edited over automatic outlines, then to the seed
    whose centre is nearest. Seeds that lose more than a couple of pixels are
    re-traced; a seed left with almost nothing (another outline was drawn over
    it) is removed. Returns (seeds, kept) where kept[j] is the input index of
    output seed j, so a selection can follow its seed.
    """
    n = len(seeds)
    if n < 2:
        return list(seeds), list(range(n))
    H, W = shape
    boxes, masks = [], []
    count = np.zeros(shape, np.uint16)
    for s in seeds:
        p = np.round(np.asarray(s["polygon"], float)).astype(np.int32).reshape(-1, 2)
        x0, y0 = max(int(p[:, 0].min()), 0), max(int(p[:, 1].min()), 0)
        x1, y1 = min(int(p[:, 0].max()) + 1, W), min(int(p[:, 1].max()) + 1, H)
        if x1 <= x0 or y1 <= y0:  # outside the image: nothing to share
            boxes.append(None)
            masks.append(None)
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [p - [x0, y0]], 1)
        boxes.append((x0, y0, x1, y1))
        masks.append(m.astype(bool))
        count[y0:y1, x0:x1] += m
    if count.max() <= 1:
        return list(seeds), list(range(n))

    best = np.full(shape, -np.inf, np.float32)
    owner = np.full(shape, -1, np.int32)
    for i, (s, b, m) in enumerate(zip(seeds, boxes, masks)):
        if m is None or not m.any():
            continue
        x0, y0, x1, y1 = b
        ys, xs = np.nonzero(m)
        yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        rank = 10 if i == protect else _RANK.get(s.get("status"), 0)
        score = rank * 1e6 - np.hypot(yy - ys.mean(), xx - xs.mean())
        sub_b, sub_o = best[y0:y1, x0:x1], owner[y0:y1, x0:x1]  # views: written in place
        win = m & (score > sub_b)
        sub_b[win] = score[win]
        sub_o[win] = i

    out, kept = [], []
    for i, (s, b, m) in enumerate(zip(seeds, boxes, masks)):
        if m is None:
            out.append(s)
            kept.append(i)
            continue
        x0, y0, x1, y1 = b
        mine = m & (owner[y0:y1, x0:x1] == i)
        if m.sum() - mine.sum() <= 2:  # a pixel or two along a shared edge: leave the outline be
            out.append(s)
            kept.append(i)
            continue
        if mine.sum() < 3:
            continue  # drawn over by another outline
        cs, _ = cv2.findContours(mine.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), epsilon, True).reshape(-1, 2)
        if len(c) < 3 or cv2.contourArea(c.astype(np.float32)) < 1:
            continue
        out.append({**s, "polygon": (c + [x0, y0]).astype(float).tolist()})
        kept.append(i)
    return out, kept


# --- detection -------------------------------------------------------------
def rgbd_xyz(depth_mm, K):
    """Depth image (mm, 0 = unknown) -> (h, w, 3) points in the colour camera frame, NaN where unknown."""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32)
    z[z == 0] = np.nan
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.dstack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])


def detect(project, sample, model_detector=None):
    """Depth + detection on a sample's raw images (stereo pair or OAK colour + depth); saves everything.

    Returns annotations. The stage plane comes from the corner markers if the
    project uses them, else the stage file, else a fit to the background's depth.
    """
    cfg, st = project.config, project.config.settings
    stage = Stage.load(project.path(cfg.stage)) if cfg.stage else None
    kw = {"min_seed_width_mm": st.seed_width, "min_area_mm2": st.min_area, "min_thickness_mm": st.min_thickness,
          "detector": model_detector, "markers": project.markers, "level": st.level_height}
    swapped = False
    if sample.is_rgbd:
        with np.load(sample.depth_file) as z:
            depth, K_rgb, K_depth = z["depth"], z["K_rgb"], z["K_depth"]
        # a stage file is in another camera's frame: only markers or the depth itself fit an OAK capture
        seeds, ex = measure_rgbd(read_image(sample.rgb), K_rgb, rgbd_xyz(depth, K_depth), K_depth, **kw)
    else:
        if not cfg.calib:
            raise RuntimeError("Set the stereo calibration file in project settings first")
        calib = project.path(cfg.calib)
        rig = _rig(calib, st.scale)
        if stage is None:
            kw.update(min_depth=st.min_depth, max_depth=st.max_depth)
        img_l, img_r = read_image(sample.left), read_image(sample.right)
        swapped = check_swapped(calib, img_l, img_r)
        if swapped:  # fix it at the source so the stored pair is right from now on
            img_l, img_r = img_r, img_l
            tmp = sample.left.with_name("swap.tmp")
            sample.left.rename(tmp)
            sample.right.rename(sample.left)
            tmp.rename(sample.right)
        seeds, ex = measure_pair(rig, img_l, img_r, stage, **kw)
    grid, stage = ex["grid"], ex["stage"]
    cv2.imwrite(str(sample.ortho_file), ex["ortho"], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    np.savez_compressed(sample.height_file, height=ex["height"].astype(np.float16),
                        valid=ex["valid"])
    depth_cov = float(np.isfinite(ex["height"][ex["valid"]]).mean()) if ex["valid"].any() else 0.0
    meta = {"res": grid.res, "a0": grid.a0, "b0": grid.b0, "stage_distance": stage.d,
            "stage_method": stage.method, "detector": cfg.detector, "swapped_fixed": swapped,
            "depth_coverage": round(depth_cov, 3), "camera": "oak" if sample.is_rgbd else "rx0",
            "levelled": st.level_height, **ex["ground"], **ex["segmentation"]}
    sample.meta_file.write_text(json.dumps(meta, indent=2))
    # colour calibration snapshot: the one it was captured with (raw/colourcal.json, from an
    # import) if there is one, else the project's current one
    cal = sample.path / "raw" / "colourcal.json"
    if not cal.exists():
        cal = project.root / "calibration" / "colourcal.json"
    if cal.exists():
        shutil.copy2(cal, sample.path / "colourcal.json")

    source = "classic" if model_detector is None else f"model:{cfg.detector}"
    polys = labels_to_polygons(ex["labels"], epsilon=max(1.0, 0.2 / grid.res))  # ~0.2 mm: few handles, exact enough
    old = sample.annotations()
    ann = {"seeds": [{"polygon": p, "status": "auto"} for p in polys],
           "auto": polys, "source": source,
           "reviewed": False, "in_dataset": old.get("in_dataset", False)}
    sample.save_annotations(ann)
    remeasure(project, sample, ann)
    return ann


def _no_seeds(ortho, valid, res, height):
    return np.zeros(ortho.shape[:2], np.int32), {}


def prepare(project, sample):
    """Build the top-down image and height map only (no seed detection), so a sample can be
    labelled by hand straight away. The sample still counts as new: PROCESS and DETECT
    treat it as not yet detected until it has seeds."""
    ann = detect(project, sample, _no_seeds)
    ann["source"] = "prepared"
    sample.save_annotations(ann)
    return ann


# --- measurement -----------------------------------------------------------
def load_layers(sample):
    ortho = cv2.imread(str(sample.ortho_file))
    with np.load(sample.height_file) as z:  # closed at once: an open file blocks deleting the sample
        return ortho, z["height"].astype(np.float32), z["valid"]


def _colour_corrected(project, ortho, sample=None):
    """Apply colour calibration: per-sample snapshot first, then project-level fallback."""
    cal = None
    if sample is not None:
        per_sample = sample.path / "colourcal.json"
        if per_sample.exists():
            cal = per_sample
    if cal is None:
        project_cal = project.root / "calibration" / "colourcal.json"
        if project_cal.exists():
            cal = project_cal
    if cal is not None:
        from colourcal.correct import load_calibration, apply_correction
        M = load_calibration(cal)
        return apply_correction(ortho, M)
    return ortho


def remeasure(project, sample, ann, layers=None):
    """Traits from the current polygons; writes the sample's CSVs. Returns (seeds, summary)."""
    meta = sample.meta()
    ortho, height, _ = layers if layers is not None else load_layers(sample)
    ortho_for_colour = _colour_corrected(project, ortho, sample)
    polys = [s["polygon"] for s in ann["seeds"]]
    labels = polygons_to_labels(polys, ortho.shape[:2])
    res = meta["res"]
    rows = measure_seeds(labels, ortho_for_colour, height, res, meta["stage_distance"],
                         pixel_offset=(meta["a0"] / res, meta["b0"] / res))
    seeds = pd.DataFrame(rows)
    if len(seeds):
        status = {i + 1: s["status"] for i, s in enumerate(ann["seeds"])}
        seeds.insert(1, "status", seeds.seed_id.map(status))
        labels = {i + 1: s.get("label", "") for i, s in enumerate(ann["seeds"])}
        seeds.insert(2, "label", seeds.seed_id.map(labels).replace("", np.nan))
    seeds.insert(0, "sample", sample.id)
    seeds.insert(1, "crop", project.config.crop)
    seeds.to_csv(sample.seeds_csv, index=False)
    numeric = seeds.drop(columns=[c for c in ("sample", "crop", "status", "label") if c in seeds])
    summary = summarise(numeric) if len(seeds) else pd.DataFrame()
    if len(summary):
        summary.insert(0, "sample", sample.id)
        summary.insert(1, "crop", project.config.crop)
    summary.to_csv(sample.summary_csv, index=False)
    return seeds, summary


def height_preview(height):
    h = np.where(np.isfinite(height), np.clip(height, 0, None), np.nan)
    return colourise(h)


def export_project(project):
    """Concatenate every sample's traits into project-wide tables."""
    out = project.root / "exports"
    out.mkdir(exist_ok=True)
    seeds, sums = [], []
    for s in project.samples():
        if s.seeds_csv.exists():
            seeds.append(pd.read_csv(s.seeds_csv))
        if s.summary_csv.exists() and s.summary_csv.stat().st_size > 2:
            sums.append(pd.read_csv(s.summary_csv))
    name = project.config.name.replace(" ", "_")
    paths = []
    if seeds:
        p = out / f"{name}_seeds.csv"
        pd.concat(seeds, ignore_index=True).to_csv(p, index=False)
        paths.append(p)
    if sums:
        p = out / f"{name}_summary.csv"
        pd.concat(sums, ignore_index=True).to_csv(p, index=False)
        paths.append(p)
    return paths
