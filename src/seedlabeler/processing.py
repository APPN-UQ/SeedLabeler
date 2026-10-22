"""Detection, polygon <-> mask conversion and (re-)measurement of a sample.

All geometry lives on the sample's top-down stage grid (ortho pixels, with
`res` mm per pixel), the same space the traits are measured in.
"""
import json

import cv2
import numpy as np
import pandas as pd

from chickpea.measure import measure_seeds
from chickpea.pipeline import measure_pair
from chickpea.stage import Stage
from chickpea.summary import summarise
from sonyrx0.stereo.common import read_image
from sonyrx0.stereo.depth import StereoRig, colourise

_rigs = {}


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


# --- detection -------------------------------------------------------------
def detect(project, sample, model_detector=None):
    """Run stereo + detection on a sample's raw pair and save everything. Returns annotations."""
    cfg, st = project.config, project.config.settings
    if not cfg.calib:
        raise RuntimeError("Set the stereo calibration file in project settings first")
    rig = _rig(project.path(cfg.calib), st.scale)
    stage = Stage.load(project.path(cfg.stage)) if cfg.stage else None
    kw = {} if stage else {"min_depth": st.min_depth, "max_depth": st.max_depth}
    seeds, ex = measure_pair(rig, read_image(sample.left), read_image(sample.right), stage,
                             min_seed_width_mm=st.seed_width, min_area_mm2=st.min_area,
                             min_thickness_mm=st.min_thickness, detector=model_detector, **kw)
    grid, stage = ex["grid"], ex["stage"]
    cv2.imwrite(str(sample.ortho_file), ex["ortho"], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    np.savez_compressed(sample.height_file, height=ex["height"].astype(np.float16),
                        valid=ex["valid"])
    meta = {"res": grid.res, "a0": grid.a0, "b0": grid.b0, "stage_distance": stage.d,
            "stage_method": stage.method, "detector": cfg.detector, **ex["segmentation"]}
    sample.meta_file.write_text(json.dumps(meta, indent=2))

    source = "classic" if model_detector is None else f"model:{cfg.detector}"
    polys = labels_to_polygons(ex["labels"], epsilon=max(1.0, 0.2 / grid.res))  # ~0.2 mm: few handles, exact enough
    old = sample.annotations()
    ann = {"seeds": [{"polygon": p, "status": "auto"} for p in polys],
           "auto": polys, "source": source,
           "reviewed": False, "in_dataset": old.get("in_dataset", False)}
    sample.save_annotations(ann)
    remeasure(project, sample, ann)
    return ann


# --- measurement -----------------------------------------------------------
def load_layers(sample):
    ortho = cv2.imread(str(sample.ortho_file))
    z = np.load(sample.height_file)
    return ortho, z["height"].astype(np.float32), z["valid"]


def remeasure(project, sample, ann, layers=None):
    """Traits from the current polygons; writes the sample's CSVs. Returns (seeds, summary)."""
    meta = sample.meta()
    ortho, height, _ = layers if layers is not None else load_layers(sample)
    polys = [s["polygon"] for s in ann["seeds"]]
    labels = polygons_to_labels(polys, ortho.shape[:2])
    res = meta["res"]
    rows = measure_seeds(labels, ortho, height, res, meta["stage_distance"],
                         pixel_offset=(meta["a0"] / res, meta["b0"] / res))
    seeds = pd.DataFrame(rows)
    if len(seeds):
        status = {i + 1: s["status"] for i, s in enumerate(ann["seeds"])}
        seeds.insert(1, "status", seeds.seed_id.map(status))
    seeds.insert(0, "sample", sample.id)
    seeds.insert(1, "crop", project.config.crop)
    seeds.to_csv(sample.seeds_csv, index=False)
    numeric = seeds.drop(columns=[c for c in ("sample", "crop", "status") if c in seeds])
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
