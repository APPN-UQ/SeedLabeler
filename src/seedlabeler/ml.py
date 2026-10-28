"""Instance-segmentation datasets and models (one of each per crop).

Datasets are in YOLO segmentation format, built from reviewed samples'
top-down images and corrected polygons. Images are resampled to a fixed scale
(~PX_PER_SEED pixels across the crop's smallest seed) and cut into overlapping
TILE-pixel tiles, so seeds look the same size to the model whatever the camera
height, and a few photos give many training examples. Detection tiles the
image the same way. Training runs in a separate process
(seedlabeler.train_worker) so the GUI stays responsive. Uses Ultralytics YOLO
(AGPL-3.0: fine for research use; check before distributing the app).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .processing import load_layers

BASE_MODEL = "yolo11s-seg.pt"
INPUTS = {"rgb": "COLOUR", "rgbh": "COLOUR + HEIGHT"}
TILE = 640
PX_PER_SEED = 64      # pixels across the smallest seed (project seed width)
OVERLAP_SEEDS = 2.5   # tile overlap in seed widths: covers a seed's full length


def make_input(ortho, height, valid, kind="rgb", height_scale_mm=6.0):
    """The image a model sees.

    rgb : the top-down colour image.
    rgbh: three channels packing what separates seeds from everything else:
          brightness (L*), colour saturation (chroma) and height above the stage
          (0 at the stage, 255 at height_scale_mm; unknown height -> 0). A shadow
          has a seed's colour but no height, so the model can learn to ignore it.
          Three channels keep it compatible with models pretrained on colour images.
    """
    if kind == "rgb":
        img = ortho.copy()
    else:
        from chickpea.segment import to_lab
        lab = to_lab(ortho)
        L = np.clip(lab[..., 0] * 2.55, 0, 255)
        C = np.clip(np.hypot(lab[..., 1], lab[..., 2]) * 4.0, 0, 255)
        H = np.clip(np.nan_to_num(height, nan=0.0) / height_scale_mm * 255.0, 0, 255)
        img = np.dstack([H, C, L]).astype(np.uint8)  # BGR order on disk: B=height, G=chroma, R=brightness
    img[~valid] = 0
    return img


def dataset_root(project, kind="rgb"):
    return project.dataset_dir if kind == "rgb" else project.dataset_dir.with_name(project.config.crop + "-rgbh")


# --- tiling ----------------------------------------------------------------
def tiling(seed_width_mm):
    """(work resolution mm/px, overlap px) for a crop."""
    work_res = seed_width_mm / PX_PER_SEED
    return work_res, int(round(OVERLAP_SEEDS * PX_PER_SEED))


def _starts(n, tile, step):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, step))
    return s + [n - tile]


def tile_grid(shape, overlap):
    h, w = shape[:2]
    step = TILE - overlap
    return [(x, y) for y in _starts(h, TILE, step) for x in _starts(w, TILE, step)]


def _crop(img, x, y):
    t = img[y:y + TILE, x:x + TILE]
    if t.shape[0] < TILE or t.shape[1] < TILE:  # image smaller than a tile: pad with black
        pad = np.zeros((TILE, TILE) + img.shape[2:], img.dtype)
        pad[:t.shape[0], :t.shape[1]] = t
        t = pad
    return t


def _resample(img, f, interp):
    return img if f >= 1 else cv2.resize(img, None, fx=f, fy=f, interpolation=interp)


# --- dataset ---------------------------------------------------------------
def _split(sample_id, val_fraction=0.2):
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def _sample_files(root, sample_id):
    """Every dataset file of one sample (whole-image or tiles), any split."""
    out = []
    for sub in ("images", "labels"):
        out += list(root.glob(f"{sub}/*/{sample_id}.*")) + list(root.glob(f"{sub}/*/{sample_id}__t*"))
    return out


def add_to_dataset(project, sample, empty_every=4):
    """Write (or refresh) one sample into the crop's datasets (colour and colour+height), as tiles."""
    ann = sample.annotations()
    ortho, height, valid = load_layers(sample)
    res = sample.meta()["res"]
    work_res, overlap = tiling(project.config.settings.seed_width)
    f = res / work_res
    polys = [np.asarray(s["polygon"], float) * min(f, 1.0) for s in ann["seeds"]]
    v = _resample(valid.astype(np.uint8), f, cv2.INTER_NEAREST).astype(bool)
    split = _split(sample.id)

    # which tiles, and their labels (shared by both input kinds)
    tiles = []
    n_empty = 0
    for x, y in tile_grid(v.shape, overlap):
        if _crop(v, x, y).mean() < 0.3:
            continue  # mostly outside the camera's view
        lines, has_centre = [], False
        for p in polys:
            q = p - [x, y]
            if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() >= TILE or q[:, 1].min() >= TILE:
                continue
            m = np.zeros((TILE, TILE), np.uint8)
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) < 30 or len(c) < 3:
                continue
            c = c.reshape(-1, 2) / TILE
            lines.append("0 " + " ".join(f"{val:.6f}" for val in c.clip(0, 1).ravel()))
            cx, cy = q.mean(axis=0)
            has_centre |= 0 <= cx < TILE and 0 <= cy < TILE
        if not has_centre:
            n_empty += 1
            if n_empty % empty_every:  # keep some empty tiles as background examples
                continue
        tiles.append((x, y, lines))

    for kind, ext, params in (("rgb", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
                              ("rgbh", ".png", [cv2.IMWRITE_PNG_COMPRESSION, 3])):  # PNG: no channel mixing
        root = dataset_root(project, kind)
        for old in _sample_files(root, sample.id):
            old.unlink()
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        img = _resample(make_input(ortho, height, valid, kind, project.config.settings.seed_width), f,
                        cv2.INTER_AREA)
        for x, y, lines in tiles:
            name = f"{sample.id}__t{x}_{y}"
            cv2.imwrite(str(root / "images" / split / f"{name}{ext}"), _crop(img, x, y), params)
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        write_data_yaml(project, kind)
    ann["in_dataset"] = True
    ann["reviewed"] = True
    sample.save_annotations(ann)
    return split


def remove_from_dataset(project, sample):
    for kind in INPUTS:
        for f in _sample_files(dataset_root(project, kind), sample.id):
            f.unlink()
    ann = sample.annotations()
    ann["in_dataset"] = False
    sample.save_annotations(ann)


def dataset_counts(project):
    """{split: (samples, tiles, seeds)} for the crop's dataset."""
    out = {}
    for split in ("train", "val"):
        d = project.dataset_dir / "labels" / split
        files = list(d.glob("*.txt")) if d.exists() else []
        samples = {f.stem.split("__t")[0] for f in files}
        out[split] = (len(samples), len(files), sum(len(f.read_text().splitlines()) for f in files))
    return out


def dataset_summary(project):
    """(samples, seeds) in the crop's dataset, counting each seed once (tiles overlap)."""
    root = dataset_root(project, "rgb")
    ids = {f.stem.split("__t")[0] for f in root.glob("labels/*/*.txt")}
    samples = [s for s in project.samples() if s.id in ids and s.processed]
    return len(samples), sum(len(s.annotations()["seeds"]) for s in samples)


def sync_datasets(project):
    """(Re)write every dataset sample in both input formats with the current tiling."""
    done = 0
    keep = set()
    for s in project.samples():
        if s.processed and s.annotations().get("in_dataset"):
            add_to_dataset(project, s)
            keep.add(s.id)
            done += 1
    # files of samples that were deleted, renamed or taken out of the dataset
    for kind in INPUTS:
        root = dataset_root(project, kind)
        for f in list(root.glob("images/*/*")) + list(root.glob("labels/*/*.txt")):
            if f.stem.split("__t")[0] not in keep:
                f.unlink()
    return done


def write_data_yaml(project, kind="rgb"):
    root = dataset_root(project, kind)
    counts = dataset_counts(project)
    # Ultralytics needs a validation split; until one exists, validate on train
    val = "images/val" if counts["val"][0] else "images/train"
    y = root / "data.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: {val}\n"
                 f"names:\n  0: {project.config.crop}\n")
    return y


# --- models ----------------------------------------------------------------
def list_models(project):
    """Trained models for the current crop, newest first: dicts with name, path, scores."""
    out = []
    d = project.models_dir
    if not d.exists():
        return out
    for run in sorted(d.iterdir(), reverse=True):
        best = run / "weights" / "best.pt"
        if not best.exists():
            continue
        info = {"name": run.name, "path": best, "mask_map50_95": None, "mask_map50": None, "epochs": None,
                "input": "rgb"}
        res = run / "results.csv"
        if res.exists():
            rows = list(csv.DictReader(res.open()))
            if rows:
                rows = [{k.strip(): v for k, v in r.items()} for r in rows]
                best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)", 0) or 0))
                info["mask_map50_95"] = float(best_row.get("metrics/mAP50-95(M)", 0) or 0)
                info["mask_map50"] = float(best_row.get("metrics/mAP50(M)", 0) or 0)
                info["epochs"] = len(rows)
        meta = run / "seedlabeler.json"
        if meta.exists():
            info.update(json.loads(meta.read_text()))
        out.append(info)
    return out


def latest_model(project):
    models = list_models(project)
    return models[0] if models else None


def samples_since_model(project, model=None):
    """Dataset samples added (or changed) since the model was trained: what a retrain would add."""
    model = model or latest_model(project)
    root = dataset_root(project, "rgb")
    times = {}
    for f in root.glob("labels/*/*.txt"):
        sid = f.stem.split("__t")[0]
        times[sid] = max(times.get(sid, 0), f.stat().st_mtime)
    if model is None:
        return len(times)
    t_model = model["path"].stat().st_mtime
    return sum(1 for t in times.values() if t > t_model)


def import_model(project, weights, name=None):
    """Copy a trained model (best.pt, ideally from a SeedLabeler run folder) into this crop's models.

    The run's seedlabeler.json (input kind, tiling) is copied too when present;
    otherwise colour input with this project's tiling is assumed.
    """
    import shutil
    import time

    weights = Path(weights)
    run_meta = weights.parent.parent / "seedlabeler.json"
    name = name or f"imported_{time.strftime('%Y%m%d_%H%M%S')}"
    run = project.models_dir / name
    (run / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, run / "weights" / "best.pt")
    if (weights.parent.parent / "results.csv").exists():
        shutil.copy2(weights.parent.parent / "results.csv", run / "results.csv")
    if run_meta.exists():
        meta = json.loads(run_meta.read_text())
        assumed = False
    else:
        work_res, overlap = tiling(project.config.settings.seed_width)
        meta = {"input": "rgb", "imgsz": TILE, "height_scale_mm": project.config.settings.seed_width,
                "work_res_mm": work_res, "tile": TILE, "overlap": overlap}
        assumed = True
    meta["imported_from"] = str(weights)
    (run / "seedlabeler.json").write_text(json.dumps(meta, indent=2))
    return run / "weights" / "best.pt", assumed


def train_command(project, run_name, epochs=150, imgsz=TILE, base=None, kind="rgb"):
    """argv for the training subprocess."""
    data = write_data_yaml(project, kind)
    work_res, overlap = tiling(project.config.settings.seed_width)
    return [sys.executable, "-m", "seedlabeler.train_worker",
            "--data", str(data), "--project", str(project.models_dir), "--name", run_name,
            "--epochs", str(epochs), "--imgsz", str(imgsz), "--base", str(base or BASE_MODEL),
            "--input", kind, "--height-scale", str(project.config.settings.seed_width),
            "--work-res", f"{work_res:.6f}", "--tile", str(TILE), "--overlap", str(overlap)]


_loaded = {}


def model_detector(model_path, conf=0.35, imgsz=None):
    """Detector callable for chickpea.pipeline.measure_pair using a trained model.

    Rebuilds exactly the input the model was trained on (kind, scale, tiles;
    recorded by train_worker). Each tile owns the seeds whose centre falls in
    its core (the tile minus half the overlap), so seeds in overlaps aren't
    counted twice.
    """
    from skimage.segmentation import relabel_sequential
    from ultralytics import YOLO

    meta_file = Path(model_path).parents[1] / "seedlabeler.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    kind, hscale = meta.get("input", "rgb"), meta.get("height_scale_mm", 6.0)
    work_res, tile, overlap = meta.get("work_res_mm"), meta.get("tile"), meta.get("overlap")
    imgsz = imgsz or meta.get("imgsz", tile or 1280)

    key = str(model_path)
    if key not in _loaded:
        _loaded[key] = YOLO(key)
    model = _loaded[key]

    def detect(ortho, valid, res, height):
        img = make_input(ortho, height, valid, kind, hscale)
        found = []  # (confidence, polygon in ortho pixels)
        if work_res and tile == TILE:
            f = res / work_res
            small = _resample(img, f, cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            small_valid = _resample(valid.astype(np.uint8), f, cv2.INTER_NEAREST).astype(bool) if f < 1 else valid
            grid = [pos for pos in tile_grid(small.shape, overlap)
                    if small_valid[pos[1]:pos[1] + TILE, pos[0]:pos[0] + TILE].any()]
            half = overlap // 2
            for start in range(0, len(grid), 16):
                batch = grid[start:start + 16]
                results = model.predict([_crop(small, x, y) for x, y in batch], imgsz=imgsz, conf=conf,
                                        verbose=False)
                for (x, y), r in zip(batch, results):
                    if r.masks is None:
                        continue
                    cx0 = x + (half if x > 0 else 0)
                    cy0 = y + (half if y > 0 else 0)
                    cx1 = x + TILE - (half if x + TILE < sw else 0)
                    cy1 = y + TILE - (half if y + TILE < sh else 0)
                    for poly, c in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
                        if len(poly) < 3:
                            continue
                        p = poly + [x, y]
                        mx, my = p.mean(axis=0)
                        if cx0 <= mx < cx1 and cy0 <= my < cy1:
                            found.append((float(c), p / min(f, 1.0)))
        else:  # models trained on whole images (before tiling)
            r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
            if r.masks is not None:
                found = [(float(c), p) for p, c in zip(r.masks.xy, r.boxes.conf.cpu().numpy()) if len(p) >= 3]
        labels = np.zeros(ortho.shape[:2], np.int32)
        for n, (_, p) in enumerate(sorted(found, key=lambda t: t[0]), start=1):  # most confident wins overlaps
            cv2.fillPoly(labels, [np.round(p).astype(np.int32)], n)
        labels, _, _ = relabel_sequential(labels)
        return labels, {"model": key, "model_conf": conf, "model_input": kind}

    return detect
