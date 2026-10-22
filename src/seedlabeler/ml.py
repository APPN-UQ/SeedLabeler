"""Instance-segmentation datasets and models (one of each per crop).

Datasets are in YOLO segmentation format, built from reviewed samples'
top-down images and corrected polygons. Training runs in a separate process
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


# --- dataset ---------------------------------------------------------------
def _split(sample_id, val_fraction=0.2):
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def add_to_dataset(project, sample):
    """Write (or refresh) one sample's image and polygons into the crop's dataset."""
    ann = sample.annotations()
    ortho, _, valid = load_layers(sample)
    root = project.dataset_dir
    split = _split(sample.id)
    for other in ("train", "val"):  # a sample lives in exactly one split
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            f = root / sub / other / f"{sample.id}{ext}"
            if f.exists() and other != split:
                f.unlink()
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    img = ortho.copy()
    img[~valid] = 0
    cv2.imwrite(str(root / "images" / split / f"{sample.id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = ortho.shape[:2]
    lines = []
    for s in ann["seeds"]:
        p = np.asarray(s["polygon"], float)
        p[:, 0] /= w
        p[:, 1] /= h
        lines.append("0 " + " ".join(f"{v:.6f}" for v in p.clip(0, 1).ravel()))
    (root / "labels" / split / f"{sample.id}.txt").write_text("\n".join(lines))
    write_data_yaml(project)
    ann["in_dataset"] = True
    ann["reviewed"] = True
    sample.save_annotations(ann)
    return split


def remove_from_dataset(project, sample):
    for split in ("train", "val"):
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            f = project.dataset_dir / sub / split / f"{sample.id}{ext}"
            if f.exists():
                f.unlink()
    ann = sample.annotations()
    ann["in_dataset"] = False
    sample.save_annotations(ann)


def dataset_counts(project):
    out = {}
    for split in ("train", "val"):
        d = project.dataset_dir / "labels" / split
        files = list(d.glob("*.txt")) if d.exists() else []
        out[split] = (len(files), sum(len(f.read_text().splitlines()) for f in files))
    return out


def write_data_yaml(project):
    root = project.dataset_dir
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
        info = {"name": run.name, "path": best, "mask_map50_95": None, "mask_map50": None, "epochs": None}
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


def train_command(project, run_name, epochs=100, imgsz=1280, base=None):
    """argv for the training subprocess."""
    data = write_data_yaml(project)
    return [sys.executable, "-m", "seedlabeler.train_worker",
            "--data", str(data), "--project", str(project.models_dir), "--name", run_name,
            "--epochs", str(epochs), "--imgsz", str(imgsz), "--base", str(base or BASE_MODEL)]


_loaded = {}


def model_detector(model_path, conf=0.35, imgsz=None):
    """Detector callable for chickpea.pipeline.measure_pair using a trained model.

    imgsz defaults to the size the model was trained at (recorded by train_worker).
    """
    from ultralytics import YOLO

    if imgsz is None:
        meta = Path(model_path).parents[1] / "seedlabeler.json"
        imgsz = json.loads(meta.read_text()).get("imgsz", 1280) if meta.exists() else 1280

    key = str(model_path)
    if key not in _loaded:
        _loaded[key] = YOLO(key)
    model = _loaded[key]

    def detect(ortho, valid, res, height):
        img = ortho.copy()
        img[~valid] = 0
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        labels = np.zeros(ortho.shape[:2], np.int32)
        if r.masks is not None:
            order = np.argsort(r.boxes.conf.cpu().numpy())  # most confident drawn last (wins overlaps)
            for n, i in enumerate(order, start=1):
                poly = r.masks.xy[i]
                if len(poly) >= 3:
                    cv2.fillPoly(labels, [np.round(poly).astype(np.int32)], n)
        from skimage.segmentation import relabel_sequential
        labels, _, _ = relabel_sequential(labels)
        return labels, {"model": key, "model_conf": conf}

    return detect
