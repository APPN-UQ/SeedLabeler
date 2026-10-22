"""Projects and samples on disk.

<project>/
    project.json              name, crop profile, calibration, detector, settings
    rig.json                  which camera serial is left/right
    samples/<id>/
        raw/left.jpg, raw/right.jpg
        ortho.png             top-down stage image (written by processing)
        height.npz            height above stage (mm, float16)
        meta.json             grid resolution/offset, stage distance, detector...
        annotations.json      seed polygons (+ the untouched automatic ones)
        <id>_seeds.csv, <id>_summary.csv
    datasets/<crop>/          training data, one dataset per crop
    models/<crop>/<run>/      trained models
    exports/                  project-wide trait tables
"""
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CROP_PROFILES = {
    # smallest expected seed width (for splitting) and minimum height above the stage
    "chickpea": {"seed_width": 6.0, "min_thickness": 1.5},
    "almond": {"seed_width": 10.0, "min_thickness": 2.0},
    "wheat": {"seed_width": 2.5, "min_thickness": 1.0},
    "custom": {"seed_width": 4.0, "min_thickness": 1.0},
}


@dataclass
class Settings:
    seed_width: float = 6.0
    min_thickness: float = 1.5
    min_area: float = 4.0
    min_depth: float = 200.0   # stereo search range (mm), used when there is no stage file
    max_depth: float = 800.0
    scale: float = 0.5         # stereo matching scale
    model_conf: float = 0.35
    sam_model: str = "mobile_sam.pt"
    sam_refine: bool = True    # detection = find seeds (classic or model), then SAM redraws every outline
    label_target: int = 20     # samples to label before training (hub's suggested workflow)


@dataclass
class ProjectConfig:
    name: str
    crop: str = "chickpea"
    calib: str = ""
    stage: str = ""
    detector: str = "classic"  # "classic" or a model path relative to the project
    camera_args: str = ""      # e.g. "--mode m --shutter 1/125 --iso 400"
    settings: Settings = field(default_factory=Settings)


class Project:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = config

    # --- persistence ---------------------------------------------------
    @classmethod
    def create(cls, root, name, crop="chickpea", calib="", stage=""):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        prof = CROP_PROFILES.get(crop, CROP_PROFILES["custom"])
        cfg = ProjectConfig(name=name, crop=crop, calib=calib, stage=stage,
                            settings=Settings(seed_width=prof["seed_width"], min_thickness=prof["min_thickness"]))
        p = cls(root, cfg)
        (root / "samples").mkdir(exist_ok=True)
        p.adopt_files()
        p.save()
        return p

    @classmethod
    def open(cls, root):
        root = Path(root)
        d = json.loads((root / "project.json").read_text())
        known = Settings.__dataclass_fields__
        d["settings"] = Settings(**{k: v for k, v in d.get("settings", {}).items() if k in known})
        p = cls(root, ProjectConfig(**d))
        if p.adopt_files():
            p.save()
        return p

    def adopt_files(self):
        """Copy calibration, its rig.json and the stage file into the project; store relative paths.

        A project then opens on any PC wherever the folder lands (e.g. another
        machine's OneDrive path). Returns True if anything changed.
        """
        changed = False
        dest = self.root / "calibration"
        for attr in ("calib", "stage"):
            val = getattr(self.config, attr)
            if not val:
                continue
            src = Path(val)
            if not src.is_absolute():
                continue  # already inside the project
            try:
                rel = src.resolve().relative_to(self.root.resolve())
                setattr(self.config, attr, rel.as_posix())
                changed = True
                continue
            except ValueError:
                pass
            if not src.exists():
                continue  # e.g. the old absolute path from another PC: keep it, nothing to copy
            dest.mkdir(exist_ok=True)
            target = dest / ("calib.npz" if attr == "calib" else "stage.npz")
            shutil.copy2(src, target)
            if attr == "calib" and (src.with_name("rig.json")).exists():
                shutil.copy2(src.with_name("rig.json"), dest / "rig.json")
            setattr(self.config, attr, target.relative_to(self.root).as_posix())
            changed = True
        return changed

    def save(self):
        (self.root / "project.json").write_text(json.dumps(asdict(self.config), indent=2))

    # --- paths -----------------------------------------------------------
    def path(self, rel):
        """Project-relative or absolute path."""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def rig_file(self):
        """Which camera serial is left/right. It belongs with the calibration (the
        calibration is only valid for that assignment), so a rig.json next to
        calib.npz wins; otherwise the project keeps its own."""
        if self.config.calib:
            beside = self.path(self.config.calib).with_name("rig.json")
            if beside.exists():
                return beside
        return self.root / "calibration" / "rig.json" if (self.root / "calibration").exists()             else self.root / "rig.json"

    @property
    def dataset_dir(self):
        return self.root / "datasets" / self.config.crop

    @property
    def models_dir(self):
        return self.root / "models" / self.config.crop

    # --- samples -------------------------------------------------------
    def samples(self):
        d = self.root / "samples"
        if not d.exists():
            return []
        return [Sample(p) for p in sorted(d.iterdir()) if (p / "raw" / "left.jpg").exists()]

    def new_sample_id(self, stem=None):
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", stem) if stem else time.strftime("%Y%m%d_%H%M%S")
        sid, n = base, 1
        while (self.root / "samples" / sid).exists():
            n += 1
            sid = f"{base}_{n}"
        return sid

    def delete_sample(self, sample):
        """Remove a sample folder and its images/labels from every crop's dataset."""
        for f in self._dataset_files(sample.id):
            f.unlink()
        _rmtree_retry(sample.path)

    def _dataset_files(self, sample_id):
        """A sample's images/labels in every crop's datasets (whole-image or tiles)."""
        ds = self.root / "datasets"
        if not ds.exists():
            return []
        out = []
        for sub in ("images", "labels"):
            out += list(ds.glob(f"*/{sub}/*/{sample_id}.*")) + list(ds.glob(f"*/{sub}/*/{sample_id}__t*"))
        return out

    def rename_sample(self, sample, new_id):
        """Rename a sample (e.g. to a scanned barcode); keeps CSVs and dataset entries consistent."""
        new_id = self.new_sample_id(new_id)
        new_path = self.root / "samples" / new_id
        sample.path.rename(new_path)
        renamed = Sample(new_path)
        for old_name, new_name in ((f"{sample.id}_seeds.csv", renamed.seeds_csv.name),
                                   (f"{sample.id}_summary.csv", renamed.summary_csv.name)):
            f = new_path / old_name
            if f.exists():
                text = f.read_text(encoding="utf-8").splitlines()
                # the first column is the sample id
                text = [text[0]] + [new_id + line[len(sample.id):] if line.startswith(sample.id) else line
                                    for line in text[1:]]
                (new_path / new_name).write_text("\n".join(text) + "\n", encoding="utf-8")
                f.unlink()
        for f in self._dataset_files(sample.id):
            f.rename(f.with_name(new_id + f.name[len(sample.id):]))
        return renamed

    def add_pair(self, left, right, sample_id=None):
        sid = self.new_sample_id(sample_id or Path(left).stem)
        raw = self.root / "samples" / sid / "raw"
        raw.mkdir(parents=True)
        shutil.copy2(left, raw / "left.jpg")
        shutil.copy2(right, raw / "right.jpg")
        return Sample(raw.parent)


def _rmtree_retry(path, tries=10, wait=0.5):
    """rmtree that waits out short locks (OneDrive uploading new files, a scan)."""
    last = None
    for _ in range(tries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as e:
            last = e
            time.sleep(wait)
    raise PermissionError(f"Could not delete {path}: a file is in use ({last}). "
                          "Close it (Excel? OneDrive syncing?) and try again.")


class Sample:
    def __init__(self, path):
        self.path = Path(path)
        self.id = self.path.name

    left = property(lambda self: self.path / "raw" / "left.jpg")
    right = property(lambda self: self.path / "raw" / "right.jpg")
    ortho_file = property(lambda self: self.path / "ortho.png")
    height_file = property(lambda self: self.path / "height.npz")
    meta_file = property(lambda self: self.path / "meta.json")
    ann_file = property(lambda self: self.path / "annotations.json")
    seeds_csv = property(lambda self: self.path / f"{self.id}_seeds.csv")
    summary_csv = property(lambda self: self.path / f"{self.id}_summary.csv")

    @property
    def processed(self):
        return self.ann_file.exists() and self.ortho_file.exists()

    def meta(self):
        return json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}

    def annotations(self):
        if not self.ann_file.exists():
            return {"seeds": [], "auto": [], "reviewed": False, "in_dataset": False}
        return json.loads(self.ann_file.read_text())

    def save_annotations(self, ann):
        self.ann_file.write_text(json.dumps(ann))

    def status(self):
        if not self.processed:
            return "new"
        a = self.annotations()
        if a.get("in_dataset"):
            return "dataset"
        return "reviewed" if a.get("reviewed") else "detected"
