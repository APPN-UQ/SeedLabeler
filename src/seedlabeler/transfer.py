"""Move work between PCs: export/import a whole project as a zip, and import captured samples.

Typical use: capture on a slow laptop, then label, train and process on a faster
PC. Either export the laptop's project (EXPORT on the hub) and import it on the
other PC (IMPORT PROJECT on the home page), or add the laptop's sample folders to
an existing project (IMPORT SAMPLES on the hub), e.g. one that already has a
trained model.
"""
import filecmp
import json
import shutil
import zipfile
from pathlib import Path

from .project import Project

# already-compressed files go into the zip as they are (deflating them only costs time)
_STORED = {".jpg", ".jpeg", ".png", ".npz", ".pt", ".zip", ".tif", ".tiff"}


# --------------------------------------------------------------------------- whole projects
def export_project(project, dest_zip, models=True):
    """Zip the project folder (samples, labels, calibration, datasets and, by default, models).

    The zip holds one top-level folder named after the project folder, so
    importing it recreates the project as it was. Returns (files, bytes).
    """
    root = project.root
    dest_zip = Path(dest_zip)
    if dest_zip.resolve().is_relative_to(root):
        raise ValueError("Save the export outside the project folder")
    n = size = 0
    tmp = dest_zip.with_name(dest_zip.name + ".part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root)
            if not models and rel.parts[0] == "models":
                continue
            comp = zipfile.ZIP_STORED if f.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED
            z.write(f, (Path(root.name) / rel).as_posix(), compress_type=comp)
            n += 1
            size += f.stat().st_size
    tmp.replace(dest_zip)
    return n, size


def import_project(src_zip, dest_parent):
    """Unpack an exported project into dest_parent; returns the new project folder.

    Refuses to overwrite an existing folder and to write outside dest_parent.
    """
    dest_parent = Path(dest_parent).resolve()
    with zipfile.ZipFile(src_zip) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1 or f"{next(iter(tops))}/project.json" not in names:
            raise ValueError("Not a SeedLabeler project export (no project.json in a single top folder)")
        top = next(iter(tops))
        target = dest_parent / top
        if target.exists():
            raise FileExistsError(f"{target} already exists: pick another place, or rename or move that folder")
        for n in names:
            out = (dest_parent / n).resolve()
            if not out.is_relative_to(target):
                raise ValueError(f"Unsafe path in the zip: {n}")
        z.extractall(dest_parent, members=names)
    Project.open(target)  # check it opens
    return target


# --------------------------------------------------------------------------- samples
def find_samples(src):
    """Sample folders under src: <id>/raw/left.jpg + right.jpg (or rgb.jpg + depth.npz), or the
    images directly in <id>/. Returns [(id, raw_dir)], sorted by path."""
    src = Path(src)
    found = []
    for left in sorted(src.rglob("left.jpg")):
        d = left.parent
        if (d / "right.jpg").exists():
            found.append(d)
    for rgb in sorted(src.rglob("rgb.jpg")):
        d = rgb.parent
        if (d / "depth.npz").exists():
            found.append(d)
    out = []
    for d in sorted(set(found)):
        if "calibration" in d.relative_to(src).parts:  # colourcal_raw/ etc. are not samples
            continue
        sid = d.parent.name if d.name == "raw" else d.name
        out.append((sid, d))
    return out


def _nearest(start, stop, rel):
    """rel (e.g. calibration/colourcal.json) in start or the closest folder above it, up to stop."""
    d = start
    while True:
        if (d / rel).exists():
            return d / rel
        if d == stop or d.parent == d:
            return None
        d = d.parent


def import_samples(project, src, progress=None):
    """Copy every sample found under src into the project (raw images only: detect them here).

    Each sample keeps the colour calibration it was captured with: the closest
    calibration/colourcal.json above it (or in it) is stored as raw/colourcal.json,
    which detection then uses instead of this project's. Sample ids that already
    exist get a _2, _3 ... suffix. Returns a summary dict.
    """
    src = Path(src).resolve()
    if src == project.root or src.is_relative_to(project.root):
        raise ValueError("That folder is inside this project")
    samples = find_samples(src)
    if not samples:
        raise ValueError("No samples found (looked for raw/left.jpg + right.jpg, or rgb.jpg + depth.npz)")
    imported, renamed, colour = [], [], 0
    stereo_differs = None
    for i, (sid, raw) in enumerate(samples):
        new_id = project.new_sample_id(sid)
        if new_id != sid:
            renamed.append(f"{sid} -> {new_id}")
        dest = project.root / "samples" / new_id / "raw"
        dest.mkdir(parents=True)
        for f in raw.iterdir():
            if f.is_file() and f.name in ("left.jpg", "right.jpg", "rgb.jpg", "depth.npz", "colourcal.json"):
                shutil.copy2(f, dest / f.name)
        cal = (raw / "colourcal.json") if (raw / "colourcal.json").exists() else \
            _nearest(raw, src, Path("calibration") / "colourcal.json")
        if cal is not None and not (dest / "colourcal.json").exists():
            shutil.copy2(cal, dest / "colourcal.json")
        if (dest / "colourcal.json").exists():
            colour += 1
        calib = _nearest(raw, src, Path("calibration") / "calib.npz")
        if calib is not None and project.config.calib and stereo_differs is None:
            mine = project.path(project.config.calib)
            stereo_differs = mine.exists() and not filecmp.cmp(calib, mine, shallow=False)
        imported.append(new_id)
        if progress:
            progress(i + 1, len(samples))
    log = project.root / "imports.json"
    history = json.loads(log.read_text()) if log.exists() else []
    history.append({"from": str(src), "samples": imported})
    log.write_text(json.dumps(history, indent=2))
    return {"imported": imported, "renamed": renamed, "with_colourcal": colour,
            "stereo_calib_differs": bool(stereo_differs)}
