"""Moving work between PCs: import captured samples, export/import a whole project."""
import json

import cv2
import numpy as np
import pytest

from seedlabeler import transfer
from seedlabeler.project import Project


def _capture_folder(root):
    """A capture laptop's folder: calibration/ + batch folders of <id>/raw/left.jpg, right.jpg."""
    img = np.full((40, 60, 3), 128, np.uint8)
    (root / "calibration" / "colourcal_raw").mkdir(parents=True)
    cv2.imwrite(str(root / "calibration" / "colourcal_raw" / "left.jpg"), img)  # not a sample
    cv2.imwrite(str(root / "calibration" / "colourcal_raw" / "right.jpg"), img)
    (root / "calibration" / "colourcal.json").write_text(json.dumps({"session": "laptop"}))
    for batch, ids in (("B1", ["S1", "S2"]), ("B2", ["S3"])):
        for sid in ids:
            raw = root / batch / sid / "raw"
            raw.mkdir(parents=True)
            cv2.imwrite(str(raw / "left.jpg"), img)
            cv2.imwrite(str(raw / "right.jpg"), img)
    (root / "B2" / "empty").mkdir()
    return root


def test_import_samples(tmp_path):
    src = _capture_folder(tmp_path / "laptop")
    p = Project.create(tmp_path / "proj", "p")
    (p.root / "calibration").mkdir(exist_ok=True)
    (p.root / "calibration" / "colourcal.json").write_text(json.dumps({"session": "desktop"}))
    (p.root / "samples" / "S1" / "raw").mkdir(parents=True)  # id already taken
    assert [s for s, _ in transfer.find_samples(src)] == ["S1", "S2", "S3"]

    r = transfer.import_samples(p, src)
    assert sorted(r["imported"]) == ["S1_2", "S2", "S3"] and r["renamed"] == ["S1 -> S1_2"]
    assert r["with_colourcal"] == 3
    s = next(x for x in p.samples() if x.id == "S2")
    assert s.status() == "new" and s.needs_detection
    # the capture-time colour calibration travels with the sample
    assert json.loads((s.path / "raw" / "colourcal.json").read_text()) == {"session": "laptop"}
    assert json.loads((p.root / "imports.json").read_text())[0]["samples"] == r["imported"]

    with pytest.raises(ValueError):
        transfer.import_samples(p, p.root)  # not from inside itself
    with pytest.raises(ValueError):
        transfer.import_samples(p, tmp_path / "laptop" / "B2" / "empty")


def test_prepared_sample_counts_as_new(tmp_path):
    p = Project.create(tmp_path / "proj", "p")
    s_dir = p.root / "samples" / "A"
    (s_dir / "raw").mkdir(parents=True)
    cv2.imwrite(str(s_dir / "raw" / "left.jpg"), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(s_dir / "ortho.png"), np.zeros((4, 4, 3), np.uint8))
    s = p.samples()[0]
    s.save_annotations({"seeds": [], "auto": [], "source": "prepared", "reviewed": False, "in_dataset": False})
    assert s.processed and s.needs_detection and s.status() == "new"
    s.save_annotations({"seeds": [{"polygon": [[0, 0], [2, 0], [2, 2]], "status": "manual"}], "auto": [],
                        "source": "prepared", "reviewed": False, "in_dataset": False})
    assert not s.needs_detection and s.status() == "detected"


def test_export_import_round_trip(tmp_path):
    src = _capture_folder(tmp_path / "laptop")
    p = Project.create(tmp_path / "proj", "p")
    transfer.import_samples(p, src)
    run = p.models_dir / "run1" / "weights"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"weights")

    with pytest.raises(ValueError):
        transfer.export_project(p, p.root / "inside.zip")
    transfer.export_project(p, tmp_path / "a.zip")
    transfer.export_project(p, tmp_path / "b.zip", models=False)

    q = Project.open(transfer.import_project(tmp_path / "a.zip", tmp_path / "pc2"))
    assert len(q.samples()) == 3 and (q.models_dir / "run1" / "weights" / "best.pt").exists()
    with pytest.raises(FileExistsError):
        transfer.import_project(tmp_path / "a.zip", tmp_path / "pc2")
    q2 = transfer.import_project(tmp_path / "b.zip", tmp_path / "pc3")
    assert not (q2 / "models").exists()
