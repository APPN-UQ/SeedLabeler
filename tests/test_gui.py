"""GUI behaviour, offscreen: vertex dragging, sample rename/delete."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from seedlabeler import processing  # noqa: E402
from seedlabeler.canvas import Canvas  # noqa: E402
from seedlabeler.project import Project  # noqa: E402

from test_workflow import CALIB, CODE, _pair  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_vertex_drag_moves_only_that_vertex(app):
    c = Canvas()
    c.resize(800, 600)
    c.set_image(np.zeros((400, 600, 3), np.uint8))
    square = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    edits = []
    c.edited.connect(edits.append)
    c.set_seeds([{"polygon": square, "status": "auto"}])
    c.show()
    c.fit()
    c.select_index(0, centre=False)
    app.processEvents()

    vp = c.viewport()
    start = c.mapFromScene(200, 200)  # bottom-right vertex handle
    end = c.mapFromScene(260, 240)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
    for t in np.linspace(0, 1, 6)[1:]:
        p = start + (end - start) * t
        QTest.mouseMove(vp, QPoint(int(p.x()), int(p.y())))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end)
    for _ in range(5):
        app.processEvents()

    assert edits, "no edit was committed"
    poly = np.array(edits[-1][0]["polygon"])
    np.testing.assert_allclose(poly[[0, 1, 3]], np.array(square)[[0, 1, 3]], atol=0.5)  # others untouched
    assert np.linalg.norm(poly[2] - [260, 240]) < 3, poly[2]
    assert edits[-1][0]["status"] == "edited"


def test_rename_and_delete(tmp_path):
    left, right = _pair()
    p = Project.create(tmp_path / "proj", "t", crop="almond", calib=str(CALIB))
    s = p.add_pair(left, right, "plot 12/A")  # barcode-ish ID with characters to sanitise
    assert s.id == "plot_12_A"
    processing.detect(p, s)
    from seedlabeler import ml
    split = ml.add_to_dataset(p, s)

    r = p.rename_sample(s, "QR-000123")
    assert r.id == "QR-000123" and not s.path.exists() and r.processed
    seeds = pd.read_csv(r.seeds_csv)
    assert set(seeds["sample"]) == {"QR-000123"}
    assert list((p.dataset_dir / "labels" / split).glob("QR-000123__t*.txt"))
    assert not list((p.dataset_dir / "labels" / split).glob("plot_12_A*"))

    p.delete_sample(r)
    assert not r.path.exists() and p.samples() == []
    assert not list(p.root.glob("datasets/*/labels/*/QR-000123*"))


def _wait(app, cond, timeout=60):
    import time
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(5):
        app.processEvents()


def test_sam_box_click_and_refine(app, tmp_path):
    """Delete a seed, redraw it with a SAM box, refine with a click, refine all."""
    import cv2
    from seedlabeler.shell import Shell

    left, right = _pair()
    p = Project.create(tmp_path / "proj", "t", crop="almond", calib=str(CALIB))
    p.config.settings.sam_refine = False  # keep the classic outlines so a SAM redraw is visible
    p.save()
    processing.detect(p, p.add_pair(left, right, "s1"))
    shell = Shell()
    shell.open_project(p.root)
    shell.go("label")
    w = shell.label
    n0 = len(w.ann["seeds"])
    target = np.array(w.ann["seeds"][5]["polygon"])
    x0, y0 = target.min(0)
    x1, y1 = target.max(0)

    w.canvas.select_index(5, centre=False)
    w.canvas.delete_selected()
    _wait(app, lambda: len(w.ann["seeds"]) == n0 - 1)
    w._set_mode("box")
    _wait(app, lambda: not shell.tasks, timeout=120)  # model preload
    w.canvas.boxDrawn.emit(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
    _wait(app, lambda: len(w.ann["seeds"]) == n0 and not w._sam_busy)
    new = w.ann["seeds"][-1]
    assert new["status"] == "manual"
    shape = w.layers[0].shape[:2]
    a, b = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillPoly(a, [np.round(target).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(np.array(new["polygon"])).astype(np.int32)], 1)
    assert (a & b).sum() / (a | b).sum() > 0.85

    cx, cy = target.mean(0)
    w.canvas.pointClicked.emit(float(cx), float(cy), True)
    _wait(app, lambda: not w._sam_busy)
    assert len(w.ann["seeds"]) == n0 and w.sam_active["points"]

    w.sam_refine_all()
    _wait(app, lambda: not shell.tasks, timeout=120)
    assert len(w.ann["seeds"]) == n0
    assert any(s.get("sam") for s in w.ann["seeds"])
    w.do_undo()
    assert not any(s.get("sam") for s in w.ann["seeds"])
    shell.close()


def test_hub_workflow_and_batch_process(app, tmp_path):
    """Hub recommends LABEL first; PROCESS batch-detects captured samples."""
    from seedlabeler.home import workflow_state
    from seedlabeler.shell import Shell

    left, right = _pair()
    p = Project.create(tmp_path / "proj", "t", crop="almond", calib=str(CALIB))
    p.config.settings.sam_refine = False
    p.save()
    for i in range(2):
        p.add_pair(left, right, f"cap{i}")  # captured, not processed (rapid-capture style)
    steps, mode, stats = workflow_state(p)
    assert mode == "label" and steps[0][0] == "next" and stats["waiting"] == 2

    shell = Shell()
    shell.open_project(p.root)
    shell.go("process")
    shell.process.process_new()
    _wait(app, lambda: shell.process.worker is None, timeout=300)
    assert all(s.processed for s in p.samples())
    assert len(p.samples()[0].annotations()["seeds"]) == 19
    p.config.settings.label_target = 1
    p.save()
    from seedlabeler import ml
    ml.add_to_dataset(p, p.samples()[0])
    steps, mode, _ = workflow_state(p)
    assert steps[0][0] == "done" and steps[1][0] == "next"  # labelled enough: train next
    shell.close()


def test_capture_locked_until_model_and_retrain_counter(app, tmp_path):
    """CAPTURE stays locked without a model; LOAD MODEL unlocks it and makes it the detector."""
    from seedlabeler import ml
    from seedlabeler.home import workflow_state
    from seedlabeler.shell import Shell

    demo_model = CODE / "SeedLabeler" / "projects" / "almond_demo" / "models" / "almond" / "rgbh_tiles" / "weights" / "best.pt"
    if not demo_model.exists():
        pytest.skip("demo model not available")
    left, right = _pair()
    p = Project.create(tmp_path / "proj", "t", crop="almond", calib=str(CALIB))
    s = p.add_pair(left, right, "a1")
    processing.detect(p, s)
    ml.add_to_dataset(p, s)
    shell = Shell()
    shell.open_project(p.root)
    shell.go("capture")
    assert shell.current != "capture" and not shell.tabs["capture"].isEnabled()

    weights, assumed = ml.import_model(p, demo_model)
    assert not assumed  # the run's settings (input, tiling) came along
    shell.use_model(weights)
    shell.models_changed()
    assert shell.tabs["capture"].isEnabled()
    shell.go("capture")
    assert shell.current == "capture"
    shell.go("hub")
    assert shell.project.config.detector.endswith("weights/best.pt")  # the app's project, saved to disk
    assert Project.open(p.root).config.detector == shell.project.config.detector
    # an imported model has never seen this project's samples: all of them count as new
    assert ml.samples_since_model(p) == 1
    import os, time
    past = time.time() - 3600  # pretend the samples were added before the model was trained
    for f in ml.dataset_root(p).glob("labels/*/*.txt"):
        os.utime(f, (past, past))
    model_file = ml.latest_model(p)["path"]
    os.utime(model_file, (time.time(), time.time()))
    assert ml.samples_since_model(p) == 0
    future = time.time() + 60  # then one is corrected after training
    f0 = next(ml.dataset_root(p).glob("labels/*/*.txt"))
    os.utime(f0, (future, future))
    assert ml.samples_since_model(p) == 1
    steps, _, stats = workflow_state(p)
    assert stats["has_model"] and stats["new_since_model"] == 1
    shell.close()
