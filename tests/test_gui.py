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

from test_workflow import CALIB, _pair  # noqa: E402


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
