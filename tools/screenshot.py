"""Render the main window offscreen to a PNG (for docs and visual checks).

    python tools/screenshot.py <project_dir> out.png [--select N]
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from seedlabeler import style  # noqa: E402
from seedlabeler.app import MainWindow  # noqa: E402


def main():
    project, out = sys.argv[1], sys.argv[2]
    select = int(sys.argv[sys.argv.index("--select") + 1]) if "--select" in sys.argv else None
    app = QApplication(sys.argv[:1])
    style.apply(app)
    w = MainWindow()
    w.open_project(project)
    w.resize(1600, 960)
    w.show()
    for _ in range(20):
        app.processEvents()
    w.canvas.fit()
    if select is not None:
        w.canvas.select_index(select, centre=False)
    for _ in range(20):
        app.processEvents()
    w.grab().save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
