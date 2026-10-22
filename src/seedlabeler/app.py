"""SeedLabeler entry point: `seedlabeler [project_dir]` or `python -m seedlabeler`."""
import sys

from PySide6.QtWidgets import QApplication

from . import style
from .shell import Shell


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SeedLabeler")
    style.apply(app)
    w = Shell()
    if len(sys.argv) > 1:
        w.open_project(sys.argv[1])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
