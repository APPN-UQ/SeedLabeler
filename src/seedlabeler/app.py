"""SeedLabeler entry point: `seedlabeler [project_dir]` or `python -m seedlabeler`."""
import sys

from PySide6.QtWidgets import QApplication

from . import style
from .shell import Shell
from .updater import check_for_update


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SeedLabeler")
    style.apply(app)
    w = Shell()
    if len(sys.argv) > 1:
        w.open_project(sys.argv[1])
    w.show()
    check_for_update(w)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
