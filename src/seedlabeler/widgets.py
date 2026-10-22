"""Small shared building blocks for the pages."""
import traceback

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from . import style


class Task(QThread):
    """Run a function off the GUI thread."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kw))
        except Exception as e:  # noqa: BLE001 - surfaced in the GUI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


def caption(text):
    lab = QLabel(text.upper())
    lab.setObjectName("caption")
    return lab


def label(text, name=None, wrap=False):
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def button(text, slot=None, checkable=False, primary=False, secondary=False, tip=None, name=None):
    b = QPushButton(text.upper())
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    if primary:
        b.setObjectName("primary")
    elif secondary:
        b.setObjectName("secondary")
    elif name:
        b.setObjectName(name)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def hline():
    w = QWidget()
    w.setFixedHeight(1)
    w.setStyleSheet(f"background: {style.LINE};")
    return w


def vline():
    w = QWidget()
    w.setFixedWidth(1)
    w.setStyleSheet(f"background: {style.LINE};")
    return w


def card(*widgets, margins=16, spacing=8):
    f = QFrame()
    f.setObjectName("card")
    v = QVBoxLayout(f)
    v.setContentsMargins(margins, margins, margins, margins)
    v.setSpacing(spacing)
    for w in widgets:
        v.addWidget(w)
    return f, v
