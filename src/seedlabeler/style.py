"""Look and feel: a light lab instrument. Warm off-white, charcoal type, one muted slate-blue accent."""
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

BG = "#f4f3ef"        # warm paper
SURFACE = "#ffffff"   # cards, tables, inputs
CANVAS = "#e9e7e1"    # behind images
LINE = "#dedbd3"      # hairlines
TEXT = "#22262b"      # charcoal
DIM = "#82868c"
ACCENT = "#2f5f86"    # slate blue: active, selected, primary
ACCENT_SOFT = "#e2eaf1"
WARN = "#b7791f"      # amber, warnings only
OK = "#2e7d32"        # green: camera connected
BAD = "#c0392b"       # red: camera not connected
GUIDE = "#e05530"     # warm red-orange: depth zone, calibration guides

SANS = "Segoe UI, Helvetica Neue, Arial"
MONO = "Consolas, Cascadia Mono, monospace"

# Instance colours: twelve mid-tone hues that read on white boards and beige or brown seeds,
# ordered so neighbouring seeds alternate warm/cool.
PALETTE = ["#e4572e", "#2e86ab", "#f2a541", "#7b4ea3", "#3aa76d", "#d64f8f",
           "#1b998b", "#f17105", "#3d5a80", "#c9a227", "#a23b72", "#4f9d69"]


def palette(i):
    return PALETTE[i % len(PALETTE)]


def palette_bgr(i):
    h = palette(i).lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return (b, g, r)


QSS = f"""
* {{
    color: {TEXT}; font-family: {SANS}; font-size: 12px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QMainWindow, QDialog, QStackedWidget, QWidget#page {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QLabel#wordmark {{ font-size: 13px; letter-spacing: 5px; font-weight: 600; }}
QLabel#hero {{ font-size: 30px; letter-spacing: 8px; font-weight: 300; }}
QLabel#h2 {{ font-size: 18px; font-weight: 400; }}
QLabel#caption {{ color: {DIM}; letter-spacing: 2px; font-size: 10px; }}
QLabel#dim {{ color: {DIM}; }}
QLabel#readout {{ font-family: {MONO}; font-size: 12px; }}
QLabel#big {{ font-size: 28px; font-weight: 300; }}
QLabel#warn {{ color: {WARN}; }}
QPushButton {{
    background: transparent; border: none; color: {DIM};
    letter-spacing: 2px; font-size: 11px; padding: 7px 10px;
}}
QPushButton:hover {{ color: {TEXT}; }}
QPushButton:checked {{ color: {ACCENT}; font-weight: 600; }}
QPushButton:disabled {{ color: #c4c1b9; }}
QPushButton#primary {{
    background: {ACCENT}; color: white; border-radius: 3px; padding: 8px 16px; font-weight: 600;
}}
QPushButton#primary:hover {{ background: #24496a; }}
QPushButton#primary:disabled {{ background: #b9c6d2; color: #f4f6f8; }}
QPushButton#secondary {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 3px; padding: 8px 14px;
}}
QPushButton#secondary:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#tab {{ padding: 6px 12px; border-bottom: 2px solid transparent; }}
QPushButton#tab:checked {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton#tile {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px;
    padding: 22px; text-align: left; color: {TEXT}; letter-spacing: 0px; font-size: 13px;
}}
QPushButton#tile:hover {{ border-color: {ACCENT}; }}
QPushButton#tile[recommended="true"] {{ border: 2px solid {ACCENT}; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; }}
QListWidget, QTableWidget, QTableView, QPlainTextEdit, QTextEdit {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 4px; outline: none;
    font-family: {MONO}; font-size: 12px;
}}
QListWidget::item {{ padding: 6px 6px; color: {TEXT}; border-bottom: 1px solid #f0eee9; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; border-left: 3px solid {ACCENT}; }}
QListWidget::item:hover {{ background: #f7f6f2; }}
QHeaderView::section {{
    background: {SURFACE}; color: {DIM}; border: none; border-bottom: 1px solid {LINE};
    letter-spacing: 1px; font-size: 10px; padding: 5px;
}}
QTableView {{ gridline-color: #f0eee9; }}
QTableView::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 3px; padding: 6px 8px;
    font-family: {MONO};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QLineEdit#scan {{ font-size: 22px; padding: 10px 12px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {LINE}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 8px; height: 8px; }}
QScrollBar::handle {{ background: #cfccc4; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px; }}
QProgressBar {{ border: none; background: {LINE}; height: 4px; border-radius: 2px; text-align: center;
               color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QMenu {{ background: {SURFACE}; border: 1px solid {LINE}; }}
QMenu::item {{ padding: 6px 18px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QCheckBox {{ spacing: 8px; }}
"""


def apply(app):
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    f = QFont("Segoe UI")
    f.setStyleHint(QFont.SansSerif)
    f.setPointSizeF(9.5)
    app.setFont(f)


class BusyLine(QWidget):
    """A thin line under the top bar: a slate segment sweeps across while work runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(2)
        self._busy = False
        self._x = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_busy(self, busy):
        if busy == self._busy:
            return
        self._busy = busy
        self._timer.start(16) if busy else self._timer.stop()
        self.update()

    def _tick(self):
        self._x = (self._x + 0.012) % 1.3
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(LINE))
        if self._busy:
            w = self.width()
            seg = 0.3 * w
            x = (self._x - 0.3) * w
            p.fillRect(int(x), 0, int(seg), self.height(), QColor(ACCENT))
