"""Look and feel: black, white type, one HAL-red accent, one data blue. After 2001's displays."""
import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

BLACK = "#000000"
PANEL = "#060606"
LINE = "#262626"
TEXT = "#e8e8e8"
DIM = "#6f6f6f"
RED = "#ff2b1c"
BLUE = "#59b0ff"

SANS = "Century Gothic, Futura, Segoe UI"
MONO = "Consolas, Cascadia Mono, monospace"

# Instance colours: a twelve-colour swatch after the Discovery One's displays (HAL red,
# instrument amber, readout blues and greens), ordered so neighbours alternate warm/cool.
PALETTE = ["#ff3b30", "#5ac8fa", "#ffb000", "#af52de", "#a3e635", "#ff6b9a",
           "#00c7be", "#ff7a1a", "#3d7bff", "#ffd60a", "#e040fb", "#6ee7b7"]


def palette(i):
    return PALETTE[i % len(PALETTE)]


def palette_bgr(i):
    h = palette(i).lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return (b, g, r)

QSS = f"""
* {{
    background: {BLACK}; color: {TEXT};
    font-family: {SANS}; font-size: 11px;
    selection-background-color: {RED}; selection-color: {BLACK};
}}
QMainWindow::separator {{ background: {LINE}; width: 1px; height: 1px; }}
QLabel#title {{ font-size: 13px; letter-spacing: 6px; color: {TEXT}; }}
QLabel#dim, QLabel#caption {{ color: {DIM}; letter-spacing: 3px; font-size: 9px; }}
QLabel#readout {{ font-family: {MONO}; font-size: 11px; color: {TEXT}; }}
QPushButton, QToolButton {{
    background: transparent; border: none; color: {DIM};
    letter-spacing: 3px; font-size: 10px; padding: 6px 10px;
}}
QPushButton:hover, QToolButton:hover {{ color: {TEXT}; }}
QPushButton:checked, QToolButton:checked {{ color: {RED}; }}
QPushButton:disabled {{ color: #2f2f2f; }}
QPushButton#primary {{ color: {TEXT}; border: 1px solid {LINE}; }}
QPushButton#primary:hover {{ border-color: {TEXT}; }}
QListWidget, QTableWidget, QTableView, QPlainTextEdit, QTextEdit {{
    border: none; outline: none; background: {BLACK};
    font-family: {MONO}; font-size: 11px;
}}
QListWidget::item {{ padding: 6px 4px; color: {DIM}; border-bottom: 1px solid #111; }}
QListWidget::item:selected {{ color: {TEXT}; background: {BLACK}; border-left: 2px solid {RED}; }}
QListWidget::item:hover {{ color: {TEXT}; }}
QHeaderView::section {{
    background: {BLACK}; color: {DIM}; border: none; border-bottom: 1px solid {LINE};
    letter-spacing: 2px; font-size: 9px; padding: 4px;
}}
QTableView {{ gridline-color: #111; }}
QTableView::item:selected {{ color: {BLACK}; background: {RED}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BLACK}; border: none; border-bottom: 1px solid {LINE};
    padding: 4px 2px; font-family: {MONO};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-bottom: 1px solid {TEXT}; }}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox QAbstractItemView {{ border: 1px solid {LINE}; selection-background-color: {RED}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {BLACK}; width: 6px; height: 6px; }}
QScrollBar::handle {{ background: {LINE}; min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QDialog {{ border: 1px solid {LINE}; }}
QToolTip {{ background: {BLACK}; color: {TEXT}; border: 1px solid {LINE}; }}
QProgressBar {{ border: none; background: #111; height: 2px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {RED}; }}
QMenu {{ border: 1px solid {LINE}; }}
QMenu::item:selected {{ background: {BLACK}; color: {RED}; }}
"""


def apply(app):
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    f = QFont("Century Gothic")
    f.setStyleHint(QFont.SansSerif)
    app.setFont(f)


class HalEye(QWidget):
    """A small red lens. Dim when idle, glowing and breathing while busy."""

    def __init__(self, size=22, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._busy = False
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_busy(self, busy):
        self._busy = busy
        if busy:
            self._timer.start(40)
        else:
            self._timer.stop()
        self.update()

    def _tick(self):
        self._t += 0.08
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.width() / 2
        c = QPointF(r, r)
        glow = 0.55 + 0.45 * math.sin(self._t) if self._busy else 0.28
        # outer ring
        p.setPen(QColor(LINE))
        p.setBrush(QColor(BLACK))
        p.drawEllipse(c, r - 1, r - 1)
        # lens
        g = QRadialGradient(c, r * 0.8)
        g.setColorAt(0.0, QColor(255, 210, 150, int(255 * glow)))
        g.setColorAt(0.18, QColor(255, 43, 28, int(255 * glow)))
        g.setColorAt(0.65, QColor(110, 0, 0, int(200 * glow)))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawEllipse(c, r * 0.8, r * 0.8)
