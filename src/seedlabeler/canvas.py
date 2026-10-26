"""Polygon editor on the top-down stage image.

Select: click a seed. Drag it to move; drag a white square to move a vertex;
double-click an edge to insert a vertex; right-click a vertex to remove it;
Delete removes the seed. Box (SAM): drag a box or click a seed. Wheel zooms,
middle drag or Space+drag pans, F fits the view. Each seed is filled with its
own palette colour.
"""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView)

from .style import ACCENT, CANVAS, GUIDE, SURFACE, TEXT, palette


def to_qimage(bgr):
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _pen(colour, width=1.0):
    p = QPen(QColor(colour), width)
    p.setCosmetic(True)  # constant on-screen width at any zoom
    return p


class VertexHandle(QGraphicsRectItem):
    """Drags only its own vertex.

    It handles the mouse itself rather than being a Qt-movable item: with
    Qt's built-in dragging, every selected movable item moves together, which
    dragged the whole (selected) seed along with the vertex.
    """
    SIZE = 6

    def __init__(self, seed_item, index):
        s = self.SIZE
        super().__init__(-s / 2, -s / 2, s, s, seed_item)
        self.seed_item, self.index = seed_item, index
        self.setFlags(QGraphicsItem.ItemIgnoresTransformations)
        self.setPen(_pen(TEXT))
        self.setBrush(QBrush(QColor(SURFACE)))
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(10)
        self._dragging = False

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.seed_item.remove_vertex(self.index)
        elif e.button() == Qt.LeftButton:
            self._dragging = True
        e.accept()  # never pass the press on: the seed must not start a drag

    def mouseMoveEvent(self, e):
        if self._dragging:
            pos = self.seed_item.mapFromScene(e.scenePos())
            self.setPos(pos)
            self.seed_item.move_vertex(self.index, pos)
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._dragging:
            self._dragging = False
            self.seed_item.canvas.commit(self.seed_item)
        e.accept()


class SeedItem(QGraphicsPolygonItem):
    def __init__(self, canvas, polygon, status, number):
        super().__init__(QPolygonF([QPointF(x, y) for x, y in polygon]))
        self.canvas, self.status = canvas, status
        self.handles = []
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.colour = QColor(palette(number - 1))
        self.label = QGraphicsSimpleTextItem(str(number), self)
        self.label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.label.setBrush(QColor(TEXT))
        self._place_label()
        self._restyle()

    # --- appearance ---
    def _restyle(self, hover=False):
        fill = QColor(self.colour)
        if self.isSelected():
            self.setPen(_pen(TEXT, 2.5))
            fill.setAlpha(130)
        else:
            self.setPen(_pen(self.colour, 1.5))
            fill.setAlpha(110 if hover else 70)
        self.setBrush(fill)

    def _place_label(self):
        c = self.polygon().boundingRect().center()
        self.label.setPos(c)

    def hoverEnterEvent(self, e):
        self._restyle(True)

    def hoverLeaveEvent(self, e):
        self._restyle(False)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._restyle()
            self._show_handles(bool(value))
        return super().itemChange(change, value)

    # --- editing ---
    def _show_handles(self, show):
        for h in self.handles:
            self.scene().removeItem(h)
        self.handles = []
        if show:
            for i, p in enumerate(self.polygon()):
                h = VertexHandle(self, i)
                h.setPos(p)
                self.handles.append(h)

    def move_vertex(self, i, pos):
        poly = self.polygon()
        if i < len(poly):
            poly[i] = pos
            self.setPolygon(poly)

    def remove_vertex(self, i):
        poly = self.polygon()
        if len(poly) <= 3:
            return
        poly.remove(i)
        self.setPolygon(poly)
        self.canvas.commit(self)  # rebuilds the handles (not here: one of them is handling this click)

    def insert_vertex(self, scene_pos):
        p = self.mapFromScene(scene_pos)
        poly = [self.polygon()[i] for i in range(len(self.polygon()))]
        best, best_d = 0, float("inf")
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ab = b - a
            t = max(0.0, min(1.0, QPointF.dotProduct(p - a, ab) / max(QPointF.dotProduct(ab, ab), 1e-9)))
            q = a + ab * t
            d = (p - q).manhattanLength()
            if d < best_d:
                best, best_d = i, d
        poly.insert(best + 1, p)
        self.setPolygon(QPolygonF(poly))
        self._show_handles(True)
        self.canvas.commit(self)

    def scene_polygon(self):
        return [[self.mapToScene(p).x(), self.mapToScene(p).y()] for p in self.polygon()]

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.pos() != QPointF(0, 0):
            self.canvas.commit(self)

    def mouseDoubleClickEvent(self, e):
        if self.isSelected():
            self.insert_vertex(e.scenePos())
        super().mouseDoubleClickEvent(e)


class Canvas(QGraphicsView):
    edited = Signal(list)       # new list of seeds after a completed edit
    seedSelected = Signal(int)  # index into the seed list, -1 for none
    boxDrawn = Signal(float, float, float, float)  # BOX mode: a dragged rectangle (scene coords)
    pointClicked = Signal(float, float, bool)       # BOX mode: a click (x, y, True = seed / False = not seed)
    resegmentRequested = Signal()                   # R: redraw the selected seed with SAM

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(CANVAS))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.image_item = None
        self.guide_item = None
        self.items_ = []
        self.mode = "select"
        self._panning = None
        self._space = False
        self._box_start = None
        self._box_item = None
        self.scene().selectionChanged.connect(self._on_selection)

    # --- content ---
    def set_image(self, bgr):
        if self.image_item is not None:
            self.scene().removeItem(self.image_item)
        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(to_qimage(bgr)))
        self.image_item.setZValue(-10)
        self.image_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.image_item)
        self.setSceneRect(self.image_item.boundingRect())

    def swap_image(self, bgr):
        """Replace the picture (e.g. colour <-> height) without touching the view."""
        if self.image_item is None:
            return self.set_image(bgr)
        self.image_item.setPixmap(QPixmap.fromImage(to_qimage(bgr)))

    def set_seeds(self, seeds, keep_selection=None):
        self.scene().blockSignals(True)
        for it in self.items_:
            self.scene().removeItem(it)
        self.items_ = []
        for n, s in enumerate(seeds, start=1):
            it = SeedItem(self, s["polygon"], s["status"], n)
            if self.mode != "select":
                it.setFlag(QGraphicsItem.ItemIsSelectable, False)
                it.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.scene().addItem(it)
            self.items_.append(it)
        self.scene().blockSignals(False)
        if keep_selection is not None and 0 <= keep_selection < len(self.items_):
            self.items_[keep_selection].setSelected(True)

    def clear(self):
        self.scene().clear()
        self.image_item, self.guide_item, self.items_ = None, None, []

    def set_guide(self, points):
        """Dashed outline (scene coordinates), e.g. the area where stereo gives depth."""
        if self.guide_item is not None:
            self.scene().removeItem(self.guide_item)
            self.guide_item = None
        if points is None or len(points) < 3:
            return
        path = QPainterPath()
        path.moveTo(QPointF(*points[0]))
        for x, y in points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()
        pen = _pen(GUIDE, 2.0)
        pen.setStyle(Qt.CustomDashLine)
        pen.setDashPattern([8, 6])
        self.guide_item = QGraphicsPathItem(path)
        self.guide_item.setPen(pen)
        self.guide_item.setZValue(30)
        self.scene().addItem(self.guide_item)

    def seeds(self):
        return [{"polygon": it.scene_polygon(), "status": it.status} for it in self.items_]

    def fit(self):
        if self.image_item is not None:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def select_index(self, i, centre=True):
        self.scene().clearSelection()
        if 0 <= i < len(self.items_):
            it = self.items_[i]
            it.setSelected(True)
            if centre:
                self.centerOn(it)

    # --- edits ---
    def commit(self, item):
        # Defer: the edit is triggered from inside the item's own mouse event, and
        # rebuilding the items there would delete the object handling the event.
        QTimer.singleShot(0, lambda: self._commit_now(item))

    def _commit_now(self, item):
        if item not in self.items_:
            return
        if item.status == "auto":
            item.status = "edited"
        idx = self.items_.index(item)
        seeds = self.seeds()
        self.set_seeds(seeds, keep_selection=idx)
        self.edited.emit(seeds)

    def delete_selected(self):
        sel = [it for it in self.items_ if it.isSelected()]
        if not sel:
            return
        seeds = [s for it, s in zip(self.items_, self.seeds()) if it not in sel]
        self.set_seeds(seeds)
        self.edited.emit(seeds)

    def set_mode(self, mode):
        """select (edit), box (SAM prompts) or view (read-only)."""
        self.mode = mode
        self.setDragMode(QGraphicsView.RubberBandDrag if mode == "select" else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.CrossCursor if mode == "box" else Qt.ArrowCursor)
        for it in self.items_:
            it.setFlag(QGraphicsItem.ItemIsSelectable, mode == "select")
            it.setFlag(QGraphicsItem.ItemIsMovable, mode == "select")
            it.setAcceptHoverEvents(mode != "view")

    # --- events ---
    def _on_selection(self):
        sel = [i for i, it in enumerate(self.items_) if it.isSelected()]
        self.seedSelected.emit(sel[0] if len(sel) == 1 else -1)

    def wheelEvent(self, e):
        f = 1.0015 ** e.angleDelta().y()
        self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton or (self._space and e.button() == Qt.LeftButton):
            self._panning = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if self.mode == "box" and e.button() in (Qt.LeftButton, Qt.RightButton):
            negative = e.button() == Qt.RightButton or bool(e.modifiers() & Qt.ShiftModifier)
            self._box_start = (e.position(), self.mapToScene(e.position().toPoint()), negative)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning is not None:
            d = e.position() - self._panning
            self._panning = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self.mode == "box" and self._box_start is not None and not self._box_start[2]:
            a, b = self._box_start[1], self.mapToScene(e.position().toPoint())
            if self._box_item is None:
                self._box_item = QGraphicsRectItem()
                pen = _pen(ACCENT, 1.5)
                pen.setStyle(Qt.DashLine)
                self._box_item.setPen(pen)
                self._box_item.setZValue(25)
                self.scene().addItem(self._box_item)
            self._box_item.setRect(QRectF(a, b).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning is not None:
            self._panning = None
            self.viewport().setCursor(Qt.CrossCursor if self.mode == "box" else Qt.ArrowCursor)
            return
        if self.mode == "box" and self._box_start is not None:
            start_view, start, negative = self._box_start
            self._box_start = None
            if self._box_item is not None:
                self.scene().removeItem(self._box_item)
                self._box_item = None
            end = self.mapToScene(e.position().toPoint())
            if not negative and (e.position() - start_view).manhattanLength() > 6:
                self.boxDrawn.emit(start.x(), start.y(), end.x(), end.y())
            else:
                self.pointClicked.emit(end.x(), end.y(), not negative)
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
        elif k in (Qt.Key_Delete,) and self.mode == "select":
            self.delete_selected()
        elif k == Qt.Key_Escape:
            self.scene().clearSelection()
        elif k == Qt.Key_F:
            self.fit()
        elif k == Qt.Key_R and self.mode != "view":
            self.resegmentRequested.emit()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.viewport().setCursor(Qt.CrossCursor if self.mode == "box" else Qt.ArrowCursor)
        super().keyReleaseEvent(e)
