# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QMainWindow,
)


@dataclass
class CanvasNode:
    name: str
    position: QPointF
    kind: str = "object"


class NodeItem(QGraphicsEllipseItem):
    def __init__(self, node: CanvasNode):
        super().__init__(-28, -28, 56, 56)
        self.node = node
        self._links: list[ConnectionItem] = []
        self.setBrush(QBrush(QColor("#4C78A8")))
        self.setPen(QPen(Qt.GlobalColor.white, 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPos(node.position)

        label = QGraphicsTextItem(node.name, self)
        label.setDefaultTextColor(Qt.GlobalColor.white)
        label.setPos(-label.boundingRect().width() / 2, 34)

    def add_link(self, link: ConnectionItem):
        self._links.append(link)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for link in self._links:
                link.update_path()
        return super().itemChange(change, value)


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        self.setPen(QPen(QColor("#666666"), 2))
        self.setZValue(-1)
        source.add_link(self)
        target.add_link(self)
        self.update_path()

    def update_path(self):
        start = self.source.scenePos()
        end = self.target.scenePos()
        path = QPainterPath(start)
        delta = end - start
        ctrl1 = start + QPointF(delta.x() * 0.35, 0)
        ctrl2 = end - QPointF(delta.x() * 0.35, 0)
        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)


class ConnectionView(QGraphicsView):
    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#1F1F1F"))


class EntityCanvas(QGraphicsScene):
    def __init__(
        self, nodes: list[CanvasNode], connections: list[tuple[str, str]], parent=None
    ):
        super().__init__(parent)
        self._node_items: dict[str, NodeItem] = {}
        for node in nodes:
            item = NodeItem(node)
            self.addItem(item)
            self._node_items[node.name] = item

        for source, target in connections:
            if source in self._node_items and target in self._node_items:
                self.addItem(
                    ConnectionItem(self._node_items[source], self._node_items[target])
                )


class CanvasWindow(QMainWindow):
    def __init__(self, nodes: list[CanvasNode], connections: list[tuple[str, str]]):
        super().__init__()
        self.setWindowTitle("Entity Canvas")
        self.scene = EntityCanvas(nodes, connections, self)
        view = ConnectionView(self.scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCentralWidget(view)


def show_canvas(nodes: list[CanvasNode], connections: list[tuple[str, str]]):
    app = QApplication.instance() or QApplication([])
    window = CanvasWindow(nodes, connections)
    window.resize(900, 600)
    window.show()
    return app, window


if __name__ == "__main__":
    nodes = [
        CanvasNode("Object A", QPointF(100, 100)),
        CanvasNode("Object B", QPointF(300, 200)),
        CanvasNode("Object C", QPointF(500, 100)),
    ]
    connections = [("Object A", "Object B"), ("Object B", "Object C")]
    app, window = show_canvas(nodes, connections)
    app.exec()
