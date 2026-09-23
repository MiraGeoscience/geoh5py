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
from pathlib import Path

import networkx as nx
import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QMainWindow,
    QMenu,
)

from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.groups import RootGroup, UIJsonGroup
from geoh5py.gui.ui_interpreter import edit_ui_json
from geoh5py.objects import ObjectBase
from geoh5py.shared.entity import Substitute
from geoh5py.ui_json import UIJson


@dataclass
class CanvasNode:
    name: str
    position: QPointF
    kind: str = "object"
    object_ref: object | None = None

    def available_functions(self) -> list[str]:
        obj = self.object_ref if self.object_ref is not None else self
        names: list[str] = []
        for member_name in dir(obj):
            if member_name.startswith("_") or member_name in {
                "available_functions",
                "execute_function",
            }:
                continue
            member = getattr(obj, member_name)
            if callable(member):
                names.append(member_name)
        return sorted(names)

    def execute_function(self, function_name: str, *args, **kwargs):
        obj = self.object_ref if self.object_ref is not None else self
        if not hasattr(obj, function_name):
            raise AttributeError(
                f"Node '{self.name}' has no function '{function_name}'."
            )

        method = getattr(obj, function_name)
        if not callable(method):
            raise TypeError(f"'{function_name}' is not callable.")

        try:
            return method(*args, **kwargs)
        except TypeError as exc:  # pragma: no cover - runtime guard for GUI feedback
            raise TypeError(
                f"Function '{function_name}' could not be executed with the supplied arguments."
            ) from exc


COLOR_MAP = {
    "object": QColor("#4C78A8"),
    "data": QColor("#F58518"),
    "group": QColor("#E45756"),
    "future": QColor("#72B7B2"),
}

SIZE_MAP = {
    "object": (32, 32),
    "data": (16, 16),
    "group": (64, 64),
    "future": (32, 32),
}


class NodeItem(QGraphicsEllipseItem):
    def __init__(self, node: CanvasNode):

        size = SIZE_MAP[node.kind]
        super().__init__(-size[0] / 2, -size[0] / 2, *size)
        self.node = node
        self._links: list[ConnectionItem] = []
        self.setBrush(QBrush(QColor(COLOR_MAP[node.kind])))
        self.setPen(QPen(Qt.GlobalColor.white, 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPos(node.position)

        label = QGraphicsTextItem(node.name, self)
        label.setDefaultTextColor(Qt.GlobalColor.white)
        label.setPos(-label.boundingRect().width() / 2, size[0] / 2)

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

        pen_args = [QColor("#666666"), 2]

        if "future" in source.node.name or "future" in target.node.name:
            pen_args += [Qt.DashLine]

        self.setPen(QPen(*pen_args))
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
        self._context_menu_node: CanvasNode | None = None
        for node in nodes:
            item = NodeItem(node)
            self.addItem(item)
            self._node_items[node.name] = item

        for source, target in connections:
            if source in self._node_items and target in self._node_items:
                self.addItem(
                    ConnectionItem(self._node_items[source], self._node_items[target])
                )

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if item is None or not isinstance(item, NodeItem):
            return

        menu = QMenu()
        self._context_menu_node = item.node
        for function_name in item.node.available_functions():
            action = QAction(function_name, menu)
            action.triggered.connect(
                lambda checked=False, fn=function_name: self._execute_function(fn)
            )
            menu.addAction(action)
        menu.exec_(event.screenPos())

    def _execute_function(self, function_name: str):
        if self._context_menu_node is None:
            return
        try:
            app, window = self._context_menu_node.execute_function(function_name)
            app.exec()
            self.sub_app = app
            self.sub_window = window
            print(f"Executed {function_name}")
        except Exception as exc:  # pragma: no cover - GUI feedback path
            print(f"Error executing {function_name}: {exc}")


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


class NodeActions:
    def __init__(self, entity: UIJson):
        self.entity = entity

    def edit_options(self):
        return edit_ui_json(self.entity)

    def run_from_here(self):
        pass


def get_tree_depth(entity: ObjectBase, depth=1) -> int:
    if isinstance(entity, RootGroup):
        return depth

    return get_tree_depth(entity.parent, depth + 1)


def set_network(file: Path):

    nodes = {}
    connections = []
    graph = nx.DiGraph()

    with Workspace(file) as workspace:
        for group in workspace.groups:
            actions = None
            if isinstance(group, UIJsonGroup):
                uijson = UIJson.from_dict(group.options)
                actions = NodeActions(uijson)

                # if group.name == "Second UI":
                #     uijson.set_values(**{"data_mesh": "{da1c8f8f-9f70-48f4-85e9-de261022f8eb}"})
                #     uijson.to_ui_json_group(workspace=workspace)
                #     workspace.remove_entity(group)
                #     del group

                options = uijson.to_params(workspace=workspace)

                for elem in options.values():
                    if not isinstance(elem, ObjectBase | Data):
                        continue

                    name = elem.name
                    if isinstance(elem, Substitute):
                        kind = "future"
                        name = name.replace(elem.parent.name, "")
                        connections.append((elem.parent.name, name))
                    elif isinstance(elem, ObjectBase):
                        kind = "object"
                        if (elem.parent.name, name) not in connections:
                            connections.append((elem.parent.name, name))
                    else:
                        kind = "data"
                        if (elem.parent.name, name) not in connections:
                            connections.append((elem.parent.name, name))

                    nodes[elem] = CanvasNode(name, QPointF(0, 0), kind=kind)
                    connections.append((name, group.name))

            elif not isinstance(group, RootGroup):
                connections.append((group.parent.name, group.name))

            nodes[group] = CanvasNode(
                group.name,
                QPointF(0, 0),
                object_ref=actions,
                kind="group",
            )

        graph.add_edges_from(connections)

        # Re-order nodes to ensure that hiarchy of operations is respected
        levels = dict.fromkeys(range(len(nodes)), 0)
        planets = {}
        satellites = {}
        max_depth = 0
        for entity, node in nodes.items():
            depth = max(
                [
                    len(path)
                    for path in nx.all_simple_paths(graph, "Workspace", node.name)
                ]
            )
            max_depth = max(depth, max_depth)
            levels[depth] += 1

            position = depth * 200, levels[depth] * 100

            if isinstance(entity, ObjectBase):
                planets[entity.parent] = planets.get(entity.parent, []) + [entity]
            elif isinstance(entity, Data):
                satellites[entity.parent] = satellites.get(entity.parent, []) + [entity]

            node.position = QPointF(*position)

        for parent, children in planets.items():
            angles = np.linspace(np.pi / 4, -np.pi / 4, len(children))

            depth = max(
                [
                    len(path)
                    for path in nx.all_simple_paths(
                        graph, "Workspace", nodes[parent].name
                    )
                ]
            )

            radius = 50 * (max_depth - depth)
            for angle, child in zip(angles, children):
                nodes[child].position = nodes[parent].position + QPointF(
                    np.cos(angle) * radius, np.sin(angle) * radius
                )

        for parent, children in satellites.items():
            angles = np.linspace(-np.pi / 4, np.pi / 4, len(children))

            radius = 100
            for angle, child in zip(angles, children):
                nodes[child].position = nodes[parent].position + QPointF(
                    np.cos(angle) * radius, np.sin(angle) * radius
                )

    return list(nodes.values()), connections


if __name__ == "__main__":
    file = r"C:\Users\dominiquef\Documents\tests\prototype_workflows\workload.geoh5"

    nodes, connections = set_network(file)

    # nodes = [
    #     CanvasNode("Object A", QPointF(0, 0), object_ref=NodeActions()),
    #     CanvasNode("Object B", QPointF(300, 200), object_ref=NodeActions()),
    #     CanvasNode("Object C", QPointF(500, 100), object_ref=NodeActions()),
    # ]
    # connections = [("Object A", "Object B"), ("Object B", "Object C")]
    app, window = show_canvas(nodes, connections)
    app.exec()
