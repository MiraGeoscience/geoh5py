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
from shutil import copy
from uuid import UUID

import networkx as nx
import numpy as np
from jupyter_server.gateway import connections
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
from geoh5py.groups import Group, RootGroup, UIJsonGroup
from geoh5py.gui.ui_interpreter import edit_ui_json
from geoh5py.objects import ObjectBase
from geoh5py.shared.entity import Entity, Substitute
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import UIJson
from geoh5py.ui_json.forms import BaseForm


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

        if "future" in source.node.name.lower() or "future" in target.node.name.lower():
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
    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.max_depth = 0
        self.levels = {}
        self.node_items: dict[UUID, NodeItem] = {}
        self.connections: list[tuple[UUID, UUID]] = []
        self._context_menu_node: CanvasNode | None = None

    def get_depth(self, uid: UUID) -> int:
        if self.connections:
            graph = nx.DiGraph()
            graph.add_edges_from(self.connections)

            return max(
                [
                    len(path)
                    for path in nx.all_simple_paths(
                        graph, list(self.node_items)[0], uid
                    )
                ]
            )
        return 0

    def add_node(self, entity: Entity):
        if entity.uid in self.node_items:
            return

        name = entity.name

        position = (0, 0)
        if isinstance(entity, Group):
            kind = "group"
        elif isinstance(entity, Substitute):
            kind = "future"
            name = name.replace(entity.parent.name, "")
        elif isinstance(entity, ObjectBase):
            kind = "object"
        else:
            kind = "data"

        node = CanvasNode(
            name,
            QPointF(*position),
            object_ref=None,
            kind=kind,
        )
        item = NodeItem(node)
        self.addItem(item)
        self.node_items[entity.uid] = item

    def set_positions(self):

        self.max_depth = 0
        self.levels = {}

        for uid in self.node_items:
            entity = self.workspace.get_entity(uid)[0]

            if not isinstance(entity, Group):
                continue

            depth = self.get_depth(uid)
            self.max_depth = max(depth, self.max_depth)
            self.levels[depth] = self.levels.get(depth, 1) + 1
            position = depth * 200, self.levels[depth] * 200 + 50 * (-1) ** depth
            position = QPointF(*position)

            self.node_items[uid].node.position = position
            self.node_items[uid].setPos(position)

        for uid in self.node_items:
            entity = self.workspace.get_entity(uid)[0]
            if not isinstance(entity, ObjectBase):
                continue
            children = [
                uids[1] for uids in self.connections if uids[0] == entity.parent.uid
            ]
            ind = children.index(entity.uid)
            angle = np.linspace(np.pi / 4, -np.pi / 4, len(children))[ind]

            radius = 150
            position = self.node_items[entity.parent.uid].node.position + QPointF(
                np.cos(angle) * radius, np.sin(angle) * radius
            )

            self.node_items[uid].node.position = position
            self.node_items[uid].setPos(position)

        for uid in self.node_items:
            entity = self.workspace.get_entity(uid)[0]
            if not isinstance(entity, Data):
                continue
            children = [
                uids[1] for uids in self.connections if uids[0] == entity.parent.uid
            ]
            ind = children.index(entity.uid)
            angle = np.linspace(np.pi / 4, -np.pi / 4, len(children))[ind]

            radius = 50
            position = self.node_items[entity.parent.uid].node.position + QPointF(
                np.cos(angle) * radius, np.sin(angle) * radius
            )

            self.node_items[uid].node.position = position
            self.node_items[uid].setPos(position)

    def add_connection(self, connection: tuple[UUID, UUID]):
        if connection not in self.connections:
            source, target = connection
            if source in self.node_items and target in self.node_items:
                self.addItem(
                    ConnectionItem(self.node_items[source], self.node_items[target])
                )
            self.connections.append(connection)

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
            app, window = self._context_menu_node.execute_function(function_name, self)
            app.exec()
            self.sub_app = app
            self.sub_window = window
            print(f"Executed {function_name}")
        except Exception as exc:  # pragma: no cover - GUI feedback path
            print(f"Error executing {function_name}: {exc}")


class CanvasWindow(QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.setWindowTitle("Entity Canvas")
        self.scene = EntityCanvas(workspace, self)

        view = ConnectionView(self.scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCentralWidget(view)


def show_canvas(workspace):
    app = QApplication.instance() or QApplication([])
    window = CanvasWindow(workspace)
    window.resize(900, 600)
    window.show()
    return app, window


class NodeActions:
    def __init__(self, entity: UIJsonGroup):
        self.entity = entity

    def edit_options(self, *_):
        return edit_ui_json(self.entity)

    def fork_here(self, canvas):
        self._recursive_add_forward(self.entity, canvas)
        canvas.set_positions()

    def _recursive_add_forward(
        self, entity: Entity, canvas: EntityCanvas, substitutes: dict[UUID, UUID] = {}
    ) -> Entity:
        with fetch_active_workspace(entity.workspace) as ws:
            new_entity = entity.copy(parent=ws)
            new_entity.name = f"{entity.name}"

            if isinstance(new_entity, UIJsonGroup):
                for orig, sub in zip(
                    entity.substitutes.values(), new_entity.substitutes.values()
                ):
                    substitutes[orig.uid] = sub.uid

                dependents = [
                    uids[1] for uids in canvas.connections if uids[0] == orig.uid
                ]

                for dependent in dependents:
                    dependent_entity = ws.get_entity(dependent)[0]
                    new_dependent = self._recursive_add_forward(
                        dependent_entity, canvas, substitutes
                    )

                uijson = UIJson.from_dict(new_entity.options)
                for key, form in uijson:
                    if isinstance(form, BaseForm) and form.value in substitutes:
                        form.value = substitutes[form.value]

                options = uijson.serialize("json")
                options = {
                    key: (item if item is not None else "")
                    for key, item in options.items()
                }
                new_entity.options = options
                recursive_add_nodes(new_entity, canvas)

        return new_entity

    def run_from_here(self, *_):
        pass


def get_tree_depth(entity: ObjectBase, depth=1) -> int:
    if isinstance(entity, RootGroup):
        return depth

    return get_tree_depth(entity.parent, depth + 1)


def recursive_add_nodes(entity: Entity, canvas: EntityCanvas):
    if entity.uid in canvas.node_items:
        return

    canvas.add_node(entity)

    if not isinstance(entity, RootGroup):
        recursive_add_nodes(entity.parent, canvas)

        connection = (entity.parent.uid, entity.uid)
        canvas.add_connection(connection)

    if isinstance(entity, UIJsonGroup):
        uijson = UIJson.from_dict(entity.options)
        actions = NodeActions(entity)
        canvas.node_items[entity.uid].node.object_ref = actions

        options = uijson.to_params(workspace=entity.workspace)
        for elem in options.values():
            if not isinstance(elem, Entity):
                continue

            recursive_add_nodes(elem, canvas)
            connection = (elem.uid, entity.uid)
            canvas.add_connection(connection)


def set_network(file: Workspace, canvas: EntityCanvas):

    with fetch_active_workspace(file) as workspace:
        for group in workspace.groups:
            recursive_add_nodes(group, canvas)
        canvas.set_positions()


def main(geoh5: Path):

    with Workspace(geoh5) as workspace:
        app, window = show_canvas(workspace)
        set_network(workspace, window.scene)

        app.exec()


def mock_linkage(geoh5):
    with Workspace(geoh5) as workspace:
        group = workspace.get_entity("Weight of Evidence")[0]
        uijson = UIJson.from_dict(group.options)
        uijson.set_values(**{"mesh": "{da1c8f8f-9f70-48f4-85e9-de261022f8eb}"})
        uijson.to_ui_json_group(workspace=workspace)
        workspace.remove_entity(group)
        del group


if __name__ == "__main__":
    file = r"C:/Users/dominiquef/AppData/Local/Mira Geoscience/Geoscience ANALYST/Session Cache/{e470a76e-568a-4654-b8da-a763415c6c33}/Python/Files/GA-WorkflowPanel_0925-084708.ui.json"
    # file = sys.argv[1]
    file_path = Path(file)
    ui_json = UIJson.read(file_path)
    working_file = file_path.parent / ui_json.workspace_geoh5.name
    copy(ui_json.workspace_geoh5, working_file)

    mock_linkage(working_file)

    main(working_file)
