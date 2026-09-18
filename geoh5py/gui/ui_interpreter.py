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

from pathlib import Path
from typing import Any
from uuid import UUID

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from geoh5py.shared.validators import name_or_uid_to_type
from geoh5py.ui_json.forms import (
    BaseForm,
    BoolForm,
    DataForm,
    FloatForm,
    GroupForm,
    IntegerForm,
    ObjectForm,
    RadioLabelForm,
)
from geoh5py.ui_json.ui_json import UIJson
from geoh5py.workspace import Workspace


def _as_uuid_text(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    if value in (None, ""):
        return ""
    return str(value)


class FormWidget(QWidget):
    def __init__(
        self,
        name: str,
        ui_json: UIJson,
        parent: QWidget | None = None,
        workspace: Workspace = None,
    ):
        super().__init__(parent)
        self.name = name
        self.ui_json = ui_json
        self.form = getattr(self.ui_json, name)
        self._build(workspace)

    def _build(self, workspace: Workspace):
        layout = QFormLayout(self)
        label = QLabel(
            self.form.label
            if isinstance(self.form.label, str)
            else " ".join(self.form.label)
        )
        label.setToolTip(
            self.form.tooltip
            if isinstance(self.form.tooltip, str)
            else " ".join(self.form.tooltip)
        )

        if isinstance(self.form, BoolForm):
            editor = QCheckBox()
            editor.setChecked(bool(self.form.value))
        elif isinstance(self.form, IntegerForm):
            editor = QSpinBox()
            editor.setRange(
                int(self.form.min if self.form.min != float("-inf") else -2147483648),
                int(self.form.max if self.form.max != float("inf") else 2147483647),
            )
            editor.setValue(int(self.form.value))
        elif isinstance(self.form, FloatForm):
            editor = QDoubleSpinBox()
            editor.setDecimals(self.form.precision)
            editor.setRange(
                float(self.form.min if self.form.min != float("-inf") else -1e308),
                float(self.form.max if self.form.max != float("inf") else 1e308),
            )
            editor.setValue(float(self.form.value))
        elif isinstance(self.form, RadioLabelForm):
            editor = QWidget()
            row = QHBoxLayout(editor)
            original = QRadioButton(self.form.original_label)
            alternate = QRadioButton(self.form.alternate_label)
            original.setChecked(self.form.value == self.form.original_label)
            alternate.setChecked(self.form.value == self.form.alternate_label)
            row.addWidget(original)
            row.addWidget(alternate)
            row.addStretch(1)
        elif hasattr(self.form, "choice_list"):
            editor = QComboBox()
            choices = list(self.form.choice_list)
            editor.addItems([str(choice) for choice in choices])
            value = getattr(self.form, "value", "")
            if isinstance(value, list):
                value = value[0] if value else ""
            if value is not None:
                editor.setCurrentText(str(value))

        elif isinstance(self.form, GroupForm):
            editor = QComboBox()
            group_type = getattr(self.form, "group_type", None)
            editor.addItems(
                [obj.name for obj in workspace.groups if isinstance(obj, group_type)]
            )
            value = getattr(self.form, "value", "")

            entity = workspace.get_entity(value)[0]
            if value is not None:
                editor.setCurrentText(entity.name)

        elif isinstance(self.form, ObjectForm):
            editor = QComboBox()
            mesh_type = tuple(getattr(self.form, "mesh_type", None))
            editor.addItems(
                [obj.name for obj in workspace.objects if isinstance(obj, mesh_type)]
            )
            value = getattr(self.form, "value", "")

            entity = workspace.get_entity(value)[0]
            if value is not None:
                editor.setCurrentText(entity.name)

        elif isinstance(self.form, DataForm):
            editor = QComboBox()
            data_type = getattr(self.form, "data_type", None)

            if not isinstance(data_type, list):
                data_type = [data_type]

            parent_form = getattr(self.ui_json, getattr(self.form, "parent", None))
            parent_value = getattr(parent_form, "value", None) if parent_form else None

            if parent_value is not None:
                parent_entity = workspace.get_entity(parent_value)[0]
                editor.addItems(
                    [
                        obj.name
                        for obj in parent_entity.children
                        if obj.entity_type.primitive_type in data_type
                    ]
                )

            value = getattr(self.form, "value", "")

            entity = workspace.get_entity(value)[0]
            if value is not None:
                editor.setCurrentText(entity.name)

        elif hasattr(self.form, "value"):
            editor = QLineEdit(_as_uuid_text(self.form.value))
        else:
            editor = QLabel("Unsupported form")

        editor.setEnabled(self.form.enabled)
        layout.addRow(label, editor)
        self.editor = editor


class UIJsonWindow(QMainWindow):
    def __init__(self, ui_json: UIJson):
        super().__init__()
        self.ui_json = ui_json
        self._field_widgets: dict[str, FormWidget] = {}
        self.setWindowTitle(ui_json.title)
        self._build()

    def _apply(self):
        values = {}
        for name, widget in self._field_widgets.items():
            editor = widget.editor
            if isinstance(editor, QCheckBox):
                values[name] = editor.isChecked()
            elif isinstance(editor, QSpinBox):
                values[name] = editor.value()
            elif isinstance(editor, QDoubleSpinBox):
                values[name] = editor.value()
            elif isinstance(editor, QComboBox):
                values[name] = editor.currentText()
            elif isinstance(editor, QLineEdit):
                values[name] = editor.text()
            else:
                values[name] = getattr(widget.form, "value", None)

        print(f"Applying UIJson values: {values}")
        self.close()

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        with Workspace(self.ui_json.geoh5) as workspace:
            for name, form in self.ui_json:
                if not isinstance(form, BaseForm) or not form.visible:
                    continue

                group = QGroupBox(form.group or "Parameters")
                group_layout = QVBoxLayout(group)
                widget = FormWidget(name, self.ui_json, workspace=workspace)
                self._field_widgets[name] = widget
                group_layout.addWidget(widget)
                layout.addWidget(group)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.addStretch(1)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply)
        button_layout.addWidget(apply_button)
        layout.addWidget(button_row)

        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)


def edit_ui_json(
    path: str | Path | dict[str, Any] | UIJson,
) -> tuple[QApplication, UIJsonWindow]:

    if isinstance(path, dict):
        ui_json = UIJson.from_dict(path)
    elif isinstance(path, str | Path):
        ui_json = UIJson.read(path)
    else:
        ui_json = path

    app = QApplication.instance() or QApplication([])
    window = UIJsonWindow(ui_json)
    window.resize(720, 700)
    window.show()

    return app, window


if __name__ == "__main__":
    import sys

    app, window = edit_ui_json(
        sys.argv[1]
        if len(sys.argv) > 1
        else r"C:\Users\dominiquef\Documents\tests\prototype_workflows\single_ui.ui.json"
    )
    app.exec()
