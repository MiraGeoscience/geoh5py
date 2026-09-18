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

import json
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

from geoh5py.ui_json.forms import (
    BaseForm,
    BoolForm,
    FloatForm,
    IntegerForm,
    RadioLabelForm,
)
from geoh5py.ui_json.ui_json import UIJson


def _as_uuid_text(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    if value in (None, ""):
        return ""
    return str(value)


class FormWidget(QWidget):
    def __init__(self, name: str, form: BaseForm, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.form = form
        self._build()

    def _build(self):
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
        self.setWindowTitle(ui_json.title)
        self._build()

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        for name in self.ui_json.model_fields:
            value = getattr(self.ui_json, name, None)
            if isinstance(value, BaseForm):
                if not value.visible:
                    continue
                group = QGroupBox(value.group or "Parameters")
                group_layout = QVBoxLayout(group)
                group_layout.addWidget(FormWidget(name, value))
                layout.addWidget(group)

        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)


def run_ui_json(path: str | Path | dict[str, Any]):
    data = (
        json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(path, dict)
        else path
    )
    ui_json = UIJson.from_dict(data)
    app = QApplication.instance() or QApplication([])
    window = UIJsonWindow(ui_json)
    window.resize(720, 640)
    window.show()
    return app, window


if __name__ == "__main__":
    import sys

    app, window = run_ui_json(
        sys.argv[1]
        if len(sys.argv) > 1
        else r"C:\Users\dominiquef\Documents\tests\dist.ui.json"
    )
    app.exec_()
