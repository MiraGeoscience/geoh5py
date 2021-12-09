#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from uuid import UUID

from ..workspace import Workspace

defaults = {
    "title": "Custom UI",
    "geoh5": "",
    "run_command": "",
    "run_command_boolean": False,
    "conda_environment": "",
    "conda_environment_boolean": False,
    "workspace": "",
    "monitoring_directory": "",
}

default_ui_json = {
    "title": "Custom UI",
    "geoh5": None,
    "run_command": "",
    "run_command_boolean": {
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "monitoring_directory": None,
    "conda_environment": "",
    "conda_environment_boolean": False,
    "workspace": None,
}
ui_validations = {
    "association": {"types": [str, [str]], "values": ["Vertex", "Cell"]},
    "dataGroupType": {"types": [str], "values": ["Multi-element"]},
    "dataType": {"types": [str], "values": ["Float", "Text", "Integer"]},
    "dependency": {"types": [str]},
    "dependencyType": {"types": [str], "values": ["enabled", "disabled"]},
    "group": {"types": [str]},
    "label": {"types": [str], "required": True},
    "main": {"types": [bool]},
    "meshType": {"types": [[str, UUID]]},
    "parent": {"types": [str]},
    "property": {"types": [str, UUID]},
    "value": {"types": [str, float, int, bool], "required": True},
    "tooltip": {"types": [str]},
}
validations = {
    "title": {"types": [str], "required": True},
    "conda_environment": {
        "types": [str],
    },
    "conda_environment_boolean": {
        "types": [bool],
    },
    "geoh5": {"types": [str, Workspace], "required": True},
    "monitoring_directory": {
        "types": [str],
    },
    "run_command": {
        "types": [str],
    },
    "run_command_boolean": {
        "types": [bool],
    },
    "workspace": {
        "types": [str, Workspace],
    },
}
