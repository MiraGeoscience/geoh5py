#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from ..groups import PropertyGroup
from ..shared import Entity
from ..workspace import Workspace

default_ui_json = {
    "title": "Custom UI",
    "geoh5": None,
    "run_command": None,
    "run_command_boolean": {
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "monitoring_directory": None,
    "conda_environment": None,
    "conda_environment_boolean": False,
    "workspace": None,
}
ui_validations = {
    "association": {"values": ["Vertex", "Cell"]},
    "dataGroupType": {
        "values": [
            "Multi-element",
            "3D vector",
            "Dip direction & dip",
            "Strike & dip",
        ],
    },
    "dataType": {"values": ["Float", "Text", "Integer", "Referenced"]},
    "dependency": {"types": [str, type(None)]},
    "dependencyType": {"values": ["enabled", "disabled"]},
    "enabled": {"types": [bool, type(None)]},
    "group": {"types": [str, type(None)]},
    "label": {"required": True, "types": [str]},
    "main": {"types": [bool, type(None)]},
    "meshType": {"types": [str, UUID, type(None)]},
    "optional": {"types": [bool, type(None)]},
    "parent": {"types": [str, UUID, type(None)]},
    "property": {"types": [str, UUID, type(None)]},
    "value": {
        "required": True,
        "types": [str, float, int, bool, type(None), Entity, UUID, PropertyGroup],
    },
    "tooltip": {"types": [str, type(None)]},
}
base_validations = {
    "title": {"required": True, "types": [str]},
    "conda_environment": {
        "types": [str, type(None)],
    },
    "conda_environment_boolean": {
        "types": [bool],
    },
    "geoh5": {"required": True, "types": [str, Workspace, type(None)]},
    "monitoring_directory": {
        "types": [str, type(None)],
    },
    "run_command": {
        "types": [str, type(None)],
    },
    "run_command_boolean": {
        "types": [bool],
    },
    "workspace": {
        "types": [str, Workspace, type(None)],
    },
}
