#  Copyright (c) 2024 Mira Geoscience Ltd.
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


from __future__ import annotations

from pathlib import Path
from uuid import UUID

from ..groups import PropertyGroup, UIJsonGroup
from ..shared import Entity
from ..workspace import Workspace


default_ui_json = {
    "version": "",
    "title": "Custom UI",
    "geoh5": None,
    "run_command": None,
    "monitoring_directory": None,
    "conda_environment": None,
    "workspace_geoh5": None,
    "out_group": None,
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
    "dataType": {"values": ["Float", "Text", "Integer", "Referenced", "Boolean"]},
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
    "version": {"types": [str, type(None)]},
    "title": {"required": True, "types": [str]},
    "conda_environment": {
        "types": [str, type(None)],
    },
    "geoh5": {"required": True, "types": [str, Path, Workspace, type(None)]},
    "monitoring_directory": {
        "types": [str, Path, type(None)],
    },
    "run_command": {
        "types": [str, type(None)],
    },
    "workspace_geoh5": {
        "types": [str, Path, Workspace, type(None)],
    },
    "out_group": {
        "types": [str, UIJsonGroup, type(None)],
    },
}
