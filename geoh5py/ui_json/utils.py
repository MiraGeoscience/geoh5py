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

from __future__ import annotations

import warnings
from os import mkdir, path
from shutil import move
from time import time
from typing import Any

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import ObjectBase
from geoh5py.workspace import Workspace


def flatten(ui_json: dict[str, dict]) -> dict[str, Any]:
    """Flattens ui.json format to simple key/value pair."""
    data: dict[str, Any] = {}
    for name, value in ui_json.items():
        if isinstance(value, dict):
            if is_form(value):
                field = "value" if truth(ui_json, name, "isValue") else "property"
                if not truth(ui_json, name, "enabled"):
                    data[name] = None
                else:
                    data[name] = value[field]
        else:
            data[name] = value

    return data


def collect(ui_json: dict[str, dict], member: str, value: Any = None) -> dict[str, Any]:
    """Collects ui parameters with common field and optional value."""

    parameters = {}
    for name, form in ui_json.items():
        if is_form(form) and member in form:
            if value is None or form[member] == value:
                parameters[name] = form

    return parameters


def find_all(ui_json: dict[str, dict], member: str, value: Any = None) -> list[str]:
    """Returns names of all collected parameters."""
    parameters = collect(ui_json, member, value)
    return list(parameters.keys())


def group_optional(ui_json: dict[str, dict], group_name: str):
    """Returns groupOptional bool for group name."""
    group = collect(ui_json, "group", group_name)
    parameters = find_all(group, "groupOptional")
    return group[parameters[0]]["groupOptional"] if parameters else False


def optional_type(ui_json: dict[str, dict], parameter: str):
    """
    Check if a ui.json parameter is optional or groupOptional

    :param ui_json: UI.json dictionary
    :param parameter: Name of parameter to check type.
    """
    is_optional = False
    if is_form(ui_json[parameter]):
        if "optional" in ui_json[parameter]:
            is_optional = ui_json[parameter]["optional"]
        elif "dependency" in ui_json[parameter]:
            is_optional = False
            if optional_type(ui_json, ui_json[parameter]["dependency"]):
                is_optional = not ui_json[ui_json[parameter]["dependency"]]["enabled"]
        elif "group" in ui_json[parameter]:
            is_optional = group_optional(ui_json, ui_json[parameter]["group"])

    return is_optional


def group_enabled(group: dict[str, dict]) -> bool:
    """Return true if groupOptional and enabled are both true."""
    parameters = find_all(group, "groupOptional")
    if not parameters:
        raise ValueError(
            "Provided group does not contain a parameter with groupOptional member."
        )
    return group[parameters[0]].get("enabled", True)


def set_enabled(ui_json: dict, parameter: str, value: bool):
    """
    Set enabled status for an optional or groupOptional parameter.

    :param ui_json: UI.json dictionary
    :param parameter: Parameter name.
    :param value: Boolean value set to parameter's enabled member.
    """
    if ui_json[parameter].get("optional", False):
        ui_json[parameter]["enabled"] = value

    is_group_optional = False
    group_name = ui_json[parameter].get("group", False)
    if group_name:
        group = collect(ui_json, "group", group_name)
        parameters = find_all(group, "groupOptional")
        if parameters:
            is_group_optional = True
            enabled_change = False
            for form in group.values():
                enabled_change |= form.get("enabled", True) != value
                form["enabled"] = value

    if (not value) and not (
        ui_json[parameter].get("optional", False) or is_group_optional
    ):
        warnings.warn(
            f"Non-option parameter '{parameter}' cannot be set to 'enabled' False "
        )

    return is_group_optional and enabled_change


def truth(ui_json: dict[str, dict], name: str, member: str) -> bool:
    """Return parameter's 'member' value with default value for non-existent members."""
    default_states = {
        "enabled": True,
        "optional": False,
        "groupOptional": False,
        "main": False,
        "isValue": True,
    }
    if member in ui_json[name]:
        return ui_json[name][member]

    if member in default_states:
        return default_states[member]

    raise ValueError(
        f"Field: {member} was not provided in ui.json and does not have a default state."
    )


def is_uijson(ui_json: dict[str, dict]):
    """Returns True if dictionary contains all the required parameters."""
    required_parameters = [
        "title",
        "monitoring_directory",
        "run_command",
        "conda_environment",
        "geoh5",
        "workspace",
    ]

    is_a_uijson = True
    for k in required_parameters:
        if k not in ui_json.keys():
            is_a_uijson = False

    return is_a_uijson


def is_form(var) -> bool:
    """Return true if dictionary 'var' contains both 'label' and 'value' members."""
    is_a_form = False
    if isinstance(var, dict):
        if all(k in var.keys() for k in ["label", "value"]):
            is_a_form = True

    return is_a_form


def list2str(value):
    if isinstance(value, list):  # & (key not in exclude):
        return str(value)[1:-1]
    return value


def none2str(value):
    if value is None:
        return ""
    return value


def inf2str(value):  # map np.inf to "inf"
    if not isinstance(value, (int, float)):
        return value
    return str(value) if not np.isfinite(value) else value


def str2list(value):  # map "[...]" to [...]
    if isinstance(value, str):
        if value in ["inf", "-inf", ""]:
            return value
        try:
            return [float(n) for n in value.split(",") if n != ""]
        except ValueError:
            return value

    return value


def str2none(value):
    if value == "":
        return None
    return value


def str2inf(value):
    if value in ["inf", "-inf"]:
        return float(value)
    return value


def workspace2path(value):
    if isinstance(value, Workspace):
        return value.h5file
    return value


def path2workspace(value):
    if isinstance(value, str) and ".geoh5" in value:
        return Workspace(value)
    return value


def container_group2name(value):
    if isinstance(value, ContainerGroup):
        return value.name
    return value


def monitored_directory_copy(
    directory: str, entity: ObjectBase, copy_children: bool = True
):
    """
    Create a temporary *.geoh5 file in the monitoring folder and export entity for update.

    :param directory: Monitoring directory
    :param entity: Entity to be updated
    :param copy_children: Option to copy children entities.
    """
    working_path = path.join(directory, ".working")
    if not path.exists(working_path):
        mkdir(working_path)

    temp_geoh5 = f"temp{time():.3f}.geoh5"
    temp_workspace = Workspace(path.join(working_path, temp_geoh5))
    entity.copy(parent=temp_workspace, copy_children=copy_children)
    move(
        path.join(working_path, temp_geoh5),
        path.join(directory, temp_geoh5),
    )

    return path.join(directory, temp_geoh5)
