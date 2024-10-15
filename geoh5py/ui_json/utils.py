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

import warnings
from io import BytesIO
from pathlib import Path
from shutil import copy, move
from time import time
from typing import Any

from geoh5py import Workspace
from geoh5py.groups import ContainerGroup, Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace


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


def group_optional(ui_json: dict[str, dict], group_name: str) -> bool:
    """Returns groupOptional bool for group name."""
    group = collect(ui_json, "group", group_name)
    parameters = find_all(group, "groupOptional")
    return group[parameters[0]]["groupOptional"] if parameters else False


def optional_requires_value(ui_json: dict[str, dict], parameter: str) -> bool:
    """
    True if enabled else False.

    :param ui_json: UI.json dictionary
    :param parameter: Name of parameter to check type.
    """
    return ui_json[parameter].get("enabled", True)


def dependency_requires_value(ui_json: dict[str, dict], parameter: str) -> bool:
    """
    Handles dependency and optional requirements.

    If dependency doesn't require a value then the function returns False. But
    if the dependency does require a value, the return value is either True,
    or will take on the enabled state if the dependent parameter is optional.

    :param ui_json: UI.json dictionary
    :param parameter: Name of parameter to check type.
    """
    dependency = ui_json[parameter]["dependency"]

    key = (
        "enabled"
        if (
            ui_json[dependency].get("optional", False)
            or "enabled" in ui_json[dependency]
        )
        else "value"
    )

    if ui_json[parameter].get("dependencyType", "enabled") == "enabled":
        is_required = bool(ui_json[dependency].get(key, True)) & bool(
            ui_json[dependency]["value"]
        )
    else:
        is_required = not ui_json[dependency].get(key, True)
    if ("optional" in ui_json[parameter]) & bool(is_required):
        is_required = ui_json[parameter]["enabled"]

    return is_required


def group_requires_value(ui_json: dict[str, dict], parameter: str) -> bool:
    """
    True is groupOptional and group is enabled else False

    :param ui_json: UI.json dictionary
    :param parameter: Name of parameter to check type.
    """
    is_required = True
    groupname = ui_json[parameter]["group"]
    group = collect(ui_json, "group", groupname)
    if group_optional(ui_json, groupname):
        is_required = group_enabled(group)
    return is_required


def requires_value(ui_json: dict[str, dict], parameter: str) -> bool:
    """
    Check if a ui.json parameter requires a value (is not optional).

    The required status of a parameter depends on a hierarchy of ui switches.
    At the top is the groupOptional switch, below that is the dependency
    switch, and on the bottom is the optional switch.  When group optional
    is disabled all parameters in the group are not required, When the
    groupOptional is enabled the required status of a parameter depends first
    any dependencies and lastly on it's optional status.

    :param ui_json: UI.json dictionary
    :param parameter: Name of parameter to check type.
    """
    is_required = True

    if is_form(ui_json[parameter]):
        if "group" in ui_json[parameter]:
            group_required = group_requires_value(ui_json, parameter)
            if group_required:
                if "dependency" in ui_json[parameter]:
                    is_required = dependency_requires_value(ui_json, parameter)
                elif "optional" in ui_json[parameter]:
                    is_required = optional_requires_value(ui_json, parameter)
            else:
                is_required = False

        elif "dependency" in ui_json[parameter]:
            is_required = dependency_requires_value(ui_json, parameter)

        elif "optional" in ui_json[parameter]:
            is_required = optional_requires_value(ui_json, parameter)

    return is_required


def group_enabled(group: dict[str, dict]) -> bool:
    """
    Return true if groupOptional and enabled are both true.

    :param group: UI.json dictionary
    """
    parameters = find_all(group, "groupOptional")
    if not parameters:
        raise ValueError(
            "Provided group does not contain a parameter with groupOptional member."
        )
    return group[parameters[0]].get("enabled", True)


def set_enabled(ui_json: dict, parameter: str, value: bool, validate=True):
    """
    Set enabled status for an optional or groupOptional parameter.

    :param ui_json: UI.json dictionary
    :param parameter: Parameter name.
    :param value: Boolean value set to parameter's enabled member.
    """
    if ui_json[parameter].get("optional", False) or bool(
        ui_json[parameter].get("dependency", False)
    ):
        ui_json[parameter]["enabled"] = value

    is_group_optional = False
    group_name = ui_json[parameter].get("group", False)
    if group_name:
        group = collect(ui_json, "group", group_name)
        parameters = find_all(group, "groupOptional")
        if parameters:
            is_group_optional = True
            if parameters[0] == parameter:
                for form in group.values():
                    form["enabled"] = value

    if validate:
        if not is_group_optional and "dependency" in ui_json[parameter]:
            is_group_optional = not dependency_requires_value(ui_json, parameter)

        if (not value) and not (
            ui_json[parameter].get("optional", False) or is_group_optional
        ):
            warnings.warn(
                f"Non-option parameter '{parameter}' cannot be set to 'enabled' False "
            )


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
        "workspace_geoh5",
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


def str2list(value):  # map "[...]" to [...]
    if isinstance(value, str):
        if value in ["inf", "-inf", ""]:
            return value
        try:
            return [float(n) for n in value.split(",") if n != ""]
        except ValueError:
            return value

    return value


def str2inf(value):
    if value in ["inf", "-inf"]:
        return float(value)
    return value


def workspace2path(value):
    if isinstance(value, Workspace):
        if isinstance(value.h5file, BytesIO):
            return "[in-memory]"
        return str(value.h5file)
    return value


def path2workspace(value):
    if isinstance(value, (str, Path)) and Path(value).suffix == ".geoh5":
        workspace = Workspace(value, mode="r")
        workspace.close()
        return workspace
    return value


def container_group2name(value):
    if isinstance(value, ContainerGroup):
        return value.name
    return value


def monitored_directory_copy(
    directory: str, entity: ObjectBase | Group, copy_children: bool = True
) -> str:
    """
    Create a temporary geoh5 file in the monitoring folder and export entity for update.

    :param directory: Monitoring directory
    :param entity: Entity to be updated
    :param copy_children: Option to copy children entities.
    """
    directory_path = Path(directory)
    working_path = directory_path / ".working"
    working_path.mkdir(exist_ok=True)

    temp_geoh5 = f"temp{time():.3f}.geoh5"

    with fetch_active_workspace(entity.workspace, mode="r"):
        with Workspace.create(working_path / temp_geoh5) as w_s:
            entity.copy(parent=w_s, copy_children=copy_children)

    move(working_path / temp_geoh5, directory_path / temp_geoh5, copy)

    return str(directory_path / temp_geoh5)
