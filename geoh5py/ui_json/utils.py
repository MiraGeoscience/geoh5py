# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

import warnings
from io import BytesIO
from logging import getLogger
from pathlib import Path
from shutil import copy, move
from time import time
from typing import Any

from geoh5py import Workspace
from geoh5py.groups import ContainerGroup, Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace


logger = getLogger(__name__)


def flatten(ui_json: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Flatten ui.json format to simple key/value pairs.

    Converts the nested ui.json structure to a flat dictionary where most fields
    have their values extracted from the 'value' member. For widgets storing
    multiple values, the values are passed as a dictionary.

    :param ui_json: The form containing the UI data to flatten.

    :return: A flattened dictionary with parameter names as keys and their values.
    """
    data: dict[str, Any] = {}
    for name, value in ui_json.items():
        if isinstance(value, dict):
            if is_form(value):
                field = "value" if truth(ui_json, name, "isValue") else "property"
                if not truth(ui_json, name, "enabled"):
                    data[name] = None
                elif (
                    "groupValue" in value
                    and "value" in value
                    and value["groupValue"] is not None
                ):
                    data[name] = {
                        "groupValue": value["groupValue"],
                        "value": value["value"],
                    }
                elif (
                    "isComplement" in value
                    and "property" in value
                    and value["property"] is not None
                ):
                    data[name] = {
                        "value": value["value"],
                        "isComplement": value["isComplement"],
                        "property": value["property"],
                    }
                else:
                    data[name] = value[field]

        else:
            data[name] = value

    return data


def collect(
    ui_json: dict[str, dict[str, Any]], member: str, value: Any = None
) -> dict[str, dict[str, Any]]:
    """
    Collect UI parameters with a common field and optional value.

    Searches through the ui_json dictionary for parameters that contain a
    specific member field, optionally filtering by the value of that field.

    :param ui_json: The UI JSON dictionary to search through.
    :param member: The field name to search for in each parameter.
    :param value: Optional value to match against the member field.

    :return: Dictionary of parameters that match the criteria.
    """

    parameters = {}
    for name, form in ui_json.items():
        if is_form(form) and member in form:
            if value is None or form[member] == value:
                parameters[name] = form

    return parameters


def find_all(
    ui_json: dict[str, dict[str, Any]], member: str, value: Any = None
) -> list[str]:
    """
    Return names of all parameters matching the given criteria.

    Convenience function that returns just the parameter names from the collect
    function results.

    :param ui_json: The UI JSON dictionary to search through.
    :param member: The field name to search for in each parameter.
    :param value: Optional value to match against the member field.

    :return: List of parameter names that match the criteria.
    """
    parameters = collect(ui_json, member, value)
    return list(parameters.keys())


def group_optional(ui_json: dict[str, dict[str, Any]], group_name: str) -> bool:
    """
    Check if a group has the groupOptional flag enabled.

    Searches for parameters belonging to the specified group and returns the
    groupOptional boolean value for that group.

    :param ui_json: The UI JSON dictionary to search through.
    :param group_name: Name of the group to check.

    :return: True if the group is optional, False otherwise.
    """
    group = collect(ui_json, "group", group_name)
    parameters = find_all(group, "groupOptional")
    return group[parameters[0]]["groupOptional"] if parameters else False


def optional_requires_value(ui_json: dict[str, dict[str, Any]], parameter: str) -> bool:
    """
    Check if an optional parameter requires a value based on its enabled state.

    For optional parameters, this function checks the 'enabled' field to determine
    if the parameter requires a value.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param parameter: Name of the parameter to check.

    :return: True if the parameter is enabled, False otherwise.
    """
    return ui_json[parameter].get("enabled", True)


def dependency_requires_value(
    ui_json: dict[str, dict[str, Any]], parameter: str
) -> bool:
    """
    Check if a parameter requires a value based on its dependency constraints.

    Handles dependency and optional requirements for parameters. If a dependency
    doesn't require a value then the function returns False. If the dependency
    does require a value, the return value is either True, or will take on the
    enabled state if the dependent parameter is optional.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param parameter: Name of the parameter to check.

    :return: True if the parameter requires a value based on dependencies, False otherwise.
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


def group_requires_value(ui_json: dict[str, dict[str, Any]], parameter: str) -> bool:
    """
    Check if a parameter requires a value based on its group's optional status.

    Determines if a parameter requires a value by checking if its group is optional
    and if the group is enabled.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param parameter: Name of the parameter to check.

    :return: True if the group is optional and enabled, or if the group is not optional.
    """
    is_required = True
    groupname = ui_json[parameter]["group"]
    group = collect(ui_json, "group", groupname)
    if group_optional(ui_json, groupname):
        is_required = group_enabled(group)
    return is_required


def requires_value(ui_json: dict[str, dict[str, Any]], parameter: str) -> bool:
    """
    Check if a UI JSON parameter requires a value based on hierarchy of switches.

    The required status of a parameter depends on a hierarchy of UI switches.
    At the top is the groupOptional switch, below that is the dependency
    switch, and at the bottom is the optional switch. When groupOptional
    is disabled, all parameters in the group are not required. When the
    groupOptional is enabled, the required status of a parameter depends first
    on any dependencies and lastly on its optional status.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param parameter: Name of the parameter to check.

    :return: True if the parameter requires a value, False otherwise.
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


def group_enabled(group: dict[str, dict[str, Any]]) -> bool:
    """
    Check if a group is enabled based on groupOptional and enabled flags.

    Returns True if both groupOptional and enabled are True for the group.

    :param group: UI JSON group dictionary containing parameters.

    :return: True if the group is enabled, False otherwise.

    :raises ValueError: If the provided group does not contain a
        parameter with groupOptional member.
    """
    parameters = find_all(group, "groupOptional")
    if not parameters:
        raise ValueError(
            "Provided group does not contain a parameter with groupOptional member."
        )
    return group[parameters[0]].get("enabled", True)


def set_enabled(
    ui_json: dict[str, Any], parameter: str, value: bool, validate: bool = True
) -> None:
    """
    Set the enabled status for an optional or groupOptional parameter.

    Updates the 'enabled' field for parameters that support it, including optional
    parameters, parameters with dependencies, and group optional parameters.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param parameter: Name of the parameter to modify.
    :param value: Boolean value to set for the parameter's enabled member.
    :param validate: Whether to perform validation checks on the operation.

    :raises Warning: If attempting to disable a non-optional parameter.
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


def truth(ui_json: dict[str, dict[str, Any]], name: str, member: str) -> bool:
    """
    Get a parameter's member value with default fallback for non-existent members.

    Returns the value of a specific member field for a parameter, or a default
    value if the member doesn't exist.

    :param ui_json: UI JSON dictionary containing the parameters.
    :param name: Name of the parameter to check.
    :param member: Name of the member field to retrieve.

    :return: The member's value or its default value.

    :raises ValueError: If the field was not provided and does not have a default state.
    """
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


def is_uijson(ui_json: dict[str, dict[str, Any]]) -> bool:
    """
    Check if a dictionary contains all the required UI JSON parameters.

    Validates that a dictionary has the structure and required fields of a
    valid UI JSON configuration.

    :param ui_json: Dictionary to validate as UI JSON.

    :return: True if the dictionary contains all required parameters, False otherwise.
    """
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


def is_form(var: Any) -> bool:
    """
    Check if a variable is a valid form dictionary.

    Returns True if the dictionary variable contains both 'label' and 'value' members,
    which are required for a valid UI form element.

    :param var: Variable to check for form structure.

    :return: True if the variable is a valid form dictionary, False otherwise.
    """
    is_a_form = False
    if isinstance(var, dict):
        if all(k in var.keys() for k in ["label", "value"]):
            is_a_form = True

    return is_a_form


def str2list(value: Any) -> Any:
    """
    Convert string representation of numbers to list of floats.

    Maps string values like "[1,2,3]" or "1,2,3" to [1.0, 2.0, 3.0].
    Handles special cases like "inf", "-inf", and empty strings.

    :param value: Value to convert, typically a string or already converted type.

    :return: List of floats if conversion successful, original value otherwise.
    """
    if isinstance(value, str):
        if value in ["inf", "-inf", ""]:
            return value
        try:
            return [float(n) for n in value.split(",") if n != ""]
        except ValueError:
            return value

    return value


def str2inf(value: Any) -> Any:
    """
    Convert string representations of infinity to float infinity values.

    Converts "inf" and "-inf" strings to their corresponding float values.

    :param value: Value to convert, typically a string.

    :return: Float infinity value if input is "inf" or "-inf", original value otherwise.
    """
    if value in ["inf", "-inf"]:
        return float(value)
    return value


def workspace2path(value: Any) -> str:
    """
    Convert a Workspace object to its file path string.

    **Deprecated:** This function has been migrated to `geoh5py.shared.utils.workspace2path`
    and will be removed in future versions.

    :param value: Workspace object or other value to convert.

    :return: File path string for Workspace objects, "[in-memory]" for in-memory workspaces,
             or the original value for non-Workspace inputs.
    """
    logger.warning(
        "Deprecation Warning - This function has been migrated to "
        "`geoh5py.shared.utils.workspace2path` and will be removed in"
        "future versions.",
    )
    if isinstance(value, Workspace):
        if isinstance(value.h5file, BytesIO):
            return "[in-memory]"
        return str(value.h5file)
    return value


def path2workspace(value: Any) -> Any:
    """
    Convert a file path string to a Workspace object if it represents a valid geoh5 file.

    Creates a Workspace object from a path string if the file exists and has a .geoh5 extension.

    :param value: File path string, Path object, or other value to convert.

    :return: Workspace object if the value represents a valid geoh5 file, original value otherwise.
    """
    if (
        isinstance(value, (str, Path))
        and Path(value).suffix == ".geoh5"
        and Path(value).exists()
    ):
        workspace = Workspace(value, mode="r")
        workspace.close()
        return workspace
    return value


def container_group2name(value: Any) -> Any:
    """
    Convert a ContainerGroup object to its name string.

    **Deprecated:** This function will be removed in future releases.

    :param value: ContainerGroup object or other value to convert.

    :return: Name string for ContainerGroup objects, original value otherwise.
    """
    logger.warning(
        "Deprecation Warning - This function will be removed in future releases."
    )
    if isinstance(value, ContainerGroup):
        return value.name
    return value


def monitored_directory_copy(
    directory: str, entity: ObjectBase | Group, copy_children: bool = True
) -> str:
    """
    Create a temporary geoh5 file in the monitoring folder and export entity for update.

    Creates a temporary workspace in the specified monitoring directory and copies
    the given entity to it. This is useful for monitoring and updating entities
    in a separate workspace environment.

    :param directory: Path to the monitoring directory where the temporary file will be created.
    :param entity: Entity (ObjectBase or Group) to be copied for monitoring.
    :param copy_children: Whether to copy children entities along with the main entity.

    :return: Full path to the created temporary geoh5 file.
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
