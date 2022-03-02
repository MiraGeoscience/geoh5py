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
from typing import Any, Callable

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace


def dict_mapper(
    val, string_funcs: list[Callable], *args, omit: dict | None = None
) -> dict:
    """
    Recurses through nested dictionary and applies mapping funcs to all values

    Parameters
    ----------
    val :
        Dictionary val (could be another dictionary).
    string_funcs:
        Function to apply to values within dictionary.
    omit: Dictionary of functions to omit.
    """
    if omit is None:
        omit = {}
    if isinstance(val, dict):
        for key, values in val.items():
            val[key] = dict_mapper(
                values,
                [fun for fun in string_funcs if fun not in omit.get(key, [])],
            )

    for fun in string_funcs:
        if args is None:
            val = fun(val)
        else:
            val = fun(val, *args)
    return val


def flatten(var: dict[str, Any]) -> dict[str, Any]:
    """Flattens ui.json format to simple key/value pair."""
    data: dict = {}
    for key, value in var.items():
        if isinstance(value, dict):
            if is_uijson({key: value}):
                field = "value" if truth(var, key, "isValue") else "property"
                if not truth(var, key, "enabled"):
                    data[key] = None
                else:
                    data[key] = value[field]
        else:
            data[key] = value

    return data


def collect(var: dict[str, Any], field: str, value: Any = None) -> dict[str, Any]:
    """Collects ui parameters with common field and optional value."""
    data = {}
    for key, values in var.items():
        if isinstance(values, dict) and field in values:
            if values[field] == value:
                data[key] = values
    return data


def group_optional(
    var: dict[str, Any], name: str | dict, return_lead: bool = False
) -> bool | str:
    """Returns groupOptional bool for group name."""
    if isinstance(name, dict):
        if "group" in name:
            name = name["group"]
        else:
            return False

    group = collect(var, "group", name)
    param = collect(group, "groupOptional", True)
    if return_lead and any(param):
        return list(param.keys())[0]
    return any(param)


def optional_type(ui_json: dict, name: str):
    """
    Check if a ui.json parameter is optional or groupOptional

    :param ui_json: UI.json dictionary
    :param name: Name of parameter to check type.
    """
    if ui_json[name].get("optional") or group_optional(ui_json, ui_json[name]):
        return True

    return False


def set_enabled(ui_json: dict, name: str, value: bool):
    """
    Set enabled status for an optional or groupOptional parameter.

    :param ui_json: UI.json dictionary
    :param name: Name of the parameter to check optional on.
    :param value: Set enable True or False.
    """
    if not optional_type(ui_json, name) and not value:
        raise UserWarning(
            f"Parameter '{name}' is not optional or groupOptional. A value is required."
        )

    ui_json[name]["enabled"] = value
    optional_group = group_optional(ui_json, ui_json[name], return_lead=True)

    if optional_group:
        if ui_json[optional_group]["enabled"] and not value:
            warnings.warn(
                f"The ui.json group {ui_json[name]['group']} was disabled "
                f"due to parameter '{name}'."
            )
        ui_json[optional_group]["enabled"] = value


def truth(var: dict[str, Any], name: str, field: str) -> bool:
    default_states = {
        "enabled": True,
        "optional": False,
        "groupOptional": False,
        "main": False,
        "isValue": True,
    }
    if field in var[name]:
        return var[name][field]

    if field in default_states:
        return default_states[field]

    raise ValueError(
        f"Field: {field} was not provided in ui.json and does not have a default state."
    )


def is_uijson(var):
    uijson_keys = [
        "title",
        "monitoring_directory",
        "run_command",
        "conda_environment",
        "geoh5",
        "workspace_geoh5",
    ]
    uijson = True
    if len(var.keys()) > 1:
        for k in uijson_keys:
            if k not in var.keys():
                uijson = False

    for value in var.values():
        if isinstance(value, dict):
            for name in ["label", "value"]:
                if name not in value.keys():
                    uijson = False

    return uijson


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
