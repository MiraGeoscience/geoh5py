#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import os
import warnings
from copy import deepcopy
from typing import Any, Callable
from uuid import UUID

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

from .constants import ui_validations
from .constants import validations as default_validations
from .validators import InputValidators


class InputFile:
    """
    Handles loading ui.json input files.

    Attributes
    ----------
    filepath : Path to input file.
    data : Input file content parsed to flat dictionary of key:value.
    ui_json: User interface serializable as ui.json format

    Methods
    -------
    default()
        Defaults values in 'data' attribute to those stored in default_ui 'default' fields.
    write_ui_json()
        Writes a ui.json formatted file from 'data' attribute contents.
    read_ui_json()
        Reads a ui.json formatted file into 'data' attribute dictionary.  Optionally filters
        ui.json fields other than 'value'.

    """

    def __init__(
        self,
        data: dict[str, Any] = None,
        filepath: str | None = None,
        ui_json: dict[str, Any] = None,
        validations: dict = None,
        workspace: Workspace = None,
    ):
        self.workpath: str | None = None
        self.validations = validations
        self.ui_validations = ui_validations
        self.ui_json = ui_json
        self.data = data
        self.workspace = workspace
        self.filepath = filepath
        self._initialize()

    def _initialize(self):
        """Default construction behaviour."""

        if self.filepath is not None:
            with open(self.filepath, encoding="utf-8") as file:
                data = json.load(file)
                self.load(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: dict[str, Any]):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("Input 'data' must be of type dict or None.")

            self.validators(value)

        self._data = value

    @property
    def ui_json(self):
        return self._ui_json

    @ui_json.setter
    def ui_json(self, value: dict[str, Any]):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("Input 'ui_json' must be of type dict or None.")

            self._ui_json = self._numify(value)
        else:
            self._ui_json = None

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        ifile = cls()
        ifile.load(input_dict)
        ifile.workpath = os.path.abspath(".")
        return ifile

    def load(self, input_dict: dict[str, Any]):
        """Load data from dictionary and validate."""
        self.ui_json = input_dict
        self.data = InputFile.flatten(input_dict)

    @property
    def filepath(self):
        if getattr(self, "_filepath", None) is None:

            if getattr(self, "workpath", None) is not None:
                self._filepath = os.path.join(self.workpath, "default.ui.json")

        return self._filepath

    @filepath.setter
    def filepath(self, path: str | None):
        if path is None:
            self._filepath = path
            self._workpath = None
            return
        if ".".join(path.split(".")[-2:]) != "ui.json":
            raise OSError("Input file must have 'ui.json' extension.")

        self._filepath = path
        self._workpath = None

    @property
    def validations(self):
        if getattr(self, "_validations", None) is None:
            self._validations = default_validations

        return self._validations

    @validations.setter
    def validations(self, valid_dict: dict | None):
        if valid_dict is not None:
            self._validators = InputValidators(valid_dict)
        self._validations = valid_dict

    @property
    def ui_validations(self):
        if getattr(self, "_ui_validations", None) is None:
            self._ui_validations = ui_validations

        return self._ui_validations

    @ui_validations.setter
    def ui_validations(self, valid_dict: dict | None):
        if valid_dict is not None:
            self._ui_validators = InputValidators(valid_dict)
        self._ui_validations = valid_dict

    @property
    def validators(self):
        if getattr(self, "_validators", None) is None:
            self._validators = InputValidators(self.validations)

        return self._validators

    @property
    def ui_validators(self):
        if getattr(self, "_ui_validators", None) is None:
            self._ui_validators = InputValidators(self.ui_validations)

        return self._ui_validators

    @property
    def workpath(self):
        if getattr(self, "_workpath", None) is None:
            path = None
            if getattr(self, "_filepath", None) is not None:
                path = self.filepath
            elif getattr(self, "workspace", None) is not None:
                if isinstance(self.workspace, str):
                    path = self.workspace
                else:
                    path = self.workspace.h5file

            if path is not None:
                self._workpath: str = (
                    os.path.dirname(os.path.abspath(path)) + os.path.sep
                )
        return self._workpath

    @workpath.setter
    def workpath(self, path):
        self._workpath = path

    def write_ui_json(
        self,
        default: bool = False,
        name: str = None,
        workspace: Workspace = None,
        none_map: dict[str, Any] = None,
    ) -> None:
        """
        Writes a ui.json formatted file from InputFile data
        Parameters
        ----------
        ui_dict :
            Dictionary in ui.json format, including defaults.
        default :
            Write default values. Ignoring contents of self.data.
        name: optional
            Name of the file
        workspace : optional
            Provide a geoh5 path to simulate auto-generated field in Geoscience ANALYST.
        none_map : optional
            Map parameter None values to non-null numeric types.  The parameters in the
            dictionary will also be map optional and disabled, ensuring that if not
            updated by the user they would read back as None.
        """
        out = deepcopy(self.ui_json)

        if workspace is not None:
            out["geoh5"] = workspace

        if not default and self.data is not None:
            for key, value in self.data.items():
                msg = f"Overriding param: {key} 'None' value to zero since 'dataType' is 'Float'."
                if isinstance(out[key], dict):
                    field = "value"
                    if "isValue" in out[key]:
                        if not out[key]["isValue"]:
                            field = "property"
                        elif (
                            ("dataType" in out[key])
                            and (out[key]["dataType"] == "Float")
                            and (value is None)
                        ):
                            value = 0.0
                            warnings.warn(msg)

                    out[key][field] = value
                    if value is not None:
                        out[key]["enabled"] = True
                else:
                    out[key] = value

        out_file = self.filepath
        if name is not None:
            if ".ui.json" not in name:
                name += ".ui.json"

            if self.workpath is not None:
                out_file = os.path.join(self.workpath, name)
            else:
                out_file = os.path.abspath(name)

        with open(out_file, "w", encoding="utf-8") as file:
            json.dump(self._stringify(self._demote(out), none_map), file, indent=4)

    def _stringify(
        self, var: dict[str, Any], none_map: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Convert inf, none, and list types to strings within a dictionary

        Parameters
        ----------

        d :
            dictionary containing ui.json keys, values, fields

        Returns
        -------
        Dictionary with inf, none and list types converted to string representations friendly for
        json format.

        """
        for key, value in var.items():
            # Handle special cases of None values

            if isinstance(value, dict) and value["value"] is None:
                if none_map is not None and key in none_map:
                    value["value"] = none_map[key]
                    if "group" in value.keys():
                        if InputFile.group_optional(var, value["group"]):
                            value["enabled"] = False
                        else:
                            value["optional"] = True
                            value["enabled"] = False
                    else:
                        value["optional"] = True
                        value["enabled"] = False
                elif "meshType" in value.keys():
                    value["value"] = ""
                elif "isValue" in value.keys() and value["isValue"]:
                    value["isValue"] = False
                    value["property"] = ""
                    value["value"] = 0.0

            exclude = ["choiceList", "meshType", "dataType", "association"]
            mappers = (
                [list2str, inf2str, uuid2str, none2str]
                if key not in exclude
                else [inf2str, uuid2str, none2str]
            )
            var[key] = self._dict_mapper(value, mappers)

        return var

    def _numify(self, var: dict[str, Any]) -> dict[str, Any]:
        """
        Convert inf, none and list strings to numerical types within a dictionary

        Parameters
        ----------

        var :
            dictionary containing ui.json keys, values, fields

        Returns
        -------
        Dictionary with inf, none and list string representations converted numerical types.

        """
        for key, value in var.items():
            if isinstance(value, dict):
                self.ui_validators(value)

            mappers = (
                [str2none, str2inf, str2uuid]
                if key == "ignore_values"
                else [str2list, str2none, str2inf, str2uuid]
            )
            var[key] = self._dict_mapper(value, mappers)

        return var

    def _demote(self, var: dict[str, Any]) -> dict[str, str]:
        """Converts promoted parameter values to their string representations."""
        mappers = [uuid2str, workspace2path, containergroup2name]
        for key, value in var.items():
            var[key] = self._dict_mapper(value, mappers)

        return var

    @staticmethod
    def _dict_mapper(val, string_funcs: list[Callable]) -> dict:
        """
        Recurses through nested dictionary and applies mapping funcs to all values

        Parameters
        ----------
        val :
            Dictionary val (could be another dictionary).
        string_funcs:
            Function to apply to values within dictionary.
        """
        for fun in string_funcs:
            val = fun(val)
        return val

    # @staticmethod
    # def get_associations(var: dict[str, Any]) -> dict:
    #     """
    #     Get parent/child associations for ui.json fields.
    #
    #     :param var: Dictionary containing ui.json keys/values/fields.
    #     """
    #     associations = {}
    #     for key, value in var.items():
    #         if isinstance(value, dict):
    #             field = InputFile.field(value)
    #             if "parent" in value.keys() and value["parent"] is not None:
    #                 try:
    #                     associations[key] = value["parent"]
    #                     try:
    #                         child_key = UUID(value[field])
    #                     except (ValueError, TypeError):
    #                         child_key = value[field]
    #                     parent_uuid = UUID(var[value["parent"]]["value"])
    #                     associations[child_key] = parent_uuid
    #                 except (ValueError, TypeError):
    #                     continue
    #         else:
    #             continue
    #
    #     #     return associations

    @staticmethod
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

    @staticmethod
    def collect(var: dict[str, Any], field: str, value: Any = None) -> dict[str, Any]:
        """Collects ui parameters with common field and optional value."""
        data = {}
        for key, values in var.items():
            if isinstance(values, dict) and field in values:
                # TODO Check with Ben if value is None makes sense
                if value is None or values[field] == value:
                    data[key] = values
        return data

    @staticmethod
    def group(var: dict[str, Any], name: str) -> dict[str, Any]:
        """Retrieves ui elements with common group name."""
        return InputFile.collect(var, "group", name)

    @staticmethod
    def group_optional(var: dict[str, Any], name: str) -> bool:
        """Returns groupOptional bool for group name."""
        group = InputFile.group(var, name)
        param = InputFile.collect(group, "groupOptional")
        return list(param.values())[0]["groupOptional"] if param is not None else False

    @staticmethod
    def group_enabled(var: dict[str, Any], name: str) -> bool:
        """Returns enabled status of member of group containing groupOptional:True field."""
        group = InputFile.group(var, name)
        if InputFile.group_optional(group, name):
            param = InputFile.collect(group, "groupOptional")
            return list(param.values())[0]["enabled"]

        return True


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


def is_uuid(value):
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


# def field(var: dict[str, Any]) -> str:
#     """Returns field in ui_json block that contains data ('value' or 'property')."""
#
#     if "isValue" in var.keys():
#         return "value" if var["isValue"] else "property"
#
#     return "value"


def list2str(value):
    if isinstance(value, list):  # & (key not in exclude):
        return str(value)[1:-1]
    return value


def uuid2str(value):
    if isinstance(value, UUID):
        return str(value)
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


def str2uuid(value):
    if is_uuid(value):
        return UUID(str(value))
    return value


#
# def uuid2str(value):
#     if isinstance(value, UUID):
#         return f"{{{str(value)}}}"
#     return value


def workspace2path(value):
    if isinstance(value, Workspace):
        return value.h5file
    return value


def containergroup2name(value):
    if isinstance(value, ContainerGroup):
        return value.name
    return value
