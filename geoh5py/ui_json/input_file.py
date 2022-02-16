#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from geoh5py.groups import ContainerGroup, PropertyGroup
from geoh5py.io.utils import as_str_if_uuid, entity2uuid, str2uuid, uuid2entity
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import BaseValidationError, JSONParameterValidationError
from geoh5py.shared.validators import UUIDValidator
from geoh5py.workspace import Workspace

from .constants import base_validations, ui_validations
from .validation import InputValidation


class InputFile:
    """
    Handles loading ui.json input files.

    Attributes
    ----------
    data : Input file content parsed to flat dictionary of key:value.
    ui_json: User interface serializable as ui.json format
    workspace: Target :obj:`geoh5py.workspace.Workspace`
    validations: Dictionary of validations for parameters in the input file

    Methods
    -------
    write_ui_json()
        Writes a ui.json formatted file from 'data' attribute contents.
    read_ui_json()
        Reads a ui.json formatted file into 'data' attribute dictionary.  Optionally filters
        ui.json fields other than 'value'.
    """

    _ui_validators = None
    _validators = None
    uuid_validator = UUIDValidator()

    def __init__(
        self,
        data: dict[str, Any] = None,
        ui_json: dict[str, Any] = None,
        validations: dict = None,
        validation_options: dict = None,
        workspace: Workspace = None,
    ):
        self._validation_options = validation_options
        self.validations = validations
        self.ui_json = ui_json
        self.data = data
        self.workspace = workspace

    @property
    def data(self):
        if (
            getattr(self, "_data", None) is None
            and getattr(self, "_ui_json", None) is not None
        ):
            self.data = self.flatten(self.ui_json)

        return self._data

    @data.setter
    def data(self, value: dict[str, Any]):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("Input 'data' must be of type dict or None.")

            if "geoh5" in value:
                self.workspace = value["geoh5"]

            value = self._promote(value)

            if self.validators is not None:
                self.validators.validate_data(value)

        self._data = value

    @property
    def ui_json(self) -> dict | None:
        """
        Dictionary representing the ui.json file with promoted values.
        """
        return self._ui_json

    @ui_json.setter
    def ui_json(self, value: dict[str, Any]):
        if value is not None:

            if not isinstance(value, dict):
                raise ValueError("Input 'ui_json' must be of type dict or None.")

            self._ui_json = self._numify(value)
            default_validations = InputValidation.infer_validations(self._ui_json)
            for key, validations in default_validations.items():
                if key in self.validations:
                    validations = {**validations, **self.validations[key]}
                self._validations[key] = validations
        else:
            self._ui_json = None
        self._validators = None

    @staticmethod
    def read_ui_json(json_file: str):
        """
        Read and create an InputFile from *.ui.json
        """
        input_file = InputFile()

        if "ui.json" not in json_file:
            raise ValueError("Input file should have the extension *.ui.json")

        with open(json_file, encoding="utf-8") as file:
            input_file.load(json.load(file))

        return input_file

    def load(self, input_dict: dict[str, Any]):
        """Load data from dictionary and validate."""
        self.ui_json = input_dict
        self.data = InputFile.flatten(input_dict)

    @property
    def validation_options(self):
        """Pass validation options to the validators."""
        if self._validation_options is None:
            return {}
        return self._validation_options

    @property
    def validations(self):
        if getattr(self, "_validations", None) is None:
            self._validations = deepcopy(base_validations)

        return self._validations

    @validations.setter
    def validations(self, valid_dict: dict | None):
        if not isinstance(valid_dict, (dict, type(None))):
            raise TypeError(
                "Input validations must be of type 'dict' or None. "
                f"Value type {type(valid_dict)} provided"
            )

        if valid_dict is not None:
            valid_dict = {**valid_dict, **deepcopy(base_validations)}

        self._validations = valid_dict

    @property
    def validators(self):
        if getattr(self, "_validators", None) is None:
            self._validators = InputValidation(
                ui_json=self.ui_json,
                validations=self.validations,
                **self.validation_options,
            )

        return self._validators

    @property
    def ui_validators(self):
        if getattr(self, "_ui_validators", None) is None:
            self._ui_validators = InputValidation(
                validations=ui_validations,
                ignore_list=("value",),
                **self.validation_options,
            )

        return self._ui_validators

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: Workspace | None):
        if workspace is not None and not isinstance(workspace, Workspace):
            raise ValueError(
                "Input 'workspace' must be a valid :obj:`geoh5py.workspace.Workspace`"
            )

        self._workspace = workspace

        if self.validators is not None:
            self.validators.workspace = workspace

    def write_ui_json(
        self,
        name: str = None,
        none_map: dict[str, Any] = None,
        path: str = None,
    ) -> str | None:
        """
        Writes a ui.json formatted file from InputFile data
        :param name: Name of the file
        :param none_map : Map parameter None values to non-null numeric types.
            The parameters in the
            dictionary will also be map optional and disabled, ensuring that if not
            updated by the user they would read back as None.
        :param path: Directory to write the ui.json to.
        """
        if self.ui_json is None or self.data is None:
            raise AttributeError(
                "The input file requires 'ui_json' and 'data' to be set before writing out."
            )

        for key, value in self.data.items():
            msg = f"Overriding param: {key} 'None' value to zero since 'dataType' is 'Float'."
            if isinstance(self.ui_json[key], dict):
                field = "value"
                if "isValue" in self.ui_json[key]:
                    if not self.ui_json[key]["isValue"]:
                        field = "property"
                    elif (
                        ("dataType" in self.ui_json[key])
                        and (self.ui_json[key]["dataType"] == "Float")
                        and (value is None)
                    ):
                        value = 0.0
                        warnings.warn(msg)

                self.ui_json[key][field] = value
                if value is not None:
                    self.ui_json[key]["enabled"] = True
            else:
                self.ui_json[key] = value

        if path is None:
            path = os.path.dirname(self.workspace.h5file)

        if name is None:
            name = self.ui_json["title"]

        if ".ui.json" not in name:
            name += ".ui.json"

        with open(os.path.join(path, name), "w", encoding="utf-8") as file:
            json.dump(
                self._stringify(self._demote(self.ui_json), none_map), file, indent=4
            )

        return os.path.join(path, name)

    def _stringify(
        self, var: dict[str, Any], none_map: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Convert inf, none, and list types to strings within a dictionary

        :param var: Dictionary containing ui.json keys, values, fields

        :return: Dictionary with inf, none and list types converted to string
            representations in json format.
        """
        for key, value in var.items():
            # Handle special cases of None values

            if (
                isinstance(value, dict)
                and "property" in value
                and value["property"] is None
            ):
                value["property"] = ""

            if isinstance(value, dict) and value["value"] is None:
                if none_map is not None and key in none_map:
                    value["value"] = none_map[key]
                    if "group" in value:
                        if InputFile.group_optional(var, value["group"]):
                            value["enabled"] = False
                        else:
                            value["optional"] = True
                            value["enabled"] = False
                    else:
                        value["optional"] = True
                        value["enabled"] = False
                elif "meshType" in value:
                    value["value"] = ""
                elif "isValue" in value and value["isValue"]:
                    value["isValue"] = False
                    value["property"] = ""
                    value["value"] = 0.0

            exclude = ["choiceList", "meshType", "dataType", "association"]
            mappers = (
                [list2str, inf2str, as_str_if_uuid, none2str]
                if key not in exclude
                else [inf2str, as_str_if_uuid, none2str]
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
                try:
                    self.ui_validators(value)
                except tuple(BaseValidationError.__subclasses__()) as error:
                    raise JSONParameterValidationError(key, error.args[0]) from error

                value = self._numify(value)

            mappers = (
                [str2none, str2inf, str2uuid, path2workspace]
                if key == "ignore_values"
                else [str2list, str2none, str2inf, str2uuid, path2workspace]
            )
            var[key] = self._dict_mapper(value, mappers)

        return var

    def _demote(self, var: dict[str, Any]) -> dict[str, str]:
        """Converts promoted parameter values to their string representations."""
        mappers = [entity2uuid, as_str_if_uuid, workspace2path, container_group2name]
        for key, value in var.items():

            if isinstance(value, dict):
                var[key] = self._demote(value)
            elif isinstance(value, (list, tuple)):
                var[key] = [self._dict_mapper(val, mappers) for val in value]
            else:
                var[key] = self._dict_mapper(value, mappers)

        return var

    def _promote(self, var: dict[str, Any]) -> dict[str, Any]:
        """Convert uuids to entities from the workspace."""
        for key, value in var.items():

            if isinstance(value, dict):
                var[key] = self._promote(value)
            elif isinstance(value, UUID):
                self.uuid_validator(key, value, self.workspace)
                var[key] = uuid2entity(value, self.workspace)

        return var

    @staticmethod
    def _dict_mapper(val, string_funcs: list[Callable], *args) -> dict:
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
            if args is None:
                val = fun(val)
            else:
                val = fun(val, *args)
        return val

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
