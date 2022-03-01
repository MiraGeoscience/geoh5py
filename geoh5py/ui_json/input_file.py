#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any
from uuid import UUID

from geoh5py.io.utils import as_str_if_uuid, entity2uuid, str2uuid, uuid2entity
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import BaseValidationError, JSONParameterValidationError
from geoh5py.shared.validators import AssociationValidator
from geoh5py.workspace import Workspace

from .constants import base_validations, ui_validations
from .utils import (
    container_group2name,
    dict_mapper,
    flatten,
    inf2str,
    list2str,
    none2str,
    optional_type,
    path2workspace,
    set_enabled,
    str2inf,
    str2list,
    str2none,
    workspace2path,
)
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
    association_validator = AssociationValidator()

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
            self.data = flatten(self.ui_json)

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

    def load(self, input_dict: dict[str, Any]):
        """Load data from dictionary and validate."""
        self.ui_json = input_dict
        self.data = flatten(input_dict)

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

    @property
    def ui_validators(self):
        if getattr(self, "_ui_validators", None) is None:

            self._ui_validators = InputValidation(
                validations=ui_validations, **{"ignore_list": ("value",)}
            )

        return self._ui_validators

    def update_ui_values(self, data: dict, none_map=None):
        """
        Update the ui.json values and enabled status from input data.

        :param ui_json: A ui.json formatted dictionary.
        :param data: Key and value pairs expected by the ui_json.
        :param none_map : Map parameter 'None' values to non-null numeric types.
            The parameters in the dictionary are mapped to optional and disabled.
        """
        if self.ui_json is None:
            raise UserWarning("InputFile requires a 'ui_json' to be defined.")

        if none_map is None:
            none_map = {}

        error_list = []
        for key, value in data.items():
            if isinstance(self.ui_json[key], dict):
                if value is None:
                    if not optional_type(self.ui_json, key):
                        error_list.append(key)
                        continue

                    value = none_map.get(key, None)
                    enabled = False
                else:
                    enabled = True

                set_enabled(self.ui_json, key, enabled)
                field = "value"
                if "isValue" in self.ui_json[key]:
                    if isinstance(value, (Entity, UUID)):
                        self.ui_json[key]["isValue"] = False
                        field = "property"
                    else:
                        self.ui_json[key]["isValue"] = True

                self.ui_json[key][field] = value

            else:
                self.ui_json[key] = value

        if any(error_list):
            raise ValueError(
                f"The following parameters are not optional. Assign value for: {error_list}"
            )

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
        Writes a formatted ui.json file from InputFile data

        :param name: Name of the file
        :param none_map : Map parameter None values to non-null numeric types.
        :param path: Directory to write the ui.json to.
        """
        if self.ui_json is None or self.data is None:
            raise AttributeError(
                "The input file requires 'ui_json' and 'data' to be set before writing out."
            )

        self.update_ui_values(self.data, none_map=none_map)

        if path is None:
            path = os.path.dirname(self.workspace.h5file)

        if name is None:
            name = self.ui_json["title"]

        if ".ui.json" not in name:
            name += ".ui.json"

        with open(os.path.join(path, name), "w", encoding="utf-8") as file:
            json.dump(self._stringify(self._demote(self.ui_json)), file, indent=4)

        return os.path.join(path, name)

    @staticmethod
    def _stringify(var: dict[str, Any]) -> dict[str, Any]:
        """
        Convert inf, none, and list types to strings within a dictionary

        :param var: Dictionary containing ui.json keys, values, fields

        :return: Dictionary with inf, none and list types converted to string
            representations in json format.
        """
        for key, value in var.items():
            exclude = ["choiceList", "meshType", "dataType", "association"]
            mappers = (
                [list2str, inf2str, as_str_if_uuid, none2str]
                if key not in exclude
                else [inf2str, as_str_if_uuid, none2str]
            )
            var[key] = dict_mapper(
                value, mappers, omit={ex: [list2str] for ex in exclude}
            )

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
            var[key] = dict_mapper(value, mappers)

        return var

    def _demote(self, var: dict[str, Any]) -> dict[str, str]:
        """Converts promoted parameter values to their string representations."""
        mappers = [entity2uuid, as_str_if_uuid, workspace2path, container_group2name]
        for key, value in var.items():

            if isinstance(value, dict):
                var[key] = self._demote(value)
            elif isinstance(value, (list, tuple)):
                var[key] = [dict_mapper(val, mappers) for val in value]
            else:
                var[key] = dict_mapper(value, mappers)

        return var

    def _promote(self, var: dict[str, Any]) -> dict[str, Any]:
        """Convert uuids to entities from the workspace."""
        for key, value in var.items():

            if isinstance(value, dict):
                var[key] = self._promote(value)
            elif isinstance(value, UUID):
                self.association_validator(key, value, self.workspace)
                var[key] = uuid2entity(value, self.workspace)

        return var
