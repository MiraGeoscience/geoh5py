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
from typing import Any
from uuid import UUID

from geoh5py.shared import Entity
from geoh5py.shared.exceptions import BaseValidationError, JSONParameterValidationError
from geoh5py.shared.validators import AssociationValidator
from geoh5py.workspace import Workspace

from ..shared.utils import (
    as_str_if_uuid,
    dict_mapper,
    entity2uuid,
    str2uuid,
    uuid2entity,
)
from .constants import base_validations, ui_validations
from .utils import (
    container_group2name,
    flatten,
    inf2str,
    list2str,
    none2str,
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

    _path: str | None = None
    _name: str | None = None
    _ui_validators: InputValidation = InputValidation(
        validations=ui_validations,
        validation_options={"ignore_list": ("value",)},
    )
    _validators = None
    association_validator = AssociationValidator()

    def __init__(
        self,
        data: dict[str, Any] = None,
        ui_json: dict[str, Any] = None,
        validations: dict = None,
        validation_options: dict = None,
    ):
        self._workspace = None
        self._validation_options = validation_options
        self.validations = validations
        self.ui_json = ui_json
        self.data = data

        if isinstance(self.workspace, Workspace):
            self.workspace.close()

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

            if self._ui_json is None:
                raise AttributeError("'ui_json' must be set before setting data.")

            if len(value) != len(self._ui_json):
                raise ValueError(
                    "The number of input values for 'data' must "
                    "equal the number of parameters in 'ui_json'."
                )

            if self.workspace is None and "geoh5" in value:
                self.workspace = value["geoh5"]

            value = self._promote(value)

            if self.validators is not None and not self.validation_options.get(
                "disabled", False
            ):
                self.validators.validate_data(value)

        self._data = value

    @property
    def name(self) -> str | None:
        """
        Name of ui.json file.
        """
        if getattr(self, "_name", None) is None and self.ui_json is not None:
            self.name = self.ui_json["title"]

        return self._name

    @name.setter
    def name(self, name: str):
        if ".ui.json" not in name:
            name += ".ui.json"

        self._name = name

    def load(self, input_dict: dict[str, Any]):
        """Load data from dictionary and validate."""
        self.ui_json = input_dict
        self.data = flatten(self.ui_json)

    @property
    def path(self) -> str | None:
        """
        Directory for the input/output ui.json file.
        """
        if getattr(self, "_path", None) is None and self.workspace is not None:
            self.path = os.path.dirname(self.workspace.h5file)

        return self._path

    @path.setter
    def path(self, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"The specified path: '{path}' does not exist.")

        self._path = path

    @property
    def path_name(self) -> str | None:
        if self.path is not None and self.name is not None:
            return os.path.join(self.path, self.name)

        return None

    @staticmethod
    def read_ui_json(json_file: str, **kwargs):
        """
        Read and create an InputFile from ui.json
        """
        input_file = InputFile(**kwargs)
        input_file.path = os.path.dirname(os.path.abspath(json_file))
        input_file.name = os.path.basename(json_file)

        if "ui.json" not in json_file:
            raise ValueError("Input file should have the extension *.ui.json")

        with open(json_file, encoding="utf-8") as file:
            input_file.load(json.load(file))

        if isinstance(input_file.workspace, Workspace):
            input_file.workspace.close()

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

            self._ui_json = self.numify(value.copy())
            default_validations = InputValidation.infer_validations(self._ui_json)
            for key, validations in default_validations.items():
                if key in self.validations:
                    validations = {**validations, **self.validations[key]}
                self._validations[key] = validations
        else:
            self._ui_json = None
        self._validators = None

    @classmethod
    def ui_validation(cls, ui_json: dict[str, Any]):
        """Validation of the ui_json forms"""
        cls._ui_validators(ui_json)

    def update_ui_values(self, data: dict, none_map=None):
        """
        Update the ui.json values and enabled status from input data.

        :param data: Key and value pairs expected by the ui_json.
        :param none_map: Map parameter 'None' values to non-null numeric types.
            The parameters in the dictionary are mapped to optional and disabled.

        :raises UserWarning: If attempting to set None value to non-optional parameter.
        """
        if self.ui_json is None:
            raise UserWarning("InputFile requires a 'ui_json' to be defined.")

        if none_map is None:
            none_map = {}

        for key, value in data.items():
            if isinstance(self.ui_json[key], dict):
                if value is None:
                    value = none_map.get(key, None)
                    enabled = False
                else:
                    enabled = True

                was_group_enabled = set_enabled(self.ui_json, key, enabled)
                if was_group_enabled:
                    warnings.warn(
                        f"Setting all member of group: {self.ui_json[key]['group']} "
                        f"to enabled: {enabled}."
                    )

                member = "value"
                if "isValue" in self.ui_json[key]:
                    if isinstance(value, (Entity, UUID)):
                        self.ui_json[key]["isValue"] = False
                        member = "property"
                    else:
                        self.ui_json[key]["isValue"] = True

                if (value is None) and (not self.ui_json[key].get("enabled", False)):
                    continue

                self.ui_json[key][member] = value

            else:
                self.ui_json[key] = value

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
                validation_options=self.validation_options,
            )

        return self._validators

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: Workspace | None):
        if workspace is not None:
            if self._workspace is not None:
                raise UserWarning(
                    "Attribute 'workspace' already set. "
                    "Consider creating a new InputFile from arguments."
                )

            if not isinstance(workspace, Workspace):
                raise ValueError(
                    "Input 'workspace' must be a valid :obj:`geoh5py.workspace.Workspace`."
                )

        self._workspace = workspace

        if self.validators is not None:
            self.validators.workspace = workspace

    def write_ui_json(
        self,
        name: str = None,
        none_map: dict[str, Any] = None,
        path: str = None,
    ):
        """
        Writes a formatted ui.json file from InputFile data

        :param name: Name of the file
        :param none_map: Map parameter None values to non-null numeric types.
        :param path: Directory to write the ui.json to.
        """

        if name is not None:
            self.name = name

        if path is not None:
            self.path = os.path.abspath(path)

        if self.path_name is None:
            raise AttributeError(
                "The input file requires 'path' and 'name' to be set before writing out."
            )

        if self.ui_json is None:
            raise AttributeError(
                "The input file requires 'ui_json' and 'data' to be set before writing out."
            )

        if self.data is not None:
            self.update_ui_values(self.data, none_map=none_map)

        with open(self.path_name, "w", encoding="utf-8") as file:
            json.dump(self._stringify(self._demote(self.ui_json)), file, indent=4)

        return self.path_name

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

    @classmethod
    def numify(cls, ui_json: dict[str, Any]) -> dict[str, Any]:
        """
        Convert inf, none and list strings to numerical types within a dictionary

        Parameters
        ----------

        ui_json :
            dictionary containing ui.json keys, values, fields

        Returns
        -------
        Dictionary with inf, none and list string representations converted numerical types.

        """
        if not isinstance(ui_json, dict):
            raise ValueError("Input value for 'numify' must be a ui_json dictionary.")

        for key, value in ui_json.items():
            if isinstance(value, dict):
                try:
                    cls.ui_validation(value)
                except tuple(BaseValidationError.__subclasses__()) as error:
                    raise JSONParameterValidationError(key, error.args[0]) from error

                value = cls.numify(value)

            mappers = (
                [str2none, str2inf, str2uuid, path2workspace]
                if key == "ignore_values"
                else [str2list, str2none, str2inf, str2uuid, path2workspace]
            )
            ui_json[key] = dict_mapper(value, mappers)

        return ui_json

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
        if self.workspace is None:
            return var

        for key, value in var.items():

            if isinstance(value, dict):
                var[key] = self._promote(value)
            elif isinstance(value, UUID):
                self.association_validator(key, value, self.workspace)
                var[key] = uuid2entity(value, self.workspace)

        return var
