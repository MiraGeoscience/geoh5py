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

import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

from geoh5py import Workspace
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import BaseValidationError, JSONParameterValidationError
from geoh5py.shared.validators import AssociationValidator

from ..shared.utils import (
    as_str_if_uuid,
    dict_mapper,
    entity2uuid,
    fetch_active_workspace,
    str2none,
    str2uuid,
    stringify,
    uuid2entity,
)
from .constants import base_validations, ui_validations
from .utils import (
    container_group2name,
    flatten,
    path2workspace,
    set_enabled,
    str2inf,
    workspace2path,
)
from .validation import InputValidation


# pylint: disable=simplifiable-if-expression, too-many-instance-attributes


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
    _data: dict[str, Any] | None
    _ui_json: dict[str, Any] | None
    _ui_validators: InputValidation = InputValidation(
        validations=ui_validations,
        validation_options={"ignore_list": ("value",)},
    )
    _validate = True
    _validators = None
    _validations: dict | None
    _validation_options: dict | None = None
    association_validator = AssociationValidator()

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        data: dict[str, Any] | None = None,
        ui_json: dict[str, Any] | None = None,
        validate: bool = True,
        validations: dict | None = None,
        validation_options: dict | None = None,
        promotion: bool = True,
    ):
        self._geoh5 = None
        self.validation_options = validation_options
        self.validate = validate
        self.validations = validations
        self.promotion = promotion
        self.ui_json = ui_json
        self.data = data

    @property
    def data(self) -> dict[str, Any] | None:
        """
        Dictionary representing the input data for the ui.json file.
        """
        if self._data is None and self.ui_json is not None:
            original = self.validation_options.get("update_enabled", True)
            self.validation_options["update_enabled"] = False
            self.data = flatten(self.ui_json)
            self.validation_options["update_enabled"] = original

        return self._data

    @data.setter
    def data(self, value: dict[str, Any] | None):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("Input 'data' must be of type dict or None.")

            if self._ui_json is None:
                raise AttributeError("'ui_json' must be set before setting data.")

            if len(value) != len(self._ui_json):
                warnings.warn(
                    "The number of input values for 'data' differs from "
                    "the number of parameters in 'ui_json'."
                )

            if self._geoh5 is None and "geoh5" in value:
                self.geoh5 = value["geoh5"]

            with fetch_active_workspace(self._geoh5):
                if self.promotion:
                    value = self.promote(value)

                if self.validators is not None and self.validate:
                    self.validators.validate_data(value)

            self.update_ui_values(value)

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

    @property
    def path(self) -> str | None:
        """
        Directory for the input/output ui.json file.
        """
        if getattr(self, "_path", None) is None and self.geoh5 is not None:
            self.path = str(Path(self.geoh5.h5file).parent)

        return self._path

    @path.setter
    def path(self, path: str):
        dir_path = Path(path).resolve(strict=True)
        if not dir_path.is_dir():
            raise ValueError(f"The specified path is not a directory: {path}")

        self._path = str(dir_path)

    @property
    def path_name(self) -> str | None:
        if self.path is not None and self.name is not None:
            return str(Path(self.path) / self.name)

        return None

    @staticmethod
    def read_ui_json(json_file: str | Path, **kwargs):
        """
        Read and create an InputFile from ui.json
        """

        json_file_path = Path(json_file).resolve()
        if "".join(json_file_path.suffixes[-2:]) != ".ui.json":
            raise ValueError("Input file should have the extension .ui.json")

        input_file = InputFile(**kwargs)
        input_file.path = str(json_file_path.parent)
        input_file.name = json_file_path.name

        with open(json_file, encoding="utf-8") as file:
            input_file.ui_json = json.load(file)

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
            infered_validations = InputValidation.infer_validations(self._ui_json)

            if self.validations is None:
                self.validations = {}

            for key, validations in infered_validations.items():
                if key in self.validations:
                    validations = {**validations, **self.validations[key]}
                self.validations[key] = validations

        else:
            self._ui_json = None
            self._validations = None

        self._validators = None

    @classmethod
    def ui_validation(cls, ui_json: dict[str, Any]):
        """Validation of the ui_json forms"""
        cls._ui_validators(ui_json)

    def update_ui_values(self, data: dict):
        """
        Update the ui.json values and enabled status from input data.

        :param data: Key and value pairs expected by the ui_json.

        :raises AttributeError: If attempting to set None value to non-optional parameter.
        """
        if self.ui_json is None:
            raise AttributeError("InputFile requires a 'ui_json' to be defined.")

        for key, value in data.items():
            if key in self.ui_json and isinstance(self.ui_json[key], dict):
                enabled = self.ui_json[key].get("enabled", None)
                if enabled is not None:
                    if self.validation_options.get("update_enabled", True):
                        enabled = False if value is None else True

                    set_enabled(self.ui_json, key, enabled, validate=self.validate)

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
    def validate(self):
        """Option to run validations."""
        return self._validate

    @validate.setter
    def validate(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Input value for `validate` should be True or False.")

        self._validate = value

    @property
    def validation_options(self):
        """
        Pass validation options to the validators.

        The following options are supported:

        - update_enabled: bool
            If True, the enabled status of the ui_json will be updated based on the
            value provided. Default is True.
        - ignore_list: tuple
            List of keys to ignore when validating the ui_json. Default is empty tuple.

        """
        if self._validation_options is None:
            self._validation_options = {
                "update_enabled": True,
                "ignore_list": (),
            }

        return self._validation_options

    @validation_options.setter
    def validation_options(self, value: dict):
        if not isinstance(value, (dict, type(None))):
            raise ValueError("Input value for `validation_options` should be a dict.")

        if value is not None:
            for key in value:
                if key not in self.validation_options:
                    raise KeyError(
                        f"Input key '{key}' not supported. "
                        f"Supported keys: {list(self.validation_options.keys())}"
                    )

        self._validation_options = value

    @property
    def validations(self) -> dict | None:
        """Dictionary of validations for the ui_json."""
        if self._validations is None:
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
    def geoh5(self):
        """Geoh5 workspace."""
        if self._geoh5 is None and self.data is not None:
            self._geoh5 = self.data["geoh5"]
        return self._geoh5

    @geoh5.setter
    def geoh5(self, geoh5: Workspace | None):
        if geoh5 is None:
            return

        if self._geoh5 is not None:
            raise UserWarning(
                "Attribute 'geoh5' already set. "
                "Consider creating a new InputFile from arguments."
            )
        if not isinstance(geoh5, Workspace):
            raise ValueError(
                "Input 'geoh5' must be a valid :obj:`geoh5py.workspace.Workspace`."
            )
        self._geoh5 = geoh5
        if self.validators is not None:
            self.validators.geoh5 = self.geoh5

    def write_ui_json(
        self,
        name: str | None = None,
        path: str | Path | None = None,
    ):
        """
        Writes a formatted ui.json file from InputFile data

        :param name: Name of the file
        :param path: Directory to write the ui.json to.
        """

        if name is not None:
            self.name = name

        if path is not None:
            self.path = str(path)

        if self.path_name is None:
            raise AttributeError(
                "The input file requires 'path' and 'name' to be set before writing out."
            )

        if self.ui_json is None:
            raise AttributeError(
                "The input file requires 'ui_json' and 'data' to be set before writing out."
            )

        if self.data is not None:
            self.update_ui_values(self.data)

        with open(self.path_name, "w", encoding="utf-8") as file:
            json.dump(self.stringify(self.demote(self.ui_json)), file, indent=4)

        return self.path_name

    def set_data_value(self, key: str, value):
        """
        Set the data and json form values from a dictionary.

        :param key: Parameter name to update.
        :param value: Value to update with.
        """
        assert self.data is not None
        if self.validate and self.validations is not None and key in self.validations:
            if "association" in self.validations[key]:
                validations = deepcopy(self.validations[key])
                parent = self.data[self.validations[key]["association"]]
                if isinstance(parent, UUID):
                    parent = self.geoh5.get_entity(parent)[0]
                validations["association"] = parent
            else:
                validations = self.validations[key]

            validations = {k: v for k, v in validations.items() if k != "one_of"}
            self.validators.validate(key, value, validations)

        self.data[key] = value

        if key == "geoh5":
            self.geoh5 = value

        self.update_ui_values({key: value})

    @staticmethod
    def stringify(var: dict[str, Any]) -> dict[str, Any]:
        """
        Convert inf, none, and list types to strings within a dictionary

        :param var: Dictionary containing ui.json keys, values, fields

        :return: Dictionary with inf and none types converted to string
            representations in json format.
        """
        var = stringify(var)

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

            mappers = [str2none, str2inf, str2uuid, path2workspace]
            ui_json[key] = dict_mapper(value, mappers)

        return ui_json

    @classmethod
    def demote(cls, var: dict[str, Any]) -> dict[str, Any]:
        """
        Converts promoted parameter values to their string representations.

        Other parameters are left unchanged.
        """
        mappers = [entity2uuid, as_str_if_uuid, workspace2path, container_group2name]
        demoted: dict[str, Any] = {}
        for key, value in var.items():
            if isinstance(value, dict):
                demoted[key] = cls.demote(value)

            elif isinstance(value, (list, tuple)):
                demoted[key] = [dict_mapper(val, mappers) for val in value]
            else:
                demoted[key] = dict_mapper(value, mappers)

        return demoted

    def promote(self, var: dict[str, Any]) -> dict[str, Any]:
        """Convert uuids to entities from the workspace."""
        if self._geoh5 is None:
            return var

        for key, value in var.items():
            if isinstance(value, dict):
                var[key] = self.promote(value)
            else:
                if isinstance(value, list):
                    var[key] = [self._uid_promotion(key, val) for val in value]
                else:
                    var[key] = self._uid_promotion(key, value)

        return var

    def _uid_promotion(self, key, value):
        """
        Check if the value needs to be promoted.
        """
        if isinstance(value, UUID) and self._geoh5 is not None:
            if self.validate:
                self.association_validator(key, value, self._geoh5)
            value = uuid2entity(value, self._geoh5)

        return value

    @property
    def workspace(self) -> Workspace | None:
        """Return the workspace associated with the input file."""

        warnings.warn(
            "The 'workspace' property is deprecated. Use 'geoh5' instead.",
            DeprecationWarning,
        )

        return self._geoh5

    @workspace.setter
    def workspace(self, value):
        warnings.warn(
            "The 'workspace' property is deprecated. Use 'geoh5' instead.",
            DeprecationWarning,
        )
        self.geoh5 = value
