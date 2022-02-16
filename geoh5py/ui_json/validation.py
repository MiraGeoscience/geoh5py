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

from copy import deepcopy
from typing import Any, cast
from uuid import UUID

from geoh5py.groups import PropertyGroup
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import RequiredValidationError
from geoh5py.shared.validators import BaseValidator, TypeValidator
from geoh5py.workspace import Workspace


class InputValidation:
    """
    Validations on dictionary of parameters.

    Attributes
    ----------
    validations : Validations dictionary with matching set of input parameter keys.
    workspace (optional) : Workspace instance needed to validate uuid types.
    ignore_requirements (optional): Omit raising error on 'required' validator.

    Methods
    -------
    validate_data(data)
        Validates data of params and contents/type/shape/keys/reqs of values.
    """

    def __init__(
        self,
        validators: dict[str, BaseValidator] = None,
        validations: dict[str, Any] | None = None,
        ignore_list: tuple = (),
        ignore_requirements: bool = False,
        workspace: Workspace = None,
        ui_json: dict[str, Any] = None,
    ):
        self.validations: dict[str, Any] | None = validations
        self.ui_json: dict[str, Any] = ui_json
        self.validators: dict[str, BaseValidator] = validators
        self.workspace: Workspace | None = workspace
        self.ignore_list: tuple = ignore_list
        self.ignore_requirements: bool = ignore_requirements

    @property
    def ui_json(self):
        return self._ui_json

    @ui_json.setter
    def ui_json(self, val):
        if val is None:
            self._inferred_validations = {}
        else:
            self._inferred_validations = self.infer_validations(val)
        self._ui_json = val

    @property
    def validations(self):
        return dict(self._inferred_validations, **self._validations)

    @validations.setter
    def validations(self, val: dict[str, Any]):
        self._validations = val

    @property
    def validators(self):
        return self._validators

    @validators.setter
    def validators(self, val):
        v = {} if val is None else val

        if self.validations is not None:
            required_validators = InputValidation._required_validators(self.validations)
            v = dict(required_validators, **v)

        self._validators = v

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, value: Workspace | None):
        if value is not None:
            TypeValidator.validate("workspace", value, Workspace)
        self._workspace = value

    @staticmethod
    def _unique_validators(validations: dict[str, Any]) -> list[str]:
        """Return names of validators required by a validations dictionary."""
        return list({key for item in validations.values() for key in item})

    @staticmethod
    def _required_validators(validations: dict[str, Any]) -> dict[str, BaseValidator]:
        unique_validators = InputValidation._unique_validators(validations)
        all_validators = {k.validator_type: k for k in BaseValidator.__subclasses__()}
        val = {}
        for k in unique_validators:
            if k not in all_validators:
                raise ValueError(f"No validator implemented for argument '{k}'.")
            else:
                val[k] = all_validators[k]()
        return val

    @staticmethod
    def infer_validations(ui_json: dict[str, Any]):
        validations = {}
        for key, item in ui_json.items():
            if not isinstance(item, dict):
                continue

            if "isValue" in item:
                validations[key] = {
                    "types": [UUID, int, float, Entity, type(None)],
                    "association": item["parent"],
                }
            elif "choiceList" in item:
                validations[key] = {"types": [str], "values": item["choiceList"]}
            elif "fileType" in item:
                validations[key] = {
                    "types": [str],
                }
            elif "meshType" in item:
                validations[key] = {
                    "types": [UUID, Entity, type(None)],
                    "association": "geoh5",
                }
            elif "parent" in item:
                validations[key] = {
                    "types": [UUID, Entity, type(None)],
                    "association": item["parent"],
                }
                if "dataGroupType" in item:
                    validations[key]["property_group_type"] = item["dataGroupType"]
                    validations[key]["types"] = [UUID, PropertyGroup, type(None)]
            elif "value" in item:
                if item["value"] is None:
                    check_type = str
                else:
                    check_type = cast(Any, type(item["value"]))
                validations[key] = {
                    "types": [check_type],
                }

        return validations

    def validate(self, name: str, value: Any, validations: dict[str, Any] = None):
        """
        Run validations on a given key and value.

        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param validations: [Optional] Validations provided on runtime
        """
        if validations is None:
            if name not in self.validations:
                raise KeyError(f"{name} is missing from the known validations.")

            validations = self.validations[name]

        for val, args in validations.items():

            if (
                val == "required" and self.ignore_requirements
            ) or name in self.ignore_list:
                continue

            self.validators[val](name, value, args)

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Calls validate method on individual key/value pairs in input.

        :param data: Input data with known validations.
        """
        for name, validations in self.validations.items():
            if name not in data.keys():
                if "required" in validations and not self.ignore_requirements:
                    raise RequiredValidationError(name)

                continue

            if "association" in validations and validations["association"] in data:
                temp_validate = deepcopy(validations)
                temp_validate["association"] = data[validations["association"]]
                self.validate(name, data[name], temp_validate)
            else:
                self.validate(name, data[name])

    def __call__(self, data, *args):
        if isinstance(data, dict):
            self.validate_data(data)
        elif isinstance(data, str) and args is not None:
            self.validate(data, args)
        else:
            raise ValueError(
                "InputValidators can only be called with dictionary of data or "
                "(key, value) pair."
            )
