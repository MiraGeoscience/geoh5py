#  Copyright (c) 2021 Mira Geoscience Ltd.
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

from typing import Any

from geoh5py.shared.exceptions import RequiredValidationError
from geoh5py.shared.validators import (
    BaseValidator,
    PropertyGroupValidator,
    RequiredValidator,
    ShapeValidator,
    TypeValidator,
    UUIDValidator,
    ValueValidator,
)
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
        validations: dict[str, Any],
        workspace: Workspace = None,
        ignore_requirements: bool = False,
    ):
        self._validators: dict[str, BaseValidator] = {}
        self.workspace: Workspace | None = workspace
        self.validations = validations
        self.ignore_requirements: bool = ignore_requirements

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, value: Workspace | None):
        if value is not None:
            TypeValidator.validate("workspace", value, Workspace)
        self._workspace = value

    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):
        if isinstance(val, dict):
            validator_list = list({key for item in val.values() for key in item})
            for key in validator_list:
                if key == "property_group":
                    self._validators[key] = PropertyGroupValidator()
                elif key == "required":
                    self._validators[key] = RequiredValidator()
                elif key == "shape":
                    self._validators[key] = ShapeValidator()
                elif key == "types":
                    self._validators[key] = TypeValidator()
                elif key == "uuid":
                    self._validators[key] = UUIDValidator()
                elif key == "values":
                    self._validators[key] = ValueValidator()
                else:
                    raise ValueError(f"No validator implemented for argument '{key}'.")
        elif val is None:
            self._validators = None
        else:
            raise ValueError(
                "Input 'validations' must be of of type 'dict' or None. "
                f"Argument of type {type(val)} provided"
            )

        self._validations = val

    def validate(self, name: str, value: Any):
        """
        Run validations on a given key and value.

        :param name: Parameter identifier.
        :param value: Input parameter value.
        """
        if name not in self.validations:
            raise KeyError(f"{name} is missing from the known validations.")

        for val, args in self.validations[name].items():

            if val == "required" and self.ignore_requirements:
                continue

            self._validators[val](name, value, args)

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Calls validate method on individual key/value pairs in input.

        :param data: Input data with known validations.
        """
        for key, validations in self.validations.items():
            if key not in data.keys():
                if "required" in validations and not self.ignore_requirements:
                    raise RequiredValidationError(key)

                continue

            self.validate(key, data[key])

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
