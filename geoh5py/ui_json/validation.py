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
from typing import Any

from geoh5py.shared.exceptions import RequiredValidationError
from geoh5py.shared.validators import (
    AssociationValidator,
    BaseValidator,
    PropertyGroupValidator,
    RequiredValidator,
    ShapeValidator,
    TypeValidator,
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
        validators: dict[str, BaseValidator] = None,
        validations: dict[str, Any] | None = None,
        ignore_list: tuple = (),
        ignore_requirements: bool = False,
        workspace: Workspace = None,
    ):
        self.validators: dict[str, BaseValidator] = validators
        self.workspace: Workspace | None = workspace
        self.validations: dict[str, Any] | None = validations
        self.ignore_list: tuple = ignore_list
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
    def validators(self):
        return self._validators

    @validators.setter
    def validators(self, val):
        if val is None:
            self._validators = {}
        else:
            self._validators = val


    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):

        if isinstance(val, dict):
            required_validators = list({key for item in val.values() for key in item})
            all_validators = {k.validator_type: k for k in BaseValidator.__subclasses__()}
            all_validators.update(self.validators)
            for key in required_validators:
                try:
                    self.validators[key] = all_validators[key]()
                except IndexError:
                    raise ValueError(f"No validator implemented for argument '{key}'.")

        elif val is None:
            self.validators = None
        else:
            raise ValueError(
                "Input 'validations' must be of of type 'dict' or None. "
                f"Argument of type {type(val)} provided"
            )

        self._validations = val

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
