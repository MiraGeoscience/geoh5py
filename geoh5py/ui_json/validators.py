#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

import numpy as np

from geoh5py.shared.utils import iterable
from geoh5py.ui_json.exceptions import (
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDStringValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.workspace import Workspace


class BaseValidator(ABC):
    """
    Base class for validators
    """

    def __init__(self, workspace: Workspace = None, **kwargs):

        self.workspace = workspace
        self._validation_type = ""

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def workspace(self):
        """Target Workspace"""
        return self._workspace

    @workspace.setter
    def workspace(self, value: Workspace | None):
        if not isinstance(value, Workspace) and value is not None:
            raise TypeError(f"Input workspace must be of type {Workspace} or None")
        self._workspace = value

    @abstractmethod
    def validate(self, name: str, value: Any, valid: Any):
        """
        Custom validation function.
        """
        ...

    @property
    @abstractmethod
    def validation_type(self):
        ...

    def __call__(self, *args):
        if hasattr(self, "validate"):
            self.validate(*args)


class RequiredValidator(BaseValidator):
    """
    Validate that required keys are present in parameter.
    """

    def validate(self, name: str, value: Any, valid: bool) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Assert to be required
        """
        if value is None and valid:
            raise RequiredValidationError(name)

    @property
    def validation_type(self):
        return "required"


class ValueValidator(BaseValidator):
    """
    Validator that ensures that values are valid entries.
    """

    def validate(self, name: str, value: Any, valid: list[float | str]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted values
        """
        if value not in valid:
            raise ValueValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "values"


class TypeValidator(BaseValidator):
    """
    Validate the value type from a list of valid types.
    """

    def validate(self, name: str, value: Any, valid: list[type]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted value types
        """
        if not iterable(value):
            value = (value,)
        for val in value:
            if type(val) not in valid:
                valid_names = [t.__name__ for t in valid if hasattr(t, "__name__")]
                type_name = type(val).__name__

                raise TypeValidationError(name, type_name, valid_names)

    @property
    def validation_type(self):
        return "types"


class ShapeValidator(BaseValidator):
    """Validate the shape of provided value."""

    def validate(self, name: str, value: Any, valid: list[tuple[int]]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        pshape = np.array(value).shape
        if pshape != valid:
            raise ShapeValidationError(name, pshape, valid)

    @property
    def validation_type(self):
        return "shape"


class UUIDValidator(BaseValidator):
    """Validate the value for uuui.UUID compliance."""

    def validate(self, name: str, value: Any, valid: list) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: (optional) List of accepted uuid
        """
        if not isinstance(value, UUID):
            try:
                value = UUID(value)
            except UUIDStringValidationError as exception:
                raise UUIDStringValidationError(name, value) from exception

        if any(valid) and value not in valid:
            raise UUIDValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "shape"


class PropertyGroupValidator(BaseValidator):
    """Validate property_group from parent entity."""

    def validate(self, name: str, value: UUID, valid: UUID | None) -> None:
        if valid is not None:
            valid_entity = self.workspace.get_entity(valid)[0]
            if value not in [pg.uid for pg in valid_entity.property_groups]:
                raise PropertyGroupValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "property_group"


class InputValidators:
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
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):
        if isinstance(val, dict):
            validator_list = list({key for item in val.values() for key in item})
            for key in validator_list:
                if key == "property_group":
                    self._validators[key] = PropertyGroupValidator(self.workspace)
                elif key == "required":
                    self._validators[key] = RequiredValidator(self.workspace)
                elif key == "shape":
                    self._validators[key] = ShapeValidator(self.workspace)
                elif key == "types":
                    self._validators[key] = TypeValidator(self.workspace)
                elif key == "uuid":
                    self._validators[key] = UUIDValidator(self.workspace)
                elif key == "values":
                    self._validators[key] = ValueValidator(self.workspace)
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


class InputFreeformValidator(InputValidators):
    """
    Validations for Octree driver parameters.
    """

    def __init__(
        self,
        validations: dict[str, Any],
        workspace: Workspace = None,
        free_params_keys: list | tuple = (),
    ):
        super().__init__(validations, workspace=workspace)
        self._free_params_keys: list | tuple = free_params_keys

    def validate_data(self, data) -> None:
        free_params_dict: dict = {}
        for key, value in data.items():
            if " " in key:

                if "template" in key.lower():
                    continue

                for param in self.free_params_keys:
                    if param in key.lower():
                        group = key.lower().replace(param, "").lstrip()
                        if group not in list(free_params_dict.keys()):
                            free_params_dict[group] = {}

                        free_params_dict[group][param] = value
                        validator = self.validations[f"template_{param}"]

                        break

            elif key not in self.validations.keys():
                raise KeyError(f"{key} is not a valid parameter name.")
            else:
                validator = self.validations[key]

            for val, args in validator.items():
                self._validators[val](value, *args)

        # TODO This check should be handled by a group validator
        if any(free_params_dict):
            for key, group in free_params_dict.items():
                if not len(list(group.values())) == len(self.free_params_keys):
                    raise ValueError(
                        f"Freeformat parameter {key} must contain one of each: "
                        + f"{self.free_params_keys}"
                    )

    @property
    def free_params_keys(self) -> list | tuple:
        return self._free_params_keys
