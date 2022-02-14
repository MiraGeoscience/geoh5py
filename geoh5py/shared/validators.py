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

from geoh5py.groups import PropertyGroup
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDStringValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.shared.utils import iterable
from geoh5py.workspace import Workspace


class BaseValidator(ABC):
    """
    Base class for validators
    """

    def __init__(self, **kwargs):

        self._validation_type = ""

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    @abstractmethod
    def validate(cls, name: str, value: Any, valid: Any):
        """
        Custom validation function.
        """
        raise NotImplementedError(
            "The 'validate' method must be implemented by the sub-class. "
            f"Must contain a 'name' {name}, 'value' {value} and 'valid' {valid} argument."
        )

    @property
    @abstractmethod
    def validation_type(self):
        """Validation type identifier."""

    def __call__(self, *args):
        if hasattr(self, "validate"):
            self.validate(*args)


class AssociationValidator(BaseValidator):
    """Validate the shape of provided value."""

    @classmethod
    def validate(cls, name: str, value: Entity, valid: Entity | Workspace) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        if not isinstance(value, Entity):
            return

        if isinstance(valid, Workspace):
            children = valid.fetch_children(valid.root, recursively=True)
        else:
            children = valid.workspace.fetch_children(valid, recursively=True)

        if value.uid not in [child.uid for child in children]:
            raise AssociationValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "association"


class PropertyGroupValidator(BaseValidator):
    """Validate property_group from parent entity."""

    @classmethod
    def validate(cls, name: str, value: PropertyGroup, valid: str) -> None:
        if value.property_group_type != valid:
            raise PropertyGroupValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "property_group"


class RequiredValidator(BaseValidator):
    """
    Validate that required keys are present in parameter.
    """

    @classmethod
    def validate(cls, name: str, value: Any, valid: bool) -> None:
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


class ShapeValidator(BaseValidator):
    """Validate the shape of provided value."""

    @classmethod
    def validate(cls, name: str, value: Any, valid: tuple[int]) -> None:
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


class TypeValidator(BaseValidator):
    """
    Validate the value type from a list of valid types.
    """

    @classmethod
    def validate(cls, name: str, value: Any, valid: list[type] | type) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted value types
        """
        if isinstance(valid, type):
            valid = [valid]

        if not iterable(value) or list in valid:
            value = (value,)
        for val in value:
            if not isinstance(val, tuple(valid)):
                valid_names = [t.__name__ for t in valid if hasattr(t, "__name__")]
                type_name = type(val).__name__

                raise TypeValidationError(name, type_name, valid_names)

    @property
    def validation_type(self):
        return "types"


class UUIDValidator(BaseValidator):
    """Validate a uuui.UUID value or uuid string."""

    @classmethod
    def validate(
        cls, name: str, value: UUID | str, valid: None | Entity | Workspace = None
    ) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter uuid.
        :param valid: [Optional] Validate uuid from parental entity or known uuids
        """
        if not isinstance(value, UUID):
            try:
                value = UUID(value)
            except ValueError as exception:
                raise UUIDStringValidationError(name, value) from exception

        if valid is not None:
            if isinstance(valid, Workspace) and valid.root is not None:
                children_list = valid.fetch_children(valid.root, recursively=True)
            elif isinstance(valid, Entity):
                children_list = valid.workspace.fetch_children(valid, recursively=True)
            else:
                raise ValueError(
                    "Type of input `valid` parameter must be one of Entity or Workspace"
                )

            if value not in [child.uid for child in children_list]:
                raise UUIDValidationError(name, value, valid)

    @property
    def validation_type(self):
        return "uuid"


class ValueValidator(BaseValidator):
    """
    Validator that ensures that values are valid entries.
    """

    @classmethod
    def validate(cls, name: str, value: Any, valid: list[float | str]) -> None:
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
