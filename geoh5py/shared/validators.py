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

# pylint: disable=too-few-public-methods

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from geoh5py import TYPE_UID_TO_CLASS, Workspace
from geoh5py.groups import Group, PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    AtLeastOneValidationError,
    OptionalValidationError,
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
    iterable,
)


def to_path(value: list[str]) -> list[Path]:
    """Promote path strings to patlib.Path objects."""
    out = []
    for path in value:
        if isinstance(path, str):
            out.append(Path(path))
        else:
            out.append(path)
    return out


def to_list(value: Any) -> list[Any]:
    """Promote single values to list."""
    if isinstance(value, str) and ";" in value:
        value = value.split(";")
    if not isinstance(value, list):
        value = [value]
    return value


def to_uuid(values):
    """Promote strings to uuid and pass anything else."""
    out = []
    for val in values:
        if isinstance(val, str):
            val = UUID(val)
        out.append(val)
    return out


def class_or_raise(value: UUID) -> type[ObjectBase] | type[Group]:
    """Promote uid to class, raise if uid is not a geoh5py type uid."""
    if value not in TYPE_UID_TO_CLASS:
        raise ValueError(
            f"Provided type_uid string {value!s} is not a recognized "
            f"geoh5py object or group type uid."
        )
    return TYPE_UID_TO_CLASS[value]


def to_class(
    values: list[UUID | type[ObjectBase] | type[Group]],
) -> list[type[ObjectBase] | type[Group]]:
    """
    Promote uid to class.

    Passes existing classes and raises if uid is not a geoh5py type uid.
    """
    out = []
    for val in values:
        if isinstance(val, UUID):
            out.append(class_or_raise(val))
        elif issubclass(val, (ObjectBase, Group)):
            out.append(val)
    return out


def empty_string_to_uid(value):
    """Promote empty string to uid, and pass all other values."""
    if value == "":
        return UUID("00000000-0000-0000-0000-000000000000")
    return value


class BaseValidator(ABC):
    """Concrete base class for validators."""

    validator_type: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __call__(self, *args):
        self.validate(*args)

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


class OptionalValidator(BaseValidator):
    """Validate that forms contain optional parameter if None value is given."""

    validator_type = "optional"

    @classmethod
    def validate(
        cls,
        name: str,
        value: Any | None,
        valid: bool,
    ) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: True if optional keyword in form for parameter.
        """
        if value is None and not valid:
            raise OptionalValidationError(name, value, valid)


class AssociationValidator(BaseValidator):
    """Validate the association between data and parent object."""

    validator_type = "association"

    @classmethod
    def validate(
        cls,
        name: str,
        value: Entity | PropertyGroup | UUID | None,
        valid: Entity | Workspace,
    ) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        if valid is None:
            return

        if isinstance(valid, list):
            warnings.warn(
                "Data associated with multiSelect dependent is not supported. Validation ignored."
            )
            return

        if not isinstance(valid, (Entity, Workspace)):
            raise ValueError(
                "'AssociationValidator.validate' requires a 'valid'"
                " input of type 'Entity', 'Workspace' or None. "
                f"Provided '{valid}' of type {type(valid)} for parameter '{name}'"
            )

        if isinstance(value, UUID):
            uid = value
        elif isinstance(value, (Entity, PropertyGroup)):
            uid = value.uid
        else:
            return

        if isinstance(valid, Workspace):
            # TODO add a generic method to workspace to get all uuid
            children = valid.get_entity(uid)
            if None in children:
                children = valid.fetch_children(valid.root, recursively=True)

        elif isinstance(valid, Entity):
            children = valid.workspace.fetch_children(valid, recursively=True)

        if uid not in [getattr(child, "uid", None) for child in children]:
            raise AssociationValidationError(name, value, valid)


class PropertyGroupValidator(BaseValidator):
    """Validate property_group from parent entity."""

    validator_type = "property_group_type"

    @classmethod
    def validate(cls, name: str, value: PropertyGroup, valid: str | list[str]) -> None:
        if isinstance(valid, str):
            valid = [valid]

        if (value is not None) and (value.property_group_type not in valid):
            raise PropertyGroupValidationError(name, value, valid)


class AtLeastOneValidator(BaseValidator):
    validator_type = "one_of"

    @classmethod
    def validate(cls, name, value, valid):
        if not any(v for v in value.values()):
            raise AtLeastOneValidationError(name, value)


class RequiredValidator(BaseValidator):
    """
    Validate that required keys are present in parameter.
    """

    validator_type = "required"

    @classmethod
    def validate(cls, name: str, value: Any, valid: bool) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Assert to be required
        """
        if value is None and valid:
            raise RequiredValidationError(name)


class ShapeValidator(BaseValidator):
    """Validate the shape of provided value."""

    validator_type = "shape"

    @classmethod
    def validate(cls, name: str, value: Any, valid: tuple[int, ...]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        if value is None:
            return

        if isinstance(value, np.ndarray):
            pshape = value.shape
        elif isinstance(value, list):
            pshape = (len(value),)
        else:
            pshape = (1,)

        if pshape != valid:
            raise ShapeValidationError(name, pshape, valid)


class TypeValidator(BaseValidator):
    """
    Validate the value type from a list of valid types.
    """

    validator_type = "types"

    @classmethod
    def validate(cls, name: str, value: Any, valid: type | list[type]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted value types
        """
        if isinstance(valid, type):
            valid = [valid]

        if not isinstance(valid, list):
            raise TypeError("Input `valid` options must be a type or list of types.")

        if not iterable(value) or (isinstance(value, list) and list in tuple(valid)):
            value = (value,)

        for val in value:
            if not isinstance(val, tuple(valid)):
                valid_names = [t.__name__ for t in valid if hasattr(t, "__name__")]
                type_name = type(val).__name__

                raise TypeValidationError(name, type_name, valid_names)


class UUIDValidator(BaseValidator):
    """Validate a uuui.UUID value or uuid string."""

    validator_type = "uuid"

    @classmethod
    def validate(cls, name: str, value: Any, valid: None = None) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter uuid.
        :param valid: [Optional] Validate uuid from parental entity or known uuids
        """

        if isinstance(value, str):
            try:
                value = UUID(value)
            except ValueError as exception:
                raise UUIDValidationError(name, str(value)) from exception


class ValueValidator(BaseValidator):
    """
    Validator that ensures that values are valid entries.
    """

    validator_type = "values"

    @classmethod
    def validate(cls, name: str, value: Any, valid: list[float | str]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted values
        """
        if value is None:
            return

        if not isinstance(value, (list, tuple)):
            value = [value]

        for val in value:
            if val is not None and val not in valid:
                raise ValueValidationError(name, val, valid)
