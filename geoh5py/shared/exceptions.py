#  Copyright (c) 2023 Mira Geoscience Ltd.
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from uuid import UUID

from geoh5py.shared.utils import iterable_message

if TYPE_CHECKING:
    from geoh5py import Workspace
    from geoh5py.groups import PropertyGroup
    from geoh5py.shared import Entity


class Geoh5FileClosedError(ABC, Exception):
    """Error for closed geoh5 file."""


class BaseValidationError(ABC, Exception):
    """Base class for custom exceptions."""

    @staticmethod
    @abstractmethod
    def message(name, value, validation):
        """Builds custom error message."""
        raise NotImplementedError()


class OptionalValidationError(BaseValidationError):
    """Error if None value provided to non-optional parameter."""

    def __init__(
        self,
        name: str,
        value: Any | None,
        validation: bool,
    ):
        super().__init__(OptionalValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return f"Cannot set a None value to non-optional parameter: {name}."


class AssociationValidationError(BaseValidationError):
    """Error on association between child and parent entity validation."""

    def __init__(
        self,
        name: str,
        value: Entity | PropertyGroup | UUID,
        validation: Entity | Workspace,
    ):
        super().__init__(AssociationValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return (
            f"Property '{name}' with value: '{value}' must be "
            f"a child entity of parent {validation}"
        )


class PropertyGroupValidationError(BaseValidationError):
    """Error on property group validation."""

    def __init__(self, name: str, value: PropertyGroup, validation: str):
        super().__init__(PropertyGroupValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return (
            f"Property group for '{name}' must be of type '{validation}'. "
            f"Provided '{value.name}' of type '{value.property_group_type}'"
        )


class AtLeastOneValidationError(BaseValidationError):
    def __init__(self, name: str, value: list[str]):
        super().__init__(AtLeastOneValidationError.message(name, value))

    @staticmethod
    def message(name, value, validation=None):
        opts = "'" + "', '".join(str(k) for k in value) + "'"
        return f"Must provide at least one {name}.  Options are: {opts}"


class RequiredValidationError(BaseValidationError):
    def __init__(self, name: str):
        super().__init__(RequiredValidationError.message(name))

    @staticmethod
    def message(name, value=None, validation=None):
        return f"Missing required parameter: '{name}'."


class ShapeValidationError(BaseValidationError):
    """Error on shape validation."""

    def __init__(self, name: str, value: tuple[int], validation: tuple[int] | str):
        super().__init__(ShapeValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return (
            f"Parameter '{name}' with shape {value} was provided. "
            f"Expected {validation}."
        )


class TypeValidationError(BaseValidationError):
    """Error on type validation."""

    def __init__(self, name: str, value: str, validation: str | list[str]):
        super().__init__(TypeValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return f"Type '{value}' provided for '{name}' is invalid. " + iterable_message(
            validation
        )


class UUIDValidationError(BaseValidationError):
    """Error on uuid string validation."""

    def __init__(self, name: str, value: str):
        super().__init__(UUIDValidationError.message(name, value))

    @staticmethod
    def message(name, value, validation=None):
        return f"Parameter '{name}' with value '{value}' is not a valid uuid string."


class ValueValidationError(BaseValidationError):
    """Error on value validation."""

    def __init__(self, name: str, value: Any, validation: list[Any]):
        super().__init__(ValueValidationError.message(name, value, validation))

    @staticmethod
    def message(name, value, validation):
        return f"Value '{value}' provided for '{name}' is invalid." + iterable_message(
            validation
        )


class JSONParameterValidationError(Exception):
    """Error on uuid validation."""

    def __init__(self, name: str, err: str):
        super().__init__(JSONParameterValidationError.message(name, err))

    @staticmethod
    def message(name, err):
        return f"Malformed ui.json dictionary for parameter '{name}'. {err}"
