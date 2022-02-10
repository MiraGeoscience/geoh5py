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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geoh5py.shared import Entity
    from geoh5py.groups import PropertyGroup
    from uuid import UUID

from abc import ABC, abstractmethod

from geoh5py.shared.utils import iterable_message
from geoh5py.workspace import Workspace

class BaseValidationError(ABC):
    """Base class for custom exceptions."""

    @staticmethod
    @abstractmethod
    def message(name):
        """Builds custom error message."""
        raise NotImplementedError(
            "The 'error_message' method must be implemented by the sub-class. "
            f"Must contain a 'name' {name} argument."
        )


class AssociationValidationError(BaseValidationError, Exception):
    """Error on association between child and parent entity validation."""

    def __init__(self, name: str, value: Entity, parent: Entity | Workspace):
        super().__init__(AssociationValidationError.message(name, value, parent))

    @staticmethod
    def message(name, value, parent):
        return f"Property '{name}' of type '{value}' must be a child entity of parent {parent}"


class JSONParameterValidationError(BaseValidationError, Exception):
    """Error on uuid validation."""

    def __init__(self, name: str, err: str):
        super().__init__(JSONParameterValidationError.message(name, err))

    @staticmethod
    def message(name, err):
        return f"Malformed ui.json dictionary for parameter '{name}'. {err}"


class PropertyGroupValidationError(BaseValidationError, Exception):
    """Error on property group validation."""

    def __init__(self, name: str, value: PropertyGroup, valid: str):
        super().__init__(PropertyGroupValidationError.message(name, value, valid))

    @staticmethod
    def message(name, value, valid):
        return (
            f"Property group for '{name}' must be of type '{valid}'. "
            f"Provided '{value.name}' of type '{value.property_group_type}'"
        )


class RequiredValidationError(BaseValidationError, Exception):
    def __init__(self, name: str):
        super().__init__(RequiredValidationError.message(name))

    @staticmethod
    def message(name):
        return f"Missing required parameter: '{name}'."


class ShapeValidationError(BaseValidationError, Exception):
    """Error on shape validation."""

    def __init__(self, name: str, value: tuple(int), valid: tuple(int)):
        super().__init__(ShapeValidationError.message(name, value, valid))

    @staticmethod
    def message(name, value, valid):
        return (
            f"Parameter '{name}' with shape {value} was provided. "
            f"Expected {valid}."
        )


class TypeValidationError(BaseValidationError, Exception):
    """Error on type validation."""

    def __init__(self, name: str, value: type, valid: type):

        super().__init__(TypeValidationError.message(name, value, valid))

    @staticmethod
    def message(name, value, valid):
        return (
            f"Type '{value}' provided for '{name}' is invalid. "
            + iterable_message(valid)
        )


class UUIDValidationError(BaseValidationError, Exception):
    """Error on uuid validation."""

    def __init__(self, name: str, value: str | UUID, valid: Entity | Workspace):
        super().__init__(UUIDValidationError.message(name, value, valid))

    @staticmethod
    def message(name, value, valid):
        valid_name = valid.h5file if isinstance(valid, Workspace) else valid.name
        return (
            f"UUID '{value}' provided for '{name}' is invalid. "
            f"Not in the list of children of {type(valid).__name__}: {valid_name} "
        )


class UUIDStringValidationError(BaseValidationError, Exception):
    """Error on uuid string validation."""

    def __init__(self, name: str, value: str):
        super().__init__(UUIDStringValidationError.message(name, value))

    @staticmethod
    def message(name, value):
        return(
            f"Parameter '{name}' with value '{value}' is not a valid uuid string."
        )


class ValueValidationError(BaseValidationError, Exception):
    """Error on value validation."""

    def __init__(self, name: str, value: Any, valid: list[Any]):

        super().__init__(ValueValidationError.message(name, value, valid))

    @staticmethod
    def message(name, value, valid):
        return (
            f"Value '{value}' provided for '{name}' is invalid."
            + iterable_message(valid)
        )
