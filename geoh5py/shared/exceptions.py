# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# pylint: disable=arguments-differ

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from uuid import UUID


if TYPE_CHECKING:
    from geoh5py import Workspace
    from geoh5py.groups import PropertyGroup
    from geoh5py.shared import Entity


class Geoh5FileClosedError(ABC, Exception):
    """Error for closed geoh5 file."""


class BaseValidationError(ABC, Exception):
    """
    Base class for custom exceptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self.message(*args, **kwargs))

    @classmethod
    @abstractmethod
    def message(cls, *args, **kwargs) -> str:
        """
        Builds custom error message.

        Each subclass of BaseValidationError should implement the message class method.
        """


class JSONParameterValidationError(BaseValidationError):
    """Error on uuid validation."""

    @classmethod
    def message(cls, name: str, err: str) -> str:
        return f"Malformed ui.json dictionary for parameter '{name}'. {err}"


class OptionalValidationError(BaseValidationError):
    """Error if None value provided to non-optional parameter."""

    @classmethod
    def message(cls, name: str, *_) -> str:
        return f"Cannot set a None value to non-optional parameter: {name}."


class AssociationValidationError(BaseValidationError):
    """Error on association between child and parent entity validation."""

    @classmethod
    def message(
        cls,
        name: str,
        value: Entity | PropertyGroup | UUID,
        validation: Entity | Workspace,
    ) -> str:
        return (
            f"Property '{name}' with value: '{value}' must be "
            f"a child entity of parent {validation}"
        )


class PropertyGroupValidationError(BaseValidationError):
    """Error on property group validation."""

    @classmethod
    def message(cls, name: str, value: PropertyGroup, validation: list[str]) -> str:
        return (
            f"Property group for '{name}' must be of type '{validation}'. "
            f"Provided '{value.name}' of type '{value.property_group_type}'"
        )


class AtLeastOneValidationError(BaseValidationError):
    """Error on at least one validation."""

    @classmethod
    def message(cls, name: str, value: list[str], *_) -> str:
        opts = "'" + "', '".join(str(k) for k in value) + "'"
        return f"Must provide at least one {name}.  Options are: {opts}"


class RequiredValidationError(BaseValidationError):
    """Error on required parameter validation."""

    @classmethod
    def message(cls, name: str, *_) -> str:
        return f"Missing required parameter: '{name}'."


class ShapeValidationError(BaseValidationError):
    """Error on shape validation."""

    @classmethod
    def message(
        cls, name: str, value: tuple[int, ...], validation: tuple[int, ...] | str
    ) -> str:
        return (
            f"Parameter '{name}' with shape {value} was provided. "
            f"Expected {validation}."
        )


class TypeValidationError(BaseValidationError):
    """Error on type validation."""

    @classmethod
    def message(cls, name: str, value: str, validation: str | list[str]) -> str:
        return f"Type '{value}' provided for '{name}' is invalid." + iterable_message(
            validation
        )


class UUIDValidationError(BaseValidationError):
    """Error on uuid string validation."""

    @classmethod
    def message(cls, name: str, value: str, *_) -> str:
        return f"Parameter '{name}' with value '{value}' is not a valid uuid string."


class ValueValidationError(BaseValidationError):
    """Error on value validation."""

    @classmethod
    def message(cls, name: str, value: Any, validation: list[Any]) -> str:
        return f"Value '{value}' provided for '{name}' is invalid." + iterable_message(
            validation
        )


class MemoryValidationError(BaseValidationError):
    """Error on memory validation."""

    @classmethod
    def message(cls, name: str, value: Any, validation: float) -> str:
        return (
            f"Parameter '{name}' with value '{value}' "
            f"exceeds memory limit of {validation / 1e6} MB."
        )


def iterable_message(valid: str | list[Any] | None) -> str:
    """
    Append possibly iterable valid: "Must be (one of): {valid}".

    :param valid: Valid value(s) to include in message. Can be a string, list of values, or None.

    :return: Message string indicating valid value(s).
    """

    if valid is None:
        return ""
    if isinstance(valid, str):
        return f" Must be: '{valid}'."
    if iterable(valid, checklen=True):
        vstr = "'" + "', '".join(str(k) for k in valid) + "'"
        return f" Must be one of: {vstr}."

    return f" Must be: '{valid[0]}'."


def iterable(value: Any, checklen: bool = False) -> bool:
    """
    Checks if object is iterable.

    Parameters
    ----------
    value : Object to check for iterableness.
    checklen : Restrict objects with __iter__ method to len > 1.

    Returns
    -------
    True if object has __iter__ attribute but is not string or dict type.
    """
    only_array_like = (not isinstance(value, str)) & (not isinstance(value, dict))
    if (hasattr(value, "__iter__")) & only_array_like:
        return not (checklen and (len(value) == 1))

    return False
