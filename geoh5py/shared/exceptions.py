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
    """Base class for custom exceptions."""

    @classmethod
    @abstractmethod
    def message(cls, name, value, validation):
        """Builds custom error message."""
        raise NotImplementedError()


class JSONParameterValidationError(Exception):
    """Error on uuid validation."""

    def __init__(self, name: str, err: str):
        super().__init__(JSONParameterValidationError.message(name, err))

    @classmethod
    def message(cls, name, err):
        return f"Malformed ui.json dictionary for parameter '{name}'. {err}"


class UIJsonFormatError(BaseValidationError):
    def __init__(self, name, msg):
        super().__init__(f"Invalid UIJson format for parameter '{name}'. {msg}")

    @classmethod
    def message(cls, name, value, validation):
        pass


class AggregateValidationError(BaseValidationError):
    def __init__(
        self,
        name: str,
        value: list[BaseValidationError],
    ):
        super().__init__(AggregateValidationError.message(name, value))

    @classmethod
    def message(cls, name, value, validation=None):
        msg = f"\n\nValidation of '{name}' collected {len(value)} errors:\n"
        for i, err in enumerate(value):
            msg += f"\t{i}. {err!s}\n"
        return msg


class OptionalValidationError(BaseValidationError):
    """Error if None value provided to non-optional parameter."""

    def __init__(
        self,
        name: str,
        value: Any | None,
        validation: bool,
    ):
        super().__init__(OptionalValidationError.message(name, value, validation))

    @classmethod
    def message(cls, name, value, validation):
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

    @classmethod
    def message(cls, name, value, validation):
        return (
            f"Property '{name}' with value: '{value}' must be "
            f"a child entity of parent {validation}"
        )


class PropertyGroupValidationError(BaseValidationError):
    """Error on property group validation."""

    def __init__(self, name: str, value: PropertyGroup, validation: list[str]):
        super().__init__(PropertyGroupValidationError.message(name, value, validation))

    @classmethod
    def message(cls, name, value, validation):
        return (
            f"Property group for '{name}' must be of type '{validation}'. "
            f"Provided '{value.name}' of type '{value.property_group_type}'"
        )


class AtLeastOneValidationError(BaseValidationError):
    def __init__(self, name: str, value: list[str]):
        super().__init__(AtLeastOneValidationError.message(name, value))

    @classmethod
    def message(cls, name, value, validation=None):
        opts = "'" + "', '".join(str(k) for k in value) + "'"
        return f"Must provide at least one {name}.  Options are: {opts}"


class TypeUIDValidationError(BaseValidationError):
    """Error on type uid validation."""

    def __init__(self, name: str, value, validation: list[str]):
        super().__init__(
            TypeUIDValidationError.message(
                name, value.default_type_uid(), list(validation)
            )
        )

    @classmethod
    def message(cls, name, value, validation):
        return (
            f"Type uid '{value}' provided for '{name}' is invalid."
            + iterable_message(validation)
        )


class RequiredValidationError(BaseValidationError):
    def __init__(self, name: str):
        super().__init__(RequiredValidationError.message(name))

    @classmethod
    def message(cls, name, value=None, validation=None):
        return f"Missing required parameter: '{name}'."


class InCollectionValidationError(BaseValidationError):
    collection = "Collection"
    item = "data"

    def __init__(self, name: str, value: list[str]):
        super().__init__(self.message(name, value))

    @classmethod
    def message(cls, name, value, validation=None):
        _ = validation
        return (
            f"{cls.collection}: '{name}' "
            f"is missing required {cls.item}(s): {value}."
        )


class RequiredFormMemberValidationError(InCollectionValidationError):
    collection = "Form"
    item = "member"


class RequiredUIJsonParameterValidationError(InCollectionValidationError):
    collection = "UIJson"
    item = "parameter"


class RequiredWorkspaceObjectValidationError(InCollectionValidationError):
    collection = "Workspace"
    item = "object"


class RequiredObjectDataValidationError(BaseValidationError):
    def __init__(self, name: str, value: list[tuple[str, str]]):
        super().__init__(self.message(name, value))

    @classmethod
    def message(cls, name, value, validation=None):
        _ = validation
        return (
            f"Workspace: '{name}' object(s) {[k[0] for k in value]} "
            f"are missing required children {[k[1] for k in value]}."
        )


class ShapeValidationError(BaseValidationError):
    """Error on shape validation."""

    def __init__(
        self, name: str, value: tuple[int, ...], validation: tuple[int, ...] | str
    ):
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
        return f"Type '{value}' provided for '{name}' is invalid." + iterable_message(
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


def iterable_message(valid: list[Any] | None) -> str:
    """Append possibly iterable valid: "Must be (one of): {valid}."."""
    if valid is None:
        msg = ""
    elif iterable(valid, checklen=True):
        vstr = "'" + "', '".join(str(k) for k in valid) + "'"
        msg = f" Must be one of: {vstr}."
    else:
        msg = f" Must be: '{valid[0]}'."

    return msg


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
