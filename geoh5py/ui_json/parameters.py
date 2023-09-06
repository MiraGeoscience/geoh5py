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

from typing import Any, Dict
from uuid import UUID

from geoh5py.ui_json.enforcers import Enforcer, TypeEnforcer, UUIDEnforcer
from geoh5py.ui_json.validation import Validations

Validation = Dict[str, Any]


class Parameter:
    """
    Basic parameter to store key/value data with validation capabilities.

    :param name: Parameter name.
    :param value: The parameters value.
    :param validations: Parameter validations
    """

    enforcers: list[Enforcer] = []

    def __init__(self, name, value=None, validations: Validations | None = None):
        self.name: str = name
        self._validations: Validations = self._setup_validations(validations)
        setattr(self, "_value" if value is None else "value", value)

    def _setup_validations(self, validations: Validations | None) -> Validations:
        """Updates validations enforcers with base enforcer instances."""

        if validations is not None:
            validations.enforcers = self._drop_class_enforcers(validations.enforcers)
        elif self.enforcers:
            validations = Validations(self.name, self.enforcers)

        return validations  # type: ignore

    def _drop_class_enforcers(self, enforcers: list[Enforcer]) -> list[Enforcer]:
        """Removes special 'class' enforcers from incoming enforcer list."""
        types = tuple(type(k) for k in self.enforcers)
        return [k for k in enforcers if not isinstance(k, types)] + self.enforcers

    @property
    def validations(self):
        return self._validations

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self.validate()

    def validate(self):
        if self.validations is None:
            raise AttributeError("Must set validations before calling validate.")
        self.validations.validate(self.value)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class TypedParameter(Parameter):
    """Parameter for typed values."""

    def __init__(
        self,
        name,
        value=None,
        validations: Validations | None = None,
        optional: bool = False,
    ):
        if optional:
            self._add_nonetype_enforcer()

        super().__init__(name, value=value, validations=validations)

    def _add_nonetype_enforcer(self):
        self.enforcers[0].validations.append(type(None))


class StringParameter(TypedParameter):
    """Parameter for string values."""

    enforcers: list[Enforcer] = [TypeEnforcer(str)]


class IntegerParameter(TypedParameter):
    """Parameter for integer values."""

    enforcers: list[Enforcer] = [TypeEnforcer(int)]


class FloatParameter(TypedParameter):
    """Parameter for float values."""

    enforcers: list[Enforcer] = [TypeEnforcer(float)]


class NumericParameter(TypedParameter):
    """Parameter for generic numeric values."""

    enforcers: list[Enforcer] = [TypeEnforcer([int, float])]


class BoolParameter(TypedParameter):
    """Parameter for boolean values."""

    enforcers: list[Enforcer] = [TypeEnforcer(bool)]


class UUIDParameter(TypedParameter):
    enforcers: list[Enforcer] = [TypeEnforcer([str, UUID]), UUIDEnforcer()]

    def _add_nonetype_enforcer(self):
        self.enforcers[0].validations.append(type(None))
        self.enforcers[1] = UUIDEnforcer("optional")


class StringListParameter(TypedParameter):
    """Parameter for list of strings."""

    enforcers: list[Enforcer] = [TypeEnforcer([list, str])]
    # TODO: introduce type alias handling so that TypeEnforcer(list[str], str)
    # is possible
