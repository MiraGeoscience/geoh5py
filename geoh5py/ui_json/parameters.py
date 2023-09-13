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

from geoh5py.ui_json.enforcers import EnforcerPool

Validation = Dict[str, Any]


class Parameter:
    """
    Basic parameter to store key/value data with validation capabilities.

    :param name: Parameter name.
    :param value: The parameters value.
    :param enforcers: A collection of enforcers.
    """

    validations: dict[str, Any] = {}

    def __init__(
        self, name: str, value: Any = None, enforcers: EnforcerPool | None = None
    ):
        self.name: str = name
        self._enforcers: EnforcerPool = self._get_enforcer_pool(enforcers)
        setattr(self, "_value" if value is None else "value", value)

    def _get_enforcer_pool(self, enforcers: EnforcerPool | None) -> EnforcerPool:
        """Updates incoming enforcers with base enforcer instances."""

        if enforcers is None:
            pool = EnforcerPool.from_validations(self.name, self.validations)
        else:
            pool = EnforcerPool.from_validations(
                self.name, dict(enforcers.validations, **self.validations)
            )

        return pool

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self.validate()

    def validate(self):
        """Validates data against the pool of enforcers."""
        self._enforcers.enforce(self.value)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class StringParameter(Parameter):
    """Parameter for string values."""

    validations = {"type": [str]}


class IntegerParameter(Parameter):
    """Parameter for integer values."""

    validations = {"type": [int]}


class FloatParameter(Parameter):
    """Parameter for float values."""

    validations = {"type": [float]}


class NumericParameter(Parameter):
    """Parameter for generic numeric values."""

    validations = {"type": [int, float]}


class BoolParameter(Parameter):
    """Parameter for boolean values."""

    validations = {"type": [bool]}


class UUIDParameter(Parameter):
    """Parameter for UUID values."""

    validations = {"type": [str, UUID], "uuid": None}


class StringListParameter(Parameter):
    """Parameter for list of strings."""

    validations = {"type": [list, str]}

    # TODO: introduce type alias handling so that TypeEnforcer(list[str], str)
    # is possible
