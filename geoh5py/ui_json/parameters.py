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

from typing import Any
from uuid import UUID

from geoh5py import Workspace
from geoh5py.groups import PropertyGroup
from geoh5py.shared.utils import SetDict
from geoh5py.ui_json.enforcers import EnforcerPool

Validation = dict[str, Any]


class Parameter:
    """
    Basic parameter to store key/value data with validation capabilities.

    :param name: Parameter name.
    :param value: The parameters value.
    :param enforcers: A collection of enforcers.
    """

    static_validations: dict[str, Any] = {}

    def __init__(self, name: str, value: Any = None):
        self.name: str = name
        self._value: Any | None = None
        self._enforcers: EnforcerPool = EnforcerPool.from_validations(
            self.name, self.validations
        )
        if value is not None:
            self.value = value

    @property
    def validations(self):
        """Returns a dictionary of static validations."""
        return SetDict(**self.static_validations)

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


class DynamicallyRestrictedParameter(Parameter):
    """Parameter whose validations are set at runtime."""

    def __init__(
        self, name: str, restrictions: Any, enforcer_type="type", value: Any = None
    ):
        self._restrictions = restrictions
        self._enforcer_type = enforcer_type
        super().__init__(name, value)

    @property
    def restrictions(self):
        if not isinstance(self._restrictions, list):
            self._restrictions = {self._restrictions}

        return self._restrictions

    @property
    def validations(self):
        return SetDict(**{self._enforcer_type: self.restrictions})


class ValueRestrictedParameter(DynamicallyRestrictedParameter):
    """Parameter with a restricted set of values."""

    def __init__(self, name: str, restrictions: Any, value: Any = None):
        super().__init__(name, restrictions, "value", value)


class TypeRestrictedParameter(DynamicallyRestrictedParameter):
    """Parameter with a restricted set of types known at runtime only."""

    def __init__(self, name: str, restrictions: list[Any], value: Any = None):
        super().__init__(name, restrictions, "type", value)


class TypeUIDRestrictedParameter(DynamicallyRestrictedParameter):
    """Parameter with a restricted set of type uids known at runtime only."""

    def __init__(self, name: str, restrictions: list[UUID], value: Any = None):
        super().__init__(name, restrictions, "type_uid", value)


class StringParameter(Parameter):
    """Parameter for string values."""

    static_validations = {"type": str}


class IntegerParameter(Parameter):
    """Parameter for integer values."""

    static_validations = {"type": int}


class FloatParameter(Parameter):
    """Parameter for float values."""

    static_validations = {"type": float}


class NumericParameter(Parameter):
    """Parameter for generic numeric values."""

    static_validations = {"type": [int, float]}


class BoolParameter(Parameter):
    """Parameter for boolean values."""

    static_validations = {"type": bool}

    def __init__(self, name: str, value: bool = False):
        super().__init__(name, value)


class StringListParameter(Parameter):
    """Parameter for list of strings."""

    static_validations = {"type": [list, str]}

    # TODO: introduce type alias handling so that TypeEnforcer(list[str], str)
    # is possible


class WorkspaceParameter(Parameter):
    """Parameter for workspace objects."""

    static_validations = {"type": Workspace}


class PropertyGroupParameter(Parameter):
    """Parameter for property group objects."""

    static_validations = {"type": PropertyGroup}
