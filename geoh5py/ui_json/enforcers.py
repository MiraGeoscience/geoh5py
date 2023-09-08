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

import uuid
from abc import ABC, abstractmethod
from typing import Any

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    BaseValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)


class EnforcerPool:
    """
    Validate values on a collection of enforcers.

    :param name: Name of parameter.
    :param enforcers: List (pool) of enforcers.
    """

    def __init__(self, name: str, enforcers: list[Enforcer] | None = None):
        self.name = name
        self.enforcers = enforcers or []
        self._errors: list[BaseValidationError] = []

    @classmethod
    def from_validations(
        cls, name: str, validations: dict[str, Any], protected: list[str] | None = None
    ):
        """
        Create enforcers pool from validations.

        :param name: Name of parameter.
        :param validations: Encodes validations as enforcer type and
            validation key value pairs.
        """
        enforcers = cls(name)
        enforcers.update(validations, protected)
        return enforcers

    def update(self, validations: dict[str, Any], protected: list[str] | None = None):
        """
        Create enforcers pool from name/validation dictionary.

        :param validations: Encodes validations as enforcer type and
            validation key value pairs.
        :param protected: Excludes listed enforcer types from updating

        """
        protected = protected or []
        enforcers = [k for k in self.enforcers if k.enforcer_type in protected]
        for name, validation in validations.items():
            if name not in protected:
                enforcers.append(self._recruit_enforcer(name, validation))

        self.enforcers = enforcers

    @property
    def validations(self) -> dict[str, Any]:
        return {k.enforcer_type: k.validations for k in self.enforcers}

    def _recruit_enforcer(self, enforcer_type: str, validation: Any) -> Enforcer:
        """Create enforcer from validation dictionary."""
        if enforcer_type == "type":
            return TypeEnforcer(validation)
        if enforcer_type == "value":
            return ValueEnforcer(validation)
        if enforcer_type == "uuid":
            return UUIDEnforcer(validation)

        raise ValueError(f"Invalid enforcer type: {validation['type']}")

    def validate(self, value: Any):
        """Validate value against all enforcers."""

        for enforcer in self.enforcers:
            self._capture_error(enforcer, value)

        self._raise_errors()

    def _capture_error(self, enforcer: Enforcer, value: Any):
        """Catch 'BaseValidationError' and return error."""
        try:
            enforcer.enforce(self.name, value)
        except BaseValidationError as err:
            self._errors.append(err)

    def _raise_errors(self):
        """Raise errors if any exist, aggregate if more than one."""
        if self._errors:
            if len(self._errors) > 1:
                raise AggregateValidationError(self.name, self._errors)
            raise self._errors.pop()


class Enforcer(ABC):
    """Base class for enforcers."""

    enforcer_type: str = ""

    def __init__(self, validations: Any | None = None):
        self._validations = validations

    @abstractmethod
    def rule(self, value: Any):
        raise NotImplementedError(
            "Enforcer is abstract. Must be implemented in subclass"
        )

    @abstractmethod
    def enforce(self, name: str, value: Any):
        raise NotImplementedError(
            "Enforcer is abstract. Must be implemented in subclass"
        )

    @property
    def validations(self):
        return self._validations

    def __eq__(self, other):
        """Equal if same type and validations."""

        is_equal = False
        if isinstance(other, type(self)):
            is_equal = other.validations == self.validations

        return is_equal

    def __str__(self):
        return f"<{type(self).__name__}> : {self.validations}"


class TypeEnforcer(Enforcer):
    """Enforces type validation."""

    enforcer_type: str = "type"

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce type validation."""
        if not self.rule(value):
            raise TypeValidationError(
                name, type(value).__name__, self._stringify(self.validations)
            )

    def rule(self, value):
        """True if value is one of the valid types."""
        return any(isinstance(value, k) for k in self.validations)

    @property
    def validations(self):
        if not isinstance(self._validations, list):
            self._validations = [self._validations]

        return self._validations

    def _stringify(self, target: type | list[type]) -> str | list[str]:
        """Converts type(s) to string for error message."""

        if isinstance(target, list):
            return [k.__name__ for k in target]

        if isinstance(target, type):
            return target.__name__

        raise TypeError(f"Cannot stringify type {type(target)}")


class ValueEnforcer(Enforcer):
    enforcer_type = "value"

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce value validation."""
        if not self.rule(value):
            raise ValueValidationError(name, value, self.validations)

    def rule(self, value: Any):
        """True if value is a valid choice."""
        return value in self.validations


class UUIDEnforcer(Enforcer):
    enforcer_type = "uuid"

    def enforce(self, name: str, value: Any):
        """Administers rule to check if valid uuid."""
        if not self.rule(value):
            raise UUIDValidationError(
                name,
                value,
            )

    def rule(self, value: Any):
        """True if value is a valid uuid string."""

        if self.validations == "optional":
            return True

        if not isinstance(value, str):
            return False

        try:
            uuid.UUID(value)
            is_uuid = True
        except ValueError:
            is_uuid = False

        return is_uuid
