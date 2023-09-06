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
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)


class Enforcer(ABC):
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


class TypeEnforcer(Enforcer):
    """Enforces type validation."""

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
    def enforce(self, name: str, value: Any):
        """Administers rule to enforce value validation."""
        if not self.rule(value):
            raise ValueValidationError(name, value, self.validations)

    def rule(self, value: Any):
        """True if value is a valid choice."""
        return value in self.validations


class UUIDEnforcer(Enforcer):
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
