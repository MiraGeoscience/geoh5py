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
from typing import Any

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    BaseValidationError,
    InCollectionValidationError,
    RequiredFormMemberValidationError,
    RequiredObjectDataValidationError,
    RequiredUIJsonParameterValidationError,
    RequiredWorkspaceObjectValidationError,
    TypeUIDValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.shared.utils import is_uuid


class EnforcerPool:
    """
    Validate data on a collection of enforcers.

    :param name: Name of parameter.
    :param enforcers: List (pool) of enforcers.
    """

    enforcer_types = [
        "type",
        "value",
        "uuid",
        "type_uid",
        "required",
        "required_uijson_parameters",
        "required_form_members",
        "required_workspace_objects",
        "required_object_data",
    ]

    def __init__(self, name: str, enforcers: list[Enforcer]):
        self.name = name
        self.enforcers: list[Enforcer] = enforcers
        self._errors: list[BaseValidationError] = []

    @classmethod
    def from_validations(
        cls,
        name: str,
        validations: dict[str, Any],
    ) -> EnforcerPool:
        """
        Create enforcers pool from validations.

        :param name: Name of parameter.
        :param validations: Encodes validations as enforcer type and
            validation key value pairs.
        :param restricted_validations: 0.

        """

        return cls(name, cls._recruit(validations))

    @property
    def validations(self) -> dict[str, Any]:
        """Returns an enforcer type / validation dictionary from pool."""
        return {k.enforcer_type: k.validations for k in self.enforcers}

    @staticmethod
    def _recruit(validations: dict[str, Any]):
        """Recruit enforcers from validations."""
        return [EnforcerPool._recruit_enforcer(k, v) for k, v in validations.items()]

    @staticmethod
    def _recruit_enforcer(enforcer_type: str, validation: Any) -> Enforcer:
        """
        Create enforcer from enforcer type and validation.

        :param enforcer_type: Type of enforcer to create.
        :param validation: Enforcer validation.
        """

        if enforcer_type not in EnforcerPool.enforcer_types:
            raise ValueError(f"Invalid enforcer type: {enforcer_type}.")

        if enforcer_type == "type":
            enforcer = TypeEnforcer(validation)
        if enforcer_type == "value":
            enforcer = ValueEnforcer(validation)  # type: ignore
        if enforcer_type == "uuid":
            enforcer = UUIDEnforcer(validation)  # type: ignore
        if enforcer_type == "type_uid":
            enforcer = TypeUIDEnforcer(validation)  # type: ignore
        if enforcer_type == "required":
            enforcer = RequiredEnforcer(validation)  # type: ignore
        if enforcer_type == "required_uijson_parameters":
            enforcer = RequiredUIJsonParameterEnforcer(validation)  # type: ignore
        if enforcer_type == "required_form_members":
            enforcer = RequiredFormMemberEnforcer(validation)  # type: ignore
        if enforcer_type == "required_workspace_objects":
            enforcer = RequiredWorkspaceObjectEnforcer(validation)  # type: ignore
        if enforcer_type == "required_object_data":
            enforcer = RequiredObjectDataEnforcer(validation)  # type: ignore

        return enforcer

    def enforce(self, value: Any):
        """Enforce rules from all enforcers in the pool."""

        for enforcer in self.enforcers:
            self._capture_error(enforcer, value)

        self._raise_errors()

    def _capture_error(self, enforcer: Enforcer, value: Any):
        """Catch and store 'BaseValidationError's for aggregation."""
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
    """
    Base class for rule enforcers.

    :param enforcer_type: Type of enforcer.
    :param validations: Value(s) to validate parameter value against.
    """

    enforcer_type: str = ""

    def __init__(self, validations: Any | None = None):
        self.validations = validations

    @abstractmethod
    def rule(self, value: Any):
        """True if 'value' adheres to enforcers rule."""

    @abstractmethod
    def enforce(self, name: str, value: Any):
        """Enforces rule on 'name' parameter's 'value'."""

    @property
    def validations(self) -> Any:
        """Enforcer validation."""
        return self._validations

    @validations.setter
    def validations(self, val: Any):
        self._validations = val

    def __eq__(self, other) -> bool:
        """Equal if same type and validations."""

        is_equal = False
        if isinstance(other, type(self)):
            is_equal = other.validations == self.validations

        return is_equal

    def __str__(self):
        return f"<{type(self).__name__}> : {self.validations}"


class TypeEnforcer(Enforcer):
    """
    Enforces valid type(s).

    :param validations: Valid type(s) for parameter value.
    :raises TypeValidationError: If value is not a valid type.
    """

    enforcer_type: str = "type"

    def __init__(self, validations: type | list[type]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce type validation."""

        if not self.rule(value):
            raise TypeValidationError(
                name, type(value).__name__, [k.__name__ for k in self.validations]
            )

    def rule(self, value) -> bool:
        """True if value is one of the valid types."""
        return any(isinstance(value, k) for k in self.validations + [type(None)])

    @property
    def validations(self) -> list[type]:
        if self._validations is None:
            self._validations = []
        return self._validations

    @validations.setter
    def validations(self, val: type | list[type]):
        if not isinstance(val, list):
            val = [val]
        self._validations = val


class ValueEnforcer(Enforcer):
    """
    Enforces restricted value choices.

    :param validations: Valid parameter value(s).
    :raises ValueValidationError: If value is not a valid value
        choice.
    """

    enforcer_type = "value"

    def __init__(self, validations: list[Any]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce value validation."""

        if not self.rule(value):
            raise ValueValidationError(name, value, self.validations)

    def rule(self, value: Any) -> bool:
        """True if value is a valid choice."""
        return value in self.validations

    @property
    def validations(self) -> list[Any]:
        if self._validations is None:
            self._validations = []
        return self._validations

    @validations.setter
    def validations(self, val: Any):
        if not isinstance(val, list):
            val = [val]
        self._validations = val


class TypeUIDEnforcer(Enforcer):
    """
    Enforces restricted geoh5 entity_type uid(s).


    :param validations: Valid geoh5py object type uid(s).

    :raises TypeValidationError: If value is not a valid type uid.
    """

    enforcer_type = "type_uid"

    def __init__(self, validations: list[str]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce type uid validation."""

        if not self.rule(value):
            raise TypeUIDValidationError(name, value, self.validations)

    def rule(self, value: Any) -> bool:
        """True if value is a valid type uid."""
        return (
            self.validations == [""]
            or str(value.default_type_uid()) in self.validations
        )


class UUIDEnforcer(Enforcer):
    """
    Enforces valid uuid string.

    :param validations: None is considered a valid uuid if
        validations is 'optional'.
    :raises UUIDValidationError: If value is not a valid uuid string.
    """

    enforcer_type = "uuid"

    def __init__(self, validations: str | None = None):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to check if valid uuid."""

        if not self.rule(value):
            raise UUIDValidationError(
                name,
                value,
            )

    def rule(self, value: Any) -> bool:
        """True if value is a valid uuid string."""

        if value is None:
            return True

        return is_uuid(value)


class RequiredEnforcer(Enforcer):
    """
    Enforces required items in a collection.

    :param validations: Items that are required in the

        collection.
    :raises InCollectionValidationError: If collection is missing one of
        the required parameters/members.
    """

    enforcer_type = "required"
    validation_error = InCollectionValidationError

    def __init__(self, validations: list[str]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to check if required items in collection."""

        if not self.rule(value):
            raise self.validation_error(
                name,
                [k for k in self.validations if k not in value],
            )

    def rule(self, value: Any) -> bool:
        """True if all required parameters are in 'value' collection."""
        return all(k in value for k in self.validations)

    @property
    def validations(self) -> list[str]:
        if self._validations is None:
            self._validations = []
        return self._validations

    @validations.setter
    def validations(self, val: str | list[str]):
        if isinstance(val, str):
            val = [val]
        self._validations = val


class RequiredUIJsonParameterEnforcer(RequiredEnforcer):
    enforcer_type = "required_uijson_parameters"
    validation_error = RequiredUIJsonParameterValidationError


class RequiredFormMemberEnforcer(RequiredEnforcer):
    enforcer_type = "required_form_members"
    validation_error = RequiredFormMemberValidationError


class RequiredWorkspaceObjectEnforcer(RequiredEnforcer):
    enforcer_type = "required_workspace_objects"
    validation_error = RequiredWorkspaceObjectValidationError


class RequiredObjectDataEnforcer(RequiredEnforcer):
    enforcer_type = "required_object_data"
    validation_error = RequiredObjectDataValidationError
