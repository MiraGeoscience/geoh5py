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
from typing import Any
from uuid import UUID

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
from geoh5py.shared.utils import SetDict, is_uuid


class Enforcer(ABC):
    """
    Base class for rule enforcers.

    :param enforcer_type: Type of enforcer.
    :param validations: Value(s) to validate parameter value against.
    """

    enforcer_type: str = ""

    def __init__(self, validations: set):
        self.validations = validations

    @abstractmethod
    def rule(self, value: Any):
        """True if 'value' adheres to enforcers rule."""

    @abstractmethod
    def enforce(self, name: str, value: Any):
        """Enforces rule on 'name' parameter's 'value'."""

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

    def __init__(self, validations: set[type]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce type validation."""

        if not self.rule(value):
            raise TypeValidationError(
                name, type(value).__name__, [k.__name__ for k in self.validations]
            )

    def rule(self, value) -> bool:
        """True if value is one of the valid types."""
        return any(isinstance(value, k) for k in self.validations.union({type(None)}))


class ValueEnforcer(Enforcer):
    """
    Enforces restricted value choices.

    :param validations: Valid parameter value(s).
    :raises ValueValidationError: If value is not a valid value
        choice.
    """

    enforcer_type = "value"

    def __init__(self, validations: set[Any]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce value validation."""

        if not self.rule(value):
            raise ValueValidationError(name, value, list(self.validations))

    def rule(self, value: Any) -> bool:
        """True if value is a valid choice."""
        return value in self.validations


class TypeUIDEnforcer(Enforcer):
    """
    Enforces restricted geoh5 entity_type uid(s).

    :param validations: Valid geoh5py object type uid(s).
    :raises TypeValidationError: If value is not a valid type uid.
    """

    enforcer_type = "type_uid"

    def __init__(self, validations: set[str]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to enforce type uid validation."""

        if not self.rule(value):
            raise TypeUIDValidationError(name, value, list(self.validations))

    def rule(self, value: Any) -> bool:
        """True if value is a valid type uid."""
        return self.validations == {""} or value.default_type_uid() in [
            UUID(k) for k in self.validations
        ]


class UUIDEnforcer(Enforcer):
    """
    Enforces valid uuid string.

    :param validations: No validations needed, can be empty set.
    :raises UUIDValidationError: If value is not a valid uuid string.
    """

    enforcer_type = "uuid"

    def __init__(self, validations=None):
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

    :param validations: Items that are required in the collection.
    :raises InCollectionValidationError: If collection is missing one of
        the required parameters/members.
    """

    enforcer_type = "required"
    validation_error = InCollectionValidationError

    def __init__(self, validations: set[str | tuple[str, str]]):
        super().__init__(validations)

    def enforce(self, name: str, value: Any):
        """Administers rule to check if required items in collection."""

        if not self.rule(value):
            raise self.validation_error(
                name,
                [k for k in self.validations if k not in self.collection(value)],
            )

    def rule(self, value: Any) -> bool:
        """True if all required parameters are in 'value' collection."""
        return all(k in self.collection(value) for k in self.validations)

    def collection(self, value: Any) -> list[Any]:
        """Returns the collection to check for required items."""
        return value


class RequiredUIJsonParameterEnforcer(RequiredEnforcer):
    enforcer_type = "required_uijson_parameters"
    validation_error = RequiredUIJsonParameterValidationError


class RequiredFormMemberEnforcer(RequiredEnforcer):
    enforcer_type = "required_form_members"
    validation_error = RequiredFormMemberValidationError


class RequiredWorkspaceObjectEnforcer(RequiredEnforcer):
    enforcer_type = "required_workspace_object"
    validation_error = RequiredWorkspaceObjectValidationError

    def rule(self, value: Any) -> bool:
        """True if all objects are in the workspace."""
        validations = [value[k]["value"].uid for k in self.validations]
        return all(k in self.collection(value) for k in validations)

    def collection(self, value: dict[str, Any]) -> list[UUID]:
        return list(value["geoh5"].list_entities_name)


class RequiredObjectDataEnforcer(Enforcer):
    enforcer_type = "required_object_data"
    validation_error = RequiredObjectDataValidationError

    def enforce(self, name: str, value: Any):
        """Administers rule to check if required items in collection."""

        if not self.rule(value):
            raise self.validation_error(
                name,
                [
                    k
                    for i, k in enumerate(self.validations)
                    if value[k[1]]["value"].uid not in self.collection(value)[i]
                ],
            )

    def rule(self, value: Any) -> bool:
        """True if object/data have parent/child relationship."""
        return all(
            value[k[1]]["value"].uid in self.collection(value)[i]
            for i, k in enumerate(self.validations)
        )

    def collection(self, value: dict[str, Any]) -> list[list[UUID]]:
        """Returns list of children for all parents in validations."""
        return [
            [c.uid for c in value[k[0]]["value"].children] for k in self.validations
        ]


class EnforcerPool:
    """
    Validate data on a collection of enforcers.

    :param name: Name of parameter.
    :param enforcers: List (pool) of enforcers.
    """

    enforcer_types = {
        "type": TypeEnforcer,
        "value": ValueEnforcer,
        "uuid": UUIDEnforcer,
        "type_uid": TypeUIDEnforcer,
        "required": RequiredEnforcer,
        "required_uijson_parameters": RequiredUIJsonParameterEnforcer,
        "required_form_members": RequiredFormMemberEnforcer,
        "required_workspace_object": RequiredWorkspaceObjectEnforcer,
        "required_object_data": RequiredObjectDataEnforcer,
    }

    def __init__(self, name: str, enforcers: list[Enforcer]):
        self.name = name
        self.enforcers: list[Enforcer] = enforcers
        self._errors: list[BaseValidationError] = []

    @classmethod
    def from_validations(
        cls,
        name: str,
        validations: SetDict,
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
    def validations(self) -> SetDict:
        """Returns an enforcer type / validation dictionary from pool."""
        return SetDict(**{k.enforcer_type: k.validations for k in self.enforcers})

    @staticmethod
    def _recruit(validations: SetDict):
        """Recruit enforcers from validations."""
        return [EnforcerPool._recruit_enforcer(k, v) for k, v in validations.items()]

    @staticmethod
    def _recruit_enforcer(enforcer_type: str, validation: set) -> Enforcer:
        """
        Create enforcer from enforcer type and validation.

        :param enforcer_type: Type of enforcer to create.
        :param validation: Enforcer validation.
        """

        if enforcer_type not in EnforcerPool.enforcer_types:
            raise ValueError(f"Invalid enforcer type: {enforcer_type}.")

        return EnforcerPool.enforcer_types[enforcer_type](validation)

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
