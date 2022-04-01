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
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

import numpy as np

from geoh5py.groups import PropertyGroup
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    OptionalValidationError,
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.shared.utils import iterable
from geoh5py.workspace import Workspace


class BaseValidator(ABC):
    """Concrete base class for validators."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __call__(self, *args):
        self.validate(*args)

    @classmethod
    @abstractmethod
    def validate(cls, name: str, value: Any, valid: Any):
        """
        Custom validation function.
        """
        raise NotImplementedError(
            "The 'validate' method must be implemented by the sub-class. "
            f"Must contain a 'name' {name}, 'value' {value} and 'valid' {valid} argument."
        )

    @property
    @classmethod
    @abstractmethod
    def validator_type(cls):
        """Validation type identifier."""
        raise NotImplementedError("Must implement the validator_type property.")


class OptionalValidator(BaseValidator):
    """Validate that forms contain optional parameter if None value is given."""

    validator_type = "optional"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(
        cls,
        name: str,
        value: Any | None,
        valid: bool,
    ) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: True if optional keyword in form for parameter.
        """
        if value is None and not valid:
            raise OptionalValidationError(name, value, valid)


class AssociationValidator(BaseValidator):
    """Validate the association between data and parent object."""

    validator_type = "association"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(
        cls,
        name: str,
        value: Entity | PropertyGroup | UUID | None,
        valid: Entity | Workspace,
    ) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        if valid is None:
            return

        if isinstance(value, UUID):
            uid = value
        elif isinstance(value, (Entity, PropertyGroup)):
            uid = value.uid
        else:
            return

        if isinstance(valid, Workspace):
            # TODO add a generic method to workspace to get all uuid
            children = valid.get_entity(uid)
            if None in children:
                children = valid.fetch_children(valid.root, recursively=True)

        elif isinstance(valid, Entity):
            children = valid.workspace.fetch_children(valid, recursively=True)

        if uid not in [getattr(child, "uid", None) for child in children]:
            raise AssociationValidationError(name, value, valid)


class PropertyGroupValidator(BaseValidator):
    """Validate property_group from parent entity."""

    validator_type = "property_group_type"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: PropertyGroup, valid: str) -> None:
        if value.property_group_type != valid:
            raise PropertyGroupValidationError(name, value, valid)


class AtLeastOneValidator(BaseValidator):

    validator_type = "oneof"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name, value, valid):
        pass


class RequiredValidator(BaseValidator):
    """
    Validate that required keys are present in parameter.
    """

    validator_type = "required"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: Any, valid: bool) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Assert to be required
        """
        if value is None and valid:
            raise RequiredValidationError(name)


class ShapeValidator(BaseValidator):
    """Validate the shape of provided value."""

    validator_type = "shape"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: Any, valid: tuple[int]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: Expected value shape
        """
        if value is None:
            return

        pshape = np.array(value).shape
        if pshape != valid:
            raise ShapeValidationError(name, pshape, valid)


class TypeValidator(BaseValidator):
    """
    Validate the value type from a list of valid types.
    """

    validator_type = "types"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: Any, valid: list[type] | type) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted value types
        """

        if isinstance(valid, type):
            valid = [valid]

        if not iterable(value) or list in valid:
            value = (value,)

        for val in value:
            if not isinstance(val, tuple(valid)):
                valid_names = [t.__name__ for t in valid if hasattr(t, "__name__")]
                type_name = type(val).__name__

                raise TypeValidationError(name, type_name, valid_names)


class UUIDValidator(BaseValidator):
    """Validate a uuui.UUID value or uuid string."""

    validator_type = "uuid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: Any, valid: None = None) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter uuid.
        :param valid: [Optional] Validate uuid from parental entity or known uuids
        """

        if isinstance(value, str):
            try:
                value = UUID(value)
            except ValueError as exception:
                raise UUIDValidationError(name, str(value)) from exception


class ValueValidator(BaseValidator):
    """
    Validator that ensures that values are valid entries.
    """

    validator_type = "values"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, name: str, value: Any, valid: list[float | str]) -> None:
        """
        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param valid: List of accepted values
        """
        if value is None:
            return

        if not isinstance(value, (list, tuple)):
            value = [value]

        for val in value:
            if val is not None and val not in valid:
                raise ValueValidationError(name, val, valid)
