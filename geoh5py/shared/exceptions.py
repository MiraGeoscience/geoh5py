#  Copyright (c) 2021 Mira Geoscience Ltd.
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

from geoh5py.shared.utils import iterable_message


class AssociationValidationError(Exception):
    """Error on association between child and parent entity validation."""

    def __init__(self, name, value, parent):
        super().__init__(
            f"Property '{name}' of type '{value}' must be a child entity of parent {parent}"
        )


class JSONParameterValidationError(Exception):
    """Error on uuid validation."""

    def __init__(self, name, error_message):
        super().__init__(
            f"Malformed ui.json dictionary for parameter '{name}'. {error_message}"
        )


class PropertyGroupValidationError(Exception):
    """Error on property group validation."""

    def __init__(self, name, value, valid):
        super().__init__(
            f"Property group for '{name}' must be of type '{valid}'. "
            f"Provided '{value.name}' of type '{value.property_group_type}'"
        )


class RequiredValidationError(Exception):
    def __init__(self, name):
        super().__init__(f"Missing '{name}' from the list of required parameters.")


class ShapeValidationError(Exception):
    """Error on shape validation."""

    def __init__(self, name, value, valid):
        super().__init__(
            f"Parameter '{name}': 'value' with shape '{value}' was provided. "
            f"Expected len({valid},)."
        )


class TypeValidationError(Exception):
    """Error on type validation."""

    def __init__(self, name, value, valid):
        super().__init__(
            f"Type '{value}' provided for '{name}' is invalid. "
            + iterable_message(valid)
        )


class UUIDValidationError(Exception):
    """Error on uuid validation."""

    def __init__(self, name, value, valid):
        super().__init__(
            f"UUID '{value}' provided for '{name}' is invalid. "
            f"Not in the list of children of {type(valid)}: {valid.name} "
        )


class UUIDStringValidationError(Exception):
    """Error on uuid string validation."""

    def __init__(self, name, value):
        super().__init__(
            f"Parameter '{name}' with value '{value}' is not a valid uuid string."
        )


class ValueValidationError(Exception):
    """Error on value validation."""

    def __init__(self, name, value, valid):
        super().__init__(
            f"Value '{value}' provided for '{name}' is invalid."
            + iterable_message(valid)
        )
