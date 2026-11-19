# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from geoh5py import Workspace
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared.entity import Entity
from geoh5py.ui_json.forms import BoolForm, DataOrValueForm


if TYPE_CHECKING:
    from geoh5py.ui_json.ui_json import UIJson

Validation = dict[str, Any]


## Validation utility for UIJson class ##
class UIJsonError(Exception):
    """Exception raised for errors in the UIJson object."""

    def __init__(self, message: str):
        super().__init__(message)


class ErrorPool:  # pylint: disable=too-few-public-methods
    """
    Stores validation errors for all UIJson members.

    :param errors: Dictionary mapping parameter names to lists of
        exceptions encountered during validation.
    """

    def __init__(self, errors: dict[str, list[Exception]]):
        self.pool = errors

    def _format_error_message(self):
        """Format the error message for the UIJsonError."""

        msg = ""
        for key, errors in self.pool.items():
            if errors:
                msg += f"\t{key}:\n"
                for i, error in enumerate(errors):
                    msg += f"\t\t{i}. {error}\n"

        return msg

    def throw(self):
        """Raise the UIJsonError with detailed list of errors per parameter."""

        message = self._format_error_message()
        if message:
            message = "Collected UIJson errors:\n" + message
            raise UIJsonError(message)


def dependency_type_validation(name: str, ui_json: UIJson):
    """
    Validate that the form depending on is optional, group_optional or bool type.

    :param name: Name of the form
    :param ui_json: A UIJson object.
    """
    dependency_form = getattr(ui_json, name)

    if not (
        dependency_form.optional
        or isinstance(dependency_form, BoolForm)
        or dependency_form.group_optional
    ):
        raise UIJsonError(
            f"Dependency form '{name}' must be either optional, group_optional or of boolean type."
        )


def get_validations(form: list[str]) -> list[Callable]:
    """
    Get callable validations based on identifying form keys.

    :param form: Form to validate sub-keys with

    :return: List of callable validations.
    """
    return [VALIDATIONS_MAP[k] for k in form if k in VALIDATIONS_MAP]


def is_of_typed_value(values: Entity | list[Entity], types: type | list[type]) -> bool:
    """
    Common function to test typed values.

    :param values: Entity or list of values to test
    :param types: Types to test against

    :return: Logic whether the type(s) is respected
    """
    if not isinstance(types, list):
        types = [types]

    if not isinstance(values, list):
        values = [values]

    for val in values:
        if not isinstance(val, tuple(types)):
            return False

    return True


def data_type_validation(name: str, data: dict[str, Any], ui_json: UIJson):
    """
    Validate that value is one of the provided data types.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)

    # For is_value case
    if getattr(form, "is_value", False):
        return

    data_types = form.data_type

    if not isinstance(data_types, list):
        data_types = [data_types]

    data_types = [elem.value for elem in data_types]

    values = data[name]

    if isinstance(values, PropertyGroup):
        if values.properties is None:
            return
        values = [values.parent.get_data(data)[0] for data in values.properties]

    elif isinstance(values, dict):
        if "property" in values:
            values = values["property"]
        elif "value" in values:
            # Don't validate for DrillholeGroupDataForm as selected
            # data *name* is not of Data type
            if "group_value" in values:
                return

            values = values["value"]

    if not is_of_typed_value(values, data_types):
        raise UIJsonError(f"Data type must be one of {data_types}.")


def mesh_type_validation(name: str, data: dict[str, Any], ui_json: UIJson):
    """
    Validate that value is one of the provided mesh types.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)
    mesh_types = form.mesh_type

    if not is_of_typed_value(data[name], mesh_types):
        raise UIJsonError(f"Object's mesh type must be one of {mesh_types}.")


def group_type_validation(name: str, data: dict[str, Any], ui_json: UIJson):
    """
    Validate that value is one of the provided group types.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)
    group_types = form.group_type

    values = data[name]

    if isinstance(values, dict):
        values = [values["group_value"]]

    if not is_of_typed_value(values, group_types):
        raise UIJsonError(f"Group's group type must be one of {group_types}.")


def parent_validation(name: str, data: dict[str, Any], ui_json: UIJson):
    """
    Validate that the data is a child of the parent object.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)
    if isinstance(form, DataOrValueForm) and form.is_value:
        return

    parent_name = form.parent
    child = data[name]

    # Special case for DataRangeForm
    if isinstance(child, dict):
        child = data[name]["property"]

    parent = data[parent_name]

    child = child if isinstance(child, list) else [child]
    if (
        not isinstance(parent, ObjectBase)
        or len(list(set(child) - set(parent.children))) > 0
    ):
        raise UIJsonError(f"{name} data is not a child of {parent_name}.")


def promote_or_catch(
    workspace: Workspace,
    value: Any,
) -> Any:
    """
    Returns an object if it exists in the workspace or an error if not.

    :param workspace: Workspace to fetch entities from.
    :param value: UUID of the object to fetch.

    :returns: The object if it exists in the workspace or a placeholder error
        to be collected and raised later with any other UIJson level validation
        errors.
    """
    if isinstance(value, list | tuple):
        return [promote_or_catch(workspace, val) for val in value]

    if isinstance(value, dict):
        return {key: promote_or_catch(workspace, val) for key, val in value.items()}

    if not isinstance(value, UUID):
        return value

    obj = workspace.get_entity(value)
    if obj[0] is not None:
        return obj[0]

    raise UIJsonError(f"Workspace does not contain an entity with uid: {value}.")


VALIDATIONS_MAP = {
    "data_type": data_type_validation,
    "mesh_type": mesh_type_validation,
    "group_type": group_type_validation,
    "parent": parent_validation,
}
