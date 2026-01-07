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

from typing import Any

from geoh5py.data import Data
from geoh5py.objects import ObjectBase


class UIJsonError(Exception):
    """Exception raised for errors in the UIJson object."""

    def __init__(self, message: str):
        super().__init__(message)


class ErrorPool:  # pylint: disable=too-few-public-methods
    """Stores validation errors for all UIJson members."""

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


def dependency_type_validation(name: str, data: dict[str, Any], params: dict[str, Any]):
    """Dependency is optional or bool type."""

    dependency = params[name]["dependency"]
    dependency_form = params[dependency]
    if "optional" not in dependency_form and not isinstance(data[dependency], bool):
        raise UIJsonError(
            f"Dependency {dependency} must be either optional or of boolean type."
        )


def mesh_type_validation(name: str, data: dict[str, Any], params: dict[str, Any]):
    """Promoted value is one of the provided mesh types."""

    mesh_types = params[name]["mesh_type"]
    obj = data[name]
    if not isinstance(obj, tuple(mesh_types)):
        raise UIJsonError(f"Object's mesh type must be one of {mesh_types}.")


def parent_validation(name: str, data: dict[str, Any], params: dict[str, Any]):
    """Data is a child of the parent object."""

    form = params[name]
    child = data[name]
    parent = data[form["parent"]]

    if not isinstance(parent, ObjectBase) or (
        isinstance(child, Data) and parent.get_entity(child.uid)[0] is None
    ):
        raise UIJsonError(f"{name} data is not a child of {form['parent']}.")
