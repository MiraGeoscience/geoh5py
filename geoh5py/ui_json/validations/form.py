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
from uuid import UUID

from geoh5py.shared import Entity


def empty_string_to_none(value):
    """Promote empty string to uid, and pass all other values."""
    if value == "":
        return None
    return value


def entity_to_uuid(value: Any | list[Entity] | Entity) -> Any | list[UUID] | UUID:
    """Demote an Entity to its UUID, and pass all other values."""
    if isinstance(value, list | tuple):
        return [entity_to_uuid(val) for val in value]

    if isinstance(value, Entity):
        return value.uid

    return value


def uuid_to_string(value: UUID | list[UUID] | None) -> str | list[str]:
    """Serialize UUID(s) as a string."""

    def convert(value: UUID | None) -> str:
        if value is None:
            return ""
        if isinstance(value, UUID):
            return f"{{{value!s}}}"
        return value

    if isinstance(value, list | tuple):
        return [convert(v) for v in value]
    return convert(value)
