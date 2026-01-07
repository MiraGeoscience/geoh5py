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

from uuid import UUID


def uuid_to_string(value: UUID | list[UUID] | None) -> str | list[str]:
    def convert(value: UUID | None) -> str:
        if value is None:
            return ""
        if isinstance(value, UUID):
            return f"{{{value!s}}}"
        return value

    if isinstance(value, list):
        return [convert(v) for v in value]
    return convert(value)


def empty_string_to_none(value):
    """Promote empty string to uid, and pass all other values."""
    if value == "":
        return None
    return value


UidOrNumeric = UUID | float | int | None
StringOrNumeric = str | float | int


def uuid_to_string_or_numeric(
    value: UidOrNumeric | list[UidOrNumeric],
) -> StringOrNumeric | list[StringOrNumeric]:
    def convert(value: UidOrNumeric) -> StringOrNumeric:
        if value is None:
            return ""
        if isinstance(value, UUID):
            return f"{{{value}}}"
        return value

    if isinstance(value, list):
        return [convert(v) for v in value]
    return convert(value)
