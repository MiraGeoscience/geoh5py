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


# pylint: disable=too-few-public-methods

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from geoh5py import groups, objects
from geoh5py.groups import Group
from geoh5py.objects import ObjectBase

from geoh5py.shared.utils import (
    ClassIdentifierEnum,
    dict_mapper,
    equalize_string,
    is_uuid,
    map_to_class,
)
from geoh5py.workspace import TYPE_UID_TO_CLASS


GA_NAME_TO_OBJECT: dict[str, type[ObjectBase]] = {
    equalize_string(k): v
    for k, v in map_to_class(ClassIdentifierEnum.DEFAULT_NAME, [objects]).items()
}
GA_NAME_TO_GROUP: dict[str, type[Group]] = {
    equalize_string(k): v
    for k, v in map_to_class(ClassIdentifierEnum.DEFAULT_NAME, [groups]).items()
}


def to_path(value: list[str]) -> list[Path]:
    """Promote path strings to patlib.Path objects."""
    out = []
    for path in value:
        if isinstance(path, str):
            out.append(Path(path))
        else:
            out.append(path)
    return out


def to_list(value: Any) -> list[Any]:
    """Promote single values to list."""
    if isinstance(value, str) and ";" in value:
        value = value.split(";")
    if not isinstance(value, list):
        value = [value]
    return value


def name_or_uid_to_type(
    value: str | UUID | type,
) -> type[ObjectBase] | type[Group] | UUID:
    """
    Convert a string to a geoh5py type or group type.

    :param value: String representing geoh5py type, either as a UUID or name.
    :return: Type of object, group or UUID
    """
    if not isinstance(value, str):
        return value

    if is_uuid(value):
        return UUID(value)

    value = equalize_string(value)
    obj: type[ObjectBase] | type[Group] | None = GA_NAME_TO_OBJECT.get(
        value, None
    ) or GA_NAME_TO_GROUP.get(value, None)
    if obj is None:
        raise ValueError(
            f"Provided string {value!s} is not a recognized "
            f"geoh5py object or group type."
        ) from None

    return obj


def to_class_type(
    values: str | UUID | list[str] | list[UUID],
) -> type[ObjectBase] | type[Group] | list[type[ObjectBase]] | list[type[Group]]:
    """
    Promote strings or uuid to a class type.

    Strings can represent both uid(s) or names that represent geoh5py objects.
    We first attempt to convert strings to uid(s) and then fall back on conversion
    from name to class(es), and finally to type uid(s).  GA naming doesn't match
    geoh5py naming, so we must map names to classes before type uid conversion.

    :param values: Value or list of strings representing either geoh5py type uids or class names.
    :return: Value or list of geoh5py objects or groups.
    """
    return dict_mapper(values, [name_or_uid_to_type, to_class])


def class_or_raise(value: UUID) -> type[ObjectBase] | type[Group]:
    """Promote uid to class, raise if uid is not a geoh5py type uid."""
    if value not in TYPE_UID_TO_CLASS:
        raise ValueError(
            f"Provided type_uid string {value!s} is not a recognized "
            f"geoh5py object or group type uid."
        )
    return TYPE_UID_TO_CLASS[value]


def to_class(
    value: UUID | type[ObjectBase] | type[Group],
) -> type[ObjectBase] | type[Group]:
    """
    Promote uid to class.

    Passes existing classes and raises if uid is not a geoh5py type uid.
    """
    if isinstance(value, UUID):
        value = class_or_raise(value)

    if not isinstance(value, type) or not issubclass(value, (ObjectBase, Group)):
        raise ValueError(
            f"Provided type_uid string {value!s} is not a recognized "
            f"geoh5py object or group type uid."
        )

    return value
