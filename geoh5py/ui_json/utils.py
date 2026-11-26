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

from io import BytesIO
from logging import getLogger
from pathlib import Path
from shutil import copy, move
from time import time
from typing import Any
from uuid import UUID

from geoh5py import Workspace, groups, objects
from geoh5py.shared.entity_container import EntityContainer
from geoh5py.shared.utils import (
    ClassIdentifierEnum,
    dict_mapper,
    entity2uuid,
    equalize_string,
    fetch_active_workspace,
    is_uuid,
    map_to_class,
    str2none,
)
from geoh5py.workspace import TYPE_UID_TO_CLASS


logger = getLogger(__name__)


GA_NAME_TO_OBJECT: dict[str, type[objects.ObjectBase]] = {
    equalize_string(k): v
    for k, v in map_to_class(ClassIdentifierEnum.DEFAULT_NAME, [objects]).items()
}
GA_NAME_TO_GROUP: dict[str, type[groups.Group]] = {
    equalize_string(k): v
    for k, v in map_to_class(ClassIdentifierEnum.DEFAULT_NAME, [groups]).items()
}


def class_or_raise(value: UUID) -> type[objects.ObjectBase] | type[groups.Group]:
    """Promote uid to class, raise if uid is not a geoh5py type uid."""
    if value not in TYPE_UID_TO_CLASS:
        raise ValueError(
            f"Provided type_uid string {value!s} is not a recognized "
            f"geoh5py object or group type uid."
        )
    return TYPE_UID_TO_CLASS[value]


def monitored_directory_copy(
    directory: str,
    entities: EntityContainer | list[EntityContainer],
    copy_children: bool = True,
) -> str:
    """
    Create a temporary geoh5 file in the monitoring folder and export entities for update.

    Creates a temporary workspace in the specified monitoring directory and copies
    the given entities to it. This is useful for monitoring and updating entities
    in a separate workspace environment.

    :param directory: Path to the monitoring directory where the temporary file will be created.
    :param entities: Entity (objects.ObjectBase or groups.Group) to be copied for monitoring.
    :param copy_children: Whether to copy children entities along with the main entities.

    :return: Full path to the created temporary geoh5 file.
    """
    directory_path = Path(directory)
    working_path = directory_path / ".working"
    working_path.mkdir(exist_ok=True)

    temp_geoh5 = f"temp{time():.3f}.geoh5"

    if not isinstance(entities, list | tuple):
        entities = [entities]

    with fetch_active_workspace(entities[0].workspace, mode="r"):
        with Workspace.create(working_path / temp_geoh5) as w_s:
            for entity in entities:
                entity.copy(parent=w_s, copy_children=copy_children)

    move(working_path / temp_geoh5, directory_path / temp_geoh5, copy)

    return str(directory_path / temp_geoh5)


def name_or_uid_to_type(
    value: str | UUID | type,
) -> type[objects.ObjectBase] | type[groups.Group] | UUID:
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
    obj: type[objects.ObjectBase] | type[groups.Group] | None = GA_NAME_TO_OBJECT.get(
        value, None
    ) or GA_NAME_TO_GROUP.get(value, None)
    if obj is None:
        raise ValueError(
            f"Provided string {value!s} is not a recognized "
            f"geoh5py object or group type."
        ) from None

    return obj


def optional_uuid_mapper(value: Any):
    """
    Take values and convert them into UUID or None (or list of).

    :param value: Either a string of entity, or list of.
    :return: UUID, Nont or list of.
    """
    return dict_mapper(value, [str2none, entity2uuid])


def str2list(value: Any) -> Any:
    """
    Convert string representation of numbers to list of floats.

    Maps string values like "[1,2,3]" or "1,2,3" to [1.0, 2.0, 3.0].
    Handles special cases like "inf", "-inf", and empty strings.

    :param value: Value to convert, typically a string or already converted type.

    :return: List of floats if conversion successful, original value otherwise.
    """
    if isinstance(value, str):
        if value in ["inf", "-inf", ""]:
            return value
        try:
            return [float(n) for n in value.split(",") if n != ""]
        except ValueError:
            return value

    return value


def to_class(
    value: UUID | type[objects.ObjectBase] | type[groups.Group],
) -> type[objects.ObjectBase] | type[groups.Group]:
    """
    Promote uid to class.

    Passes existing classes and raises if uid is not a geoh5py type uid.
    """
    if isinstance(value, UUID):
        value = class_or_raise(value)

    if not isinstance(value, type) or not issubclass(
        value, (objects.ObjectBase, groups.Group)
    ):
        raise ValueError(
            f"Provided type_uid string {value!s} is not a recognized "
            f"geoh5py object or group type uid."
        )

    return value


def to_class_type(
    values: str | UUID | list[str] | list[UUID],
) -> (
    type[objects.ObjectBase]
    | type[groups.Group]
    | list[type[objects.ObjectBase]]
    | list[type[groups.Group]]
):
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


def to_list(value: Any) -> list[Any]:
    """Promote single values to list."""
    if isinstance(value, str) and ";" in value:
        value = value.split(";")
    if not isinstance(value, list):
        value = [value]
    return value


def to_path(value: list[str]) -> list[Path | None]:
    """Promote path strings to patlib.Path objects."""
    out: list[Path | None] = []
    for path in value:
        if path is None or path == "":
            out.append(None)
        else:
            out.append(Path(path))

    return out


def workspace2path(value: Any) -> str:
    """
    Convert a Workspace object to its file path string.

    **Deprecated:** This function has been migrated to `geoh5py.shared.utils.workspace2path`
    and will be removed in future versions.

    :param value: Workspace object or other value to convert.

    :return: File path string for Workspace objects, "[in-memory]" for in-memory workspaces,
             or the original value for non-Workspace inputs.
    """
    logger.warning(
        "Deprecation Warning - This function has been migrated to "
        "`geoh5py.shared.utils.workspace2path` and will be removed in"
        "future versions.",
    )
    if isinstance(value, Workspace):
        if isinstance(value.h5file, BytesIO):
            return "[in-memory]"
        return str(value.h5file)
    return value
