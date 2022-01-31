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

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from ..groups import PropertyGroup
from ..shared import Entity


def is_uuid(value: str):
    """Check if a string is UUID compliant."""
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


def entity2uuid(value):
    """Convert an entity to its UUID."""
    if isinstance(value, (Entity, PropertyGroup)):
        return value.uid
    return value


def uuid2entity(value, workspace):
    """Convert UUID to a known entity."""
    if isinstance(value, UUID):
        entity = [
            child
            for child in workspace.fetch_children(workspace.root, recursively=True)
            if child.uid == value
        ]
        return entity[0] if entity else None
    return value


def str2uuid(value):
    """Convert string to UUID"""
    if is_uuid(value):
        return UUID(str(value))
    return value


def as_str_if_uuid(value: UUID | Any) -> str | Any:
    """Convert :obj:`UUID` to string used in geoh5."""
    if isinstance(value, UUID):
        return "{" + str(value) + "}"
    return value


def bool_value(value: np.int8) -> bool:
    """Convert logical int8 to bool."""
    return bool(value)


def str_from_utf8_bytes(value: bytes | str) -> str:
    """Convert bytes to string"""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value
