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

from typing import Any, Callable
from uuid import UUID

import numpy as np

from ..groups import PropertyGroup
from ..shared import Entity

key_map = {
    "values": "Data",
    "cells": "Cells",
    "surveys": "Surveys",
    "trace": "Trace",
    "trace_depth": "TraceDepth",
    "vertices": "Vertices",
    "octree_cells": "Octree Cells",
    "property_groups": "PropertyGroups",
    "u_cell_delimiters": "U cell delimiters",
    "v_cell_delimiters": "V cell delimiters",
    "z_cell_delimiters": "Z cell delimiters",
    "color_map": "Color map",
    "metadata": "Metadata",
    "options": "options",
    "concatenated_object_ids": "Concatenated object IDs",
    "attributes": "Attributes",
    "property_group_ids": "Property Group IDs",
}


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
        if value in workspace.list_entities_name:
            return workspace.get_entity(value)[0]

        # Search for property groups
        for obj in workspace.objects:
            if getattr(obj, "property_groups", None) is not None:
                prop_group = [
                    prop_group
                    for prop_group in obj.property_groups
                    if prop_group.uid == value
                ]

                if prop_group:
                    return prop_group[0]

        return None

    return value


def str2uuid(value):
    """Convert string to UUID"""
    if is_uuid(value):
        # TODO insert validation
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


def dict_mapper(
    val, string_funcs: list[Callable], *args, omit: dict | None = None
) -> dict:
    """
    Recurses through nested dictionary and applies mapping funcs to all values

    Parameters
    ----------
    val :
        Dictionary val (could be another dictionary).
    string_funcs:
        Function to apply to values within dictionary.
    omit: Dictionary of functions to omit.
    """
    if omit is None:
        omit = {}
    if isinstance(val, dict):
        for key, values in val.items():
            val[key] = dict_mapper(
                values,
                [fun for fun in string_funcs if fun not in omit.get(key, [])],
            )

    for fun in string_funcs:
        val = fun(val, *args)
    return val
