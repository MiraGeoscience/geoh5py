#  Copyright (c) 2023 Mira Geoscience Ltd.
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

import warnings
from abc import ABC
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

import h5py
import numpy as np

if TYPE_CHECKING:
    from ..workspace import Workspace
    from .entity import Entity


@contextmanager
def fetch_active_workspace(workspace: Workspace | None, mode: str = "r"):
    """
    Open a workspace in the requested 'mode'.

    If receiving an opened Workspace instead, merely return the given workspace.

    :param workspace: A Workspace class
    :param mode: Set the h5 read/write mode

    :return h5py.File: Handle to an opened Workspace.
    """
    if (
        workspace is None
        or getattr(workspace, "_geoh5")
        and mode in workspace.geoh5.mode
    ):
        try:
            yield workspace
        finally:
            pass
    else:
        if getattr(workspace, "_geoh5"):
            warnings.warn(
                f"Closing the workspace in mode '{workspace.geoh5.mode}' "
                f"and re-opening in mode '{mode}'."
            )
            workspace.close()

        try:
            yield workspace.open(mode=mode)
        finally:
            workspace.close()


@contextmanager
def fetch_h5_handle(file: str | h5py.File | Path, mode: str = "r") -> h5py.File:
    """
    Open in read+ mode a geoh5 file from string.
    If receiving a file instead of a string, merely return the given file.

    :param file: Name or handle to a geoh5 file.
    :param mode: Set the h5 read/write mode

    :return h5py.File: Handle to an opened h5py file.
    """
    if isinstance(file, h5py.File):
        try:
            yield file
        finally:
            pass
    else:
        if Path(file).suffix != ".geoh5":
            raise ValueError("Input h5 file must have a 'geoh5' extension.")

        h5file = h5py.File(file, mode)

        try:
            yield h5file
        finally:
            h5file.close()


def match_values(vec_a, vec_b, collocation_distance=1e-4) -> np.ndarray:
    """
    Find indices of matching values between two arrays, within collocation_distance.

    :param: vec_a, list or numpy.ndarray
        Input sorted values

    :param: vec_b, list or numpy.ndarray
        Query values

    :return: indices, numpy.ndarray
        Pairs of indices for matching values between the two arrays such
        that vec_a[ind[:, 0]] == vec_b[ind[:, 1]].
    """
    ind_sort = np.argsort(vec_a)
    ind = np.minimum(
        np.searchsorted(vec_a[ind_sort], vec_b, side="right"), vec_a.shape[0] - 1
    )
    nearests = np.c_[ind, ind - 1]
    match = np.where(
        np.abs(vec_a[ind_sort][nearests] - vec_b[:, None]) < collocation_distance
    )
    indices = np.c_[ind_sort[nearests[match[0], match[1]]], match[0]]
    return indices


def merge_arrays(
    head,
    tail,
    replace="A->B",
    mapping=None,
    collocation_distance=1e-4,
    return_mapping=False,
) -> np.ndarray:
    """
    Given two numpy.arrays of different length, find the matching values and append both arrays.

    :param: head, numpy.array of float
        First vector of shape(M,) to be appended.
    :param: tail, numpy.array of float
        Second vector of shape(N,) to be appended
    :param: mapping=None, numpy.ndarray of int
        Optional array where values from the head are replaced by the tail.
    :param: collocation_distance=1e-4, float
        Tolerance between matching values.

    :return: numpy.array shape(O,)
        Unique values from head to tail without repeats, within collocation_distance.
    """

    if mapping is None:
        mapping = match_values(head, tail, collocation_distance=collocation_distance)

    if mapping.shape[0] > 0:
        if replace == "B->A":
            head[mapping[:, 0]] = tail[mapping[:, 1]]
        else:
            tail[mapping[:, 1]] = head[mapping[:, 0]]

        tail = np.delete(tail, mapping[:, 1])

    if return_mapping:
        return np.r_[head, tail], mapping

    return np.r_[head, tail]


def clear_array_attributes(entity: Entity, recursive: bool = False):
    """
    Clear all stashed values of attributes from an entity to free up memory.

    :param entity: Entity to clear attributes from.
    :param recursive: Clear attributes from children entities.
    """
    if isinstance(entity.workspace.h5file, BytesIO):
        return

    for attribute in ["vertices", "cells", "values", "prisms", "layers"]:
        if hasattr(entity, attribute):
            setattr(entity, f"_{attribute}", None)

    if recursive:
        for child in entity.children:
            clear_array_attributes(child, recursive=recursive)


def compare_entities(
    object_a, object_b, ignore: list | None = None, decimal: int = 6
) -> None:
    ignore_list = ["_workspace", "_children", "_visual_parameters"]
    if ignore is not None:
        for item in ignore:
            ignore_list.append(item)

    for attr in object_a.__dict__.keys():
        if attr in ignore_list:
            continue
        if isinstance(getattr(object_a, attr[1:]), ABC):
            compare_entities(
                getattr(object_a, attr[1:]), getattr(object_b, attr[1:]), ignore=ignore
            )
        else:
            if isinstance(getattr(object_a, attr[1:]), np.ndarray):
                attr_a = getattr(object_a, attr[1:]).tolist()
                if len(attr_a) > 0 and isinstance(attr_a[0], str):
                    assert all(
                        a == b
                        for a, b in zip(
                            getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                        )
                    ), f"Error comparing attribute '{attr}'."
                else:
                    np.testing.assert_array_almost_equal(
                        attr_a,
                        getattr(object_b, attr[1:]).tolist(),
                        decimal=decimal,
                        err_msg=f"Error comparing attribute '{attr}'.",
                    )
            elif isinstance(getattr(object_a, attr[1:]), float):
                np.testing.assert_almost_equal(
                    getattr(object_a, attr[1:]),
                    getattr(object_b, attr[1:]),
                    decimal=decimal,
                    err_msg=f"Error comparing attribute '{attr}'.",
                )
            else:
                assert np.all(
                    getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                ), f"Output attribute '{attr[1:]}' for {object_a} do not match input {object_b}"


def iterable(value: Any, checklen: bool = False) -> bool:
    """
    Checks if object is iterable.

    Parameters
    ----------
    value : Object to check for iterableness.
    checklen : Restrict objects with __iter__ method to len > 1.

    Returns
    -------
    True if object has __iter__ attribute but is not string or dict type.
    """
    only_array_like = (not isinstance(value, str)) & (not isinstance(value, dict))
    if (hasattr(value, "__iter__")) & only_array_like:
        return not (checklen and (len(value) == 1))

    return False


def iterable_message(valid: list[Any] | None) -> str:
    """Append possibly iterable valid: "Must be (one of): {valid}."."""
    if valid is None:
        msg = ""
    elif iterable(valid, checklen=True):
        vstr = "'" + "', '".join(str(k) for k in valid) + "'"
        msg = f" Must be one of: {vstr}."
    else:
        msg = f" Must be: '{valid[0]}'."

    return msg


KEY_MAP = {
    "cells": "Cells",
    "color_map": "Color map",
    "concatenated_attributes": "Attributes",
    "concatenated_object_ids": "Concatenated object IDs",
    "layers": "Layers",
    "metadata": "Metadata",
    "octree_cells": "Octree Cells",
    "options": "options",
    "prisms": "Prisms",
    "property_groups": "PropertyGroups",
    "property_group_ids": "Property Group IDs",
    "surveys": "Surveys",
    "trace": "Trace",
    "trace_depth": "TraceDepth",
    "u_cell_delimiters": "U cell delimiters",
    "v_cell_delimiters": "V cell delimiters",
    "values": "Data",
    "vertices": "Vertices",
    "z_cell_delimiters": "Z cell delimiters",
    "INVALID": "Invalid",
    "INTEGER": "Integer",
    "FLOAT": "Float",
    "TEXT": "Text",
    "REFERENCED": "Referenced",
    "FILENAME": "Filename",
    "BLOB": "Blob",
    "VECTOR": "Vector",
    "DATETIME": "DateTime",
    "GEOMETRIC": "Geometric",
    "MULTI_TEXT": "Multi-Text",
    "UNKNOWN": "Unknown",
    "OBJECT": "Object",
    "CELL": "Cell",
    "VERTEX": "Vertex",
    "FACE": "Face",
    "GROUP": "Group",
    "DEPTH": "Depth",
}
INV_KEY_MAP = {value: key for key, value in KEY_MAP.items()}


def is_uuid(value: str) -> bool:
    """Check if a string is UUID compliant."""
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


def entity2uuid(value: Any) -> UUID | Any:
    """Convert an entity to its UUID."""
    if hasattr(value, "uid"):
        return value.uid
    return value


def uuid2entity(value: UUID, workspace: Workspace) -> Entity | Any:
    """Convert UUID to a known entity."""
    if isinstance(value, UUID):
        if value in workspace.list_entities_name:
            return workspace.get_entity(value)[0]

        # Search for property groups
        for obj in workspace.objects:
            if getattr(obj, "property_groups", None) is not None:
                prop_group = [
                    prop_group
                    for prop_group in getattr(obj, "property_groups")
                    if prop_group.uid == value
                ]

                if prop_group:
                    return prop_group[0]

        return None

    return value


def str2uuid(value: Any) -> UUID | Any:
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


def as_str_if_utf8_bytes(value) -> str:
    """Convert bytes to string"""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def dict_mapper(val, string_funcs: list[Callable], *args, omit: dict | None = None):
    """
    Recursion through nested dictionaries and applies mapping functions to values.

    :param val: Value (could be another dictionary) to apply transform functions.
    :param string_funcs: Functions to apply on values within the input dictionary.
    :param omit: Dictionary of functions to omit.

    :return val: Transformed values
    """
    if isinstance(val, dict):
        for key, values in val.items():
            short_list = string_funcs.copy()
            if omit is not None:
                short_list = [
                    fun for fun in string_funcs if fun not in omit.get(key, [])
                ]

            val[key] = dict_mapper(values, short_list)

    if isinstance(val, list):
        out = []
        for elem in val:
            for fun in string_funcs:
                elem = fun(elem, *args)
            out += [elem]
        return out

    for fun in string_funcs:
        val = fun(val, *args)
    return val


def box_intersect(extent_a: np.ndarray, extent_b: np.ndarray) -> bool:
    """
    Compute the intersection of two axis-aligned bounding extents defined by their
    arrays of minimum and maximum bounds in N-D space.

    :param extent_a: First extent or shape (2, N)
    :param extent_b: Second extent or shape (2, N)

    :return: Logic if the box extents intersect along all dimensions.
    """
    for extent in [extent_a, extent_b]:
        if not isinstance(extent, np.ndarray) or extent.ndim != 2:
            raise TypeError("Input extents must be 2D numpy.ndarrays.")

        if extent.shape[0] != 2 or not np.all(extent[0, :] <= extent[1, :]):
            raise ValueError(
                "Extents must be of shape (2, N) containing the minimum and maximum "
                "bounds in nd-space on the first and second row respectively."
            )

    for comp_a, comp_b in zip(extent_a.T, extent_b.T):
        min_ext = max(comp_a[0], comp_b[0])
        max_ext = min(comp_a[1], comp_b[1])

        if min_ext > max_ext:
            return False

    return True


def mask_by_extent(
    locations: np.ndarray, extent: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """
    Find indices of locations within a rectangular extent.

    :param locations: shape(*, 3) or shape(*, 2) Coordinates to be evaluated.
    :param extent: shape(2, 2) Limits defined by the South-West and
        North-East corners. Extents can also be provided as 3D coordinates
        with shape(2, 3) defining the top and bottom limits.
    :param inverse: Return the complement of the mask extent.

    :returns: Array of bool for the locations inside or outside the box extent.
    """
    if not isinstance(extent, np.ndarray) or extent.ndim != 2:
        raise ValueError("Input 'extent' must be a 2D array-like.")

    if not isinstance(locations, np.ndarray) or locations.ndim != 2:
        raise ValueError(
            "Input 'locations' must be an array-like of shape(*, 3) or (*, 2)."
        )

    indices = np.ones(locations.shape[0], dtype=bool)
    for loc, lim in zip(locations.T, extent.T):
        indices &= (lim[0] <= loc) & (loc <= lim[1])

    if inverse:
        return ~indices

    return indices


def get_attributes(entity, omit_list=(), attributes=None):
    """Extract the attributes of an object with omissions."""
    if attributes is None:
        attributes = {}
    for key in vars(entity):
        if key not in omit_list:
            if key[0] == "_":
                key = key[1:]

            attr = getattr(entity, key)
            attributes[key] = attr

    return attributes


def xy_rotation_matrix(angle: float) -> np.ndarray:
    """
    Rotation matrix about the z-axis.

    :param angle: Rotation angle in radians.

    :return rot: Rotation matrix.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def yz_rotation_matrix(angle: float) -> np.ndarray:
    """
    Rotation matrix about the x-axis.
    :param angle: Rotation angle in radians.
    :return: rot: Rotation matrix.
    """

    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def dip_points(points: np.ndarray, dip: float, rotation: float = 0) -> np.ndarray:
    """
    Rotate points about the x-axis by the dip angle and then about the z-axis by the rotation angle.
    :param points: an array of points to rotate
    :param dip: the dip angle in radians
    :param rotation: the rotation angle in radians
    :return: the rotated points
    """
    # Assert points is a numpy array containing 3D points
    if not isinstance(points, np.ndarray) and points.ndim != 2 and points.shape[1] != 3:
        raise TypeError("Input points must be a 2D numpy array of shape (N, 3).")

    # rotate the points about the z-axis by the inverse rotation angle
    points = xy_rotation_matrix(-rotation) @ points.T

    # Rotate points with the dip angle
    points = yz_rotation_matrix(dip) @ points

    # Rotate back the points to initial orientation
    points = xy_rotation_matrix(rotation) @ points

    return points.T
