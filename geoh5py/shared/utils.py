#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from abc import ABC
from collections.abc import Callable
from contextlib import contextmanager
from io import BytesIO
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID
from warnings import warn

import h5py
import numpy as np

from .exceptions import Geoh5FileClosedError


if TYPE_CHECKING:
    from ..workspace import Workspace
    from .entity import Entity

INV_KEY_MAP = {
    "Allow delete": "allow_delete",
    "Allow delete contents": "allow_delete_content",
    "Allow move": "allow_move",
    "Allow move contents": "allow_move_content",
    "Allow rename": "allow_rename",
    "Association": "association",
    "Attributes": "concatenated_attributes",
    "Blob": "BLOB",
    "Boolean": "BOOLEAN",
    "Cell": "CELL",
    "Cells": "cells",
    "Clipping IDs": "clipping_ids: list | None",
    "Collar": "collar",
    "Color map": "color_map",
    "Contributors": "contributors",
    "Concatenated object IDs": "concatenated_object_ids",
    "Cost": "cost",
    "Current line property ID": "current_line_id",
    "Data": "values",
    "DateTime": "DATETIME",
    "Description": "description",
    "Dip": "dip",
    "Distance unit": "distance_unit",
    "Dynamic implementation ID": "dynamic_implementation_id",
    "Duplicate type on copy": "duplicate_type_on_copy",
    "End of hole": "end_of_hole",
    "Face": "FACE",
    "File name": "name",
    "Filename": "FILENAME",
    "Float": "FLOAT",
    "Geometric": "GEOMETRIC",
    "Group": "GROUP",
    "Group Name": "name",
    "GA Version": "ga_version",
    "Hidden": "hidden",
    "Invalid": "INVALID",
    "Integer": "INTEGER",
    "ID": "uid",
    "Last focus": "last_focus",
    "Layers": "layers",
    "Mapping": "mapping",
    "Metadata": "metadata",
    "Modifiable": "modifiable",
    "Multi-Text": "MULTI_TEXT",
    "Name": "name",
    "Number of bins": "number_of_bins",
    "NU": "u_count",
    "NV": "v_count",
    "NW": "w_count",
    "options": "options",
    "Object": "OBJECT",
    "Origin": "origin",
    "Octree Cells": "octree_cells",
    "Partially hidden": "partially_hidden",
    "Planning": "planning",
    "Precision": "precision",
    "Primitive type": "primitive_type",
    "Prisms": "prisms",
    "Properties": "properties",
    "Property Group IDs": "property_group_ids",
    "Property Group Type": "property_group_type",
    "PropertyGroups": "property_groups",
    "Public": "public",
    "Referenced": "REFERENCED",
    "Rotation": "rotation",
    "Scale": "scale",
    "Scientific notation": "scientific_notation",
    "Surveys": "surveys",
    "Text": "TEXT",
    "Trace": "trace",
    "TraceDepth": "trace_depth",
    "Transparent no data": "transparent_no_data",
    "Unknown": "UNKNOWN",
    "U cell delimiters": "u_cell_delimiters",
    "V cell delimiters": "v_cell_delimiters",
    "Z cell delimiters": "z_cell_delimiters",
    "U Cell Size": "u_cell_size",
    "U Count": "u_count",
    "U Size": "u_cell_size",
    "V Cell Size": "v_cell_size",
    "V Count": "v_count",
    "V Size": "v_cell_size",
    "Vector": "VECTOR",
    "Version": "version",
    "Vertical": "vertical",
    "Vertices": "vertices",
    "Vertex": "VERTEX",
    "Visible": "visible",
    "W Cell Size": "w_cell_size",
}

KEY_MAP = {value: key for key, value in INV_KEY_MAP.items()}

PNG_KWARGS = {"format": "PNG", "compress_level": 9}
JPG_KWARGS = {"format": "JPEG", "quality": 85}
TIF_KWARGS = {"format": "TIFF"}

PILLOW_ARGUMENTS = {
    "1": PNG_KWARGS,
    "L": PNG_KWARGS,
    "P": PNG_KWARGS,
    "RGB": PNG_KWARGS,
    "RGBA": PNG_KWARGS,
    "CMYK": JPG_KWARGS,
    "YCbCr": JPG_KWARGS,
    "I": TIF_KWARGS,
    "F": TIF_KWARGS,
}


@contextmanager
def fetch_active_workspace(workspace: Workspace | None, mode: str = "r"):
    """
    Open a workspace in the requested 'mode'.

    If receiving an opened Workspace instead, merely return the given workspace.

    :param workspace: A Workspace class
    :param mode: Set the h5 read/write mode

    :return h5py.File: Handle to an opened Workspace.
    """
    try:
        geoh5 = None if workspace is None else workspace.geoh5
    except Geoh5FileClosedError:
        geoh5 = None

    if workspace is None or (geoh5 is not None and mode in workspace.geoh5.mode):
        try:
            yield workspace
        finally:
            pass
    else:
        if geoh5 is not None:
            warn(
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
    *,
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

    for attribute in [
        "vertices",
        "cells",
        "values",
        "prisms",
        "layers",
        "octree_cells",
    ]:
        if hasattr(entity, attribute):
            setattr(entity, f"_{attribute}", None)

    if recursive and hasattr(entity, "children"):
        for child in entity.children:
            clear_array_attributes(child, recursive=recursive)


def are_objects_similar(obj1, obj2, ignore: list[str] | None):
    """
    Compare two objects to see if they are similar. This is a shallow comparison.

    :param obj1: The first object.
    :param obj2: The first object.
    :param ignore: List of attributes to ignore.

    :return: If attributes similar or not.
    """
    assert isinstance(obj1, type(obj2)), "Objects are not the same type."

    attributes1 = getattr(obj1, "__dict__", obj1)
    attributes2 = getattr(obj2, "__dict__", obj2)

    # remove the ignore attributes
    if isinstance(ignore, list) and isinstance(attributes1, dict):
        for item in ignore:
            attributes1.pop(item, None)
            attributes2.pop(item, None)

    return attributes1 == attributes2


def compare_arrays(object_a, object_b, attribute: str, decimal: int = 6):
    """
    Utility to compare array properties from two Entities

    :param object_a: First Entity
    :param object_b: Second Entity
    :param attribute: Attribute to compare
    :param decimal: Decimal precision for comparison
    """
    array_a_values = getattr(object_a, attribute)
    array_b_values = getattr(object_b, attribute)

    if array_b_values is None:
        raise ValueError(f"attr {attribute} is None for object {object_b.name}")

    if array_b_values.dtype.names is not None:
        assert all(
            np.all(array_a_values[name] == array_b_values[name])
            for name in array_b_values.dtype.names
        ), f"Error comparing attribute '{attribute}'."

    elif len(array_a_values) > 0 and isinstance(array_a_values[0], str):
        assert all(
            array_a_values == array_b_values
        ), f"Error comparing attribute '{attribute}'."
    else:
        np.testing.assert_array_almost_equal(
            array_a_values,
            array_b_values,
            decimal=decimal,
            err_msg=f"Error comparing attribute '{attribute}'.",
        )


def compare_floats(object_a, object_b, attribute: str, decimal: int = 6):
    np.testing.assert_almost_equal(
        getattr(object_a, attribute),
        getattr(object_b, attribute),
        decimal=decimal,
        err_msg=f"Error comparing attribute '{attribute}'.",
    )


def compare_list(object_a, object_b, attribute: str, ignore: list[str] | None):
    get_object_a = getattr(object_a, attribute)
    get_object_b = getattr(object_b, attribute)
    assert isinstance(get_object_a, list)
    assert len(get_object_a) == len(get_object_b)
    for obj_a, obj_b in zip(get_object_a, get_object_b, strict=False):
        assert are_objects_similar(obj_a, obj_b, ignore)


def compare_bytes(object_a, object_b):
    assert (
        object_a == object_b
    ), f"{type(object_a)} objects: {object_a}, {object_b} are not equal."


def compare_entities(
    object_a, object_b, ignore: list[str] | None = None, decimal: int = 6
) -> None:
    if isinstance(object_a, bytes):
        compare_bytes(object_a, object_b)
        return

    base_ignore = ["_workspace", "_children", "_visual_parameters", "_entity_class"]
    ignore_list = base_ignore + ignore if ignore else base_ignore

    for attr in [k for k in object_a.__dict__ if k not in ignore_list]:
        if isinstance(getattr(object_a, attr.lstrip("_")), ABC):
            compare_entities(
                getattr(object_a, attr.lstrip("_")),
                getattr(object_b, attr.lstrip("_")),
                ignore=ignore,
                decimal=decimal,
            )
        else:
            if isinstance(getattr(object_a, attr.lstrip("_")), np.ndarray):
                compare_arrays(object_a, object_b, attr.lstrip("_"), decimal=decimal)
            elif isinstance(getattr(object_a, attr.lstrip("_")), float):
                compare_floats(object_a, object_b, attr.lstrip("_"), decimal=decimal)
            elif isinstance(getattr(object_a, attr.lstrip("_")), list):
                compare_list(object_a, object_b, attr.lstrip("_"), ignore)
            else:
                try:
                    assert np.all(
                        getattr(object_a, attr.lstrip("_"))
                        == getattr(object_b, attr.lstrip("_"))
                    ), (
                        f"Output attribute '{attr.lstrip('_')}' for {object_a} do "
                        f"not match input {object_b}"
                    )
                except AssertionError:
                    pass


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
            if obj.property_groups is not None:
                prop_group = [
                    prop_group
                    for prop_group in obj.property_groups
                    if prop_group.uid == value
                ]

                if prop_group:
                    return prop_group[0]

        return None

    return value


def str2uuid(value: Any) -> UUID | Any:
    """Convert string to UUID"""
    if isinstance(value, bytes):
        value = value.decode("utf-8")

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


def as_float_if_isnumeric(value: str) -> float | str:
    """Convert bytes to string"""
    if value.isnumeric():
        return float(value)
    return value


def str_json_to_dict(string: str | bytes) -> dict:
    """
    Convert a json string or bytes to a dictionary.

    :param string: The json string or bytes to convert to a dictionary.

    :return: The dictionary representation of the json string with uuid promoted.
    """
    value = as_str_if_utf8_bytes(string)
    json_dict = loads(value)

    for key, val in json_dict.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                json_dict[key][sub_key] = str2uuid(sub_val)
        else:
            json_dict[key] = str2uuid(val)

    return json_dict


def ensure_uuid(value: UUID | str) -> UUID:
    """
    Ensure that the value is a UUID.

    If not, it raises a type error.

    :param value: The value to ensure is a UUID.

    :return: The verified UUID.
    """
    value = str2uuid(value)

    if not isinstance(value, UUID):
        raise TypeError(f"Value {value} is not a UUID but a {type(value)}.")

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

    for comp_a, comp_b in zip(extent_a.T, extent_b.T, strict=False):
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
    for loc, lim in zip(locations.T, extent.T, strict=False):
        indices &= (lim[0] <= loc) & (loc <= lim[1])

    if inverse:
        return ~indices

    return indices


def get_attributes(entity, omit_list=(), attributes=None) -> dict:
    """Extract the attributes of an object with omissions."""
    if attributes is None:
        attributes = {}
    for key in vars(entity):
        if key not in omit_list:
            key = key.lstrip("_")
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


def set_attributes(entity, **kwargs):
    """
    Loop over kwargs and set attributes to an entity.

    TODO: Deprecate in favor of explicit attribute setting.
    """
    for key, value in kwargs.items():
        try:
            setattr(entity, key, value)
        except AttributeError:
            continue


def map_name_attributes(object_, **kwargs: dict) -> dict:
    """
    Map attributes to an object. The object must have an '_attribute_map'.

    :param object_: The object to map the attributes to.
    :param kwargs: Dictionary of attributes.
    """
    mapping = getattr(object_, "_attribute_map", None)

    if mapping is None:
        raise AttributeError("Object must have an '_attribute_map' attribute.")

    new_args = {}
    for attr, item in kwargs.items():
        if attr in mapping:
            new_args[mapping[attr]] = item
        else:
            new_args[attr] = item

    return new_args


def map_attributes(object_, **kwargs):
    """
    Map attributes to an object. The object must have an '_attribute_map'.

    :param entity: The object to map the attributes to.
    :param kwargs: The kwargs to map to the object.
    """

    values = map_name_attributes(object_, **kwargs)  # Swap duplicates
    set_attributes(object_, **values)


def stringify(values: dict[str, Any]) -> dict[str, Any]:
    """
    Convert all values in a dictionary to string.

    :param values: Dictionary of values to be converted.
    """
    string_dict = {}
    for key, value in values.items():
        mappers = [nan2str, inf2str, as_str_if_uuid, none2str]
        string_dict[key] = dict_mapper(value, mappers)

    return string_dict


def to_list(value: Any) -> list:
    """
    Convert value to a list.

    :param value: The value to convert.

    :return: A list
    """
    # ensure the names are a list
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def to_tuple(value: Any) -> tuple:
    """
    Convert value to a tuple.

    :param value: The value to convert.

    :return: A tuple
    """
    # ensure the names are a tuple
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


class SetDict(dict):
    def __init__(self, **kwargs):
        kwargs = {k: self.make_set(v) for k, v in kwargs.items()}
        super().__init__(kwargs)

    def make_set(self, value):
        if isinstance(value, (set, tuple, list)):
            value = set(value)
        else:
            value = {value}
        return value

    def __setitem__(self, key, value):
        value = self.make_set(value)
        super().__setitem__(key, value)

    def update(self, value: dict, **kwargs) -> None:  # type: ignore
        for key, val in value.items():
            val = self.make_set(val)
            if key in self:
                val = self[key].union(val)
            value[key] = val
        super().update(value, **kwargs)


def inf2str(value):  # map np.inf to "inf"
    if not isinstance(value, (int, float)):
        return value
    return str(value) if not np.isfinite(value) else value


def list2str(value):
    if isinstance(value, list):  # & (key not in exclude):
        return str(value)[1:-1]
    return value


def none2str(value):
    if value is None:
        return ""
    return value


def nan2str(value):
    if value is np.nan:
        return ""
    return value


def str2none(value):
    if value == "":
        return None
    return value


def find_unique_name(name: str, names: list[str]) -> str:
    """
    Get a unique name not in a list of names.

    :param name: The name to check.
    :param names: The list of names to avoid.

    :return: a unique name.
    """

    if name not in names:
        return name

    count = 1
    while f"{name}({count})" in names:
        count += 1

    return f"{name}({count})"


def remove_duplicates_in_list(input_list: list) -> list:
    """
    Remove duplicates from a list without changing the sorting.

    :param input_list: the list to remove duplicates from.

    :return: The sorted list
    """
    return sorted(set(input_list), key=input_list.index)
