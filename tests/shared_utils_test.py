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

import re
import uuid

import numpy as np
import pytest
from pydantic import BaseModel

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.shared.exceptions import iterable, iterable_message
from geoh5py.shared.utils import (
    all_subclasses,
    as_str_if_uuid,
    best_match_subclass,
    box_intersect,
    dict_to_json_str,
    inf2str,
    mask_by_extent,
    model_fields_difference,
    nan2str,
    no_required_indicators,
    uuid_from_values,
)


def test_inf2str():
    assert inf2str(np.inf) == "inf"
    assert inf2str(1) == 1
    assert inf2str("a") == "a"


def test_nan2str():
    assert nan2str(np.nan) == ""
    assert nan2str(1) == 1
    assert nan2str("a") == "a"


def test_iterable():
    assert iterable([1, 2, 4])
    assert not iterable(2)
    assert not iterable({"a": 1, "b": 2})
    assert not iterable("lskdjfs")
    assert not iterable([1], checklen=True)


def test_iterable_message():
    assert iterable_message(None) == ""
    assert "Must be one of:" in iterable_message([1, 2, 3])
    assert "Must be:" in iterable_message([1])


def test_mask_by_extent():
    corners = [[-1, -2], [4, 5], [2, 3]]
    points = [-100, 100, 0]
    with pytest.raises(ValueError, match="Input 'extent' must be a 2D array-like."):
        mask_by_extent(np.vstack([points]), 1.0)

    with pytest.raises(
        ValueError,
        match=re.escape("Input 'locations' must be an array-like of"),
    ):
        mask_by_extent("abc", corners)

    assert not mask_by_extent(np.vstack([points]), corners[:2]), (
        "Point should have been outside."
    )


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        (np.vstack([[-2, -2, -2], [2, 2, 2]]), np.vstack([[0, 0, 0], [4, 4, 4]])),
        (np.vstack([[-2, -2], [2, 2]]), np.vstack([[0, 0], [4, 4]])),
    ],
)
def test_box_intersect_corner(box_a, box_b):
    assert box_intersect(box_a, box_b)


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        (np.vstack([[-2, -2, -2], [2, 2, 2]]), np.vstack([[-1, -1, 0], [1, 1, 4]])),
        (np.vstack([[-2, -2], [2, 2]]), np.vstack([[-1, -1], [1, 1]])),
    ],
)
def test_box_intersect_face(box_a, box_b):
    assert box_intersect(box_a, box_b)


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        (np.vstack([[-2, -2, -2], [2, 2, 2]]), np.vstack([[-1, -1, -1], [1, 1, 1]])),
        (np.vstack([[-2, -2], [2, 2]]), np.vstack([[-1, -1], [1, 1]])),
    ],
)
def test_box_intersect_inside(box_a, box_b):
    assert box_intersect(box_a, box_b)


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        (np.vstack([[-2, -2, -2], [2, 2, 2]]), np.vstack([[1, 1, 4], [2, 2, 5]])),
        (np.vstack([[-2, -2], [2, 2]]), np.vstack([[1, 4], [2, 5]])),
    ],
)
def test_box_intersect_disjoint(box_a, box_b):
    assert not box_intersect(box_a, box_b)


def test_box_intersect_input():
    box_a = np.vstack([[-2, -2, -2], [2, 2, 2]])
    # One corner inside
    box_b = np.vstack([[0, 0, 0], [4, 4, 4]])

    with pytest.raises(
        TypeError,
        match=re.escape("Input extents must be 2D numpy.ndarrays."),
    ):
        box_intersect(np.r_[1], box_b)

    with pytest.raises(
        ValueError,
        match=re.escape("Extents must be of shape (2, N) containing the minimum"),
    ):
        box_intersect(box_a, box_b[::-1, :])


def test_dict_to_str():
    workspace = Workspace()
    pts = Points.create(workspace)
    uid = uuid.uuid4()
    data = {
        "key1": "value1",
        "key2": 123,
        "key3": 45.67,
        "key4": uid,
        "key5": workspace,
        "key6": pts,
    }
    str_dict = dict_to_json_str(data)

    assert all(
        (elem in str_dict)
        for elem in ["4.5670e+01", str(uid), '"key2": 123', str(pts.uid)]
    )


def test_uuid_from_values():
    uid = uuid.uuid4()
    data = {
        "key1": "value1",
        "key2": 123,
        "key3": 45.67,
        "key4": uid,
    }
    uid_a = uuid_from_values(data)

    data["key4"] = as_str_if_uuid(uid)
    assert uid_a == uuid_from_values(data)

    data["key5"] = "abc"

    assert uid_a != uuid_from_values(data)


def test_all_subclasses():
    class TestOne:
        pass

    class TestTwo(TestOne):
        pass

    class TestThree(TestTwo):
        pass

    class TestFour(TestOne):
        pass

    assert len(all_subclasses(TestOne)) == 3
    assert all(k in all_subclasses(TestOne) for k in [TestTwo, TestThree, TestFour])
    assert len(all_subclasses(TestTwo)) == 1
    assert all_subclasses(TestTwo)[0] == TestThree
    assert all_subclasses(TestThree) == []
    assert all_subclasses(TestFour) == []


def test_model_fields_difference():
    class MyParent(BaseModel):
        a: str

    class MyChild(MyParent):
        b: int
        c: float

    class MyOtherChild(MyParent):
        d: float
        e: float = 1.0

    diff = model_fields_difference(MyParent, MyChild)
    assert diff == {"b", "c"}


def test_best_match_subclass():
    class Parent(BaseModel):
        a: str
        b: int

    class ChildA(Parent):
        c: float

    class ChildB(ChildA):
        d: float

    class ChildC(Parent):
        e: float

    best_match = best_match_subclass(
        parent=Parent,
        children=[ChildA, ChildB, ChildC],
        data={"a": "test", "b": 1, "c": 1.0, "d": 1.0},
    )

    assert best_match == ChildB

    best_match = best_match_subclass(
        parent=Parent,
        children=[ChildA, ChildB, ChildC],
        data={"a": "test", "b": 1, "c": 1.0},
    )

    assert best_match == ChildA


def test_no_required_indicators():
    class Parent(BaseModel):
        a: str
        b: int

    class ChildA(Parent):
        c: float = 1.0

    class ChildB(Parent):
        d: float

    candidates = no_required_indicators(Parent, [ChildA, ChildB])
    assert candidates == [ChildA]
