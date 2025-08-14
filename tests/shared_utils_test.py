# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import numpy as np
import pytest

from geoh5py.shared.exceptions import iterable, iterable_message
from geoh5py.shared.utils import (
    box_intersect,
    inf2str,
    mask_by_extent,
    nan2str,
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
    with pytest.raises(
        ValueError, match=re.escape("Input 'extent' must be a 2D array-like.")
    ):
        mask_by_extent(points, 1.0)

    with pytest.raises(
        ValueError,
        match=re.escape("Input 'extent' must be a 2D array-like"),
    ):
        mask_by_extent("abc", corners)

    assert not mask_by_extent(np.vstack([points]), np.vstack(corners[:2])), (
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
