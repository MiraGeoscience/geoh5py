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

import re

import numpy as np
import pytest

from geoh5py.shared.utils import (
    box_intersect,
    iterable,
    iterable_message,
    mask_by_extent,
)


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

    assert not mask_by_extent(
        np.vstack([points]), np.vstack(corners[:2])
    ), "Point should have been outside."


def test_box_intersect():
    box_a = np.vstack([[-2, -2, -2], [2, 2, 2]])
    # One corner inside
    box_b = np.vstack([[0, 0, 0], [4, 4, 4]])

    with pytest.raises(
        TypeError,
        match=re.escape("Input extents must be a 2D numpy.ndarray."),
    ):
        box_intersect(np.r_[1], box_b)

    with pytest.raises(
        ValueError,
        match=re.escape("Extents must be of shape (2, N) containing the minimum and"),
    ):
        box_intersect(box_a, box_b[::-1, :])

    intersect = box_intersect(box_a, box_b, return_array=True)
    np.testing.assert_array_almost_equal(intersect, np.vstack([[0, 0, 0], [2, 2, 2]]))
    assert (np.diff(intersect, axis=0) > 0).all()
    # One face inside
    box_b = np.vstack([[-1, -1, 0], [1, 1, 4]])
    intersect = box_intersect(box_a, box_b, return_array=True)
    np.testing.assert_array_almost_equal(intersect, np.vstack([[-1, -1, 0], [1, 1, 2]]))
    assert (np.diff(intersect, axis=0) > 0).all()
    # All inside
    box_b = np.vstack([[-1, -1, -1], [1, 1, 1]])
    intersect = box_intersect(box_a, box_b, return_array=True)
    np.testing.assert_array_almost_equal(
        intersect, np.vstack([[-1, -1, -1], [1, 1, 1]])
    )
    assert (np.diff(intersect, axis=0) > 0).all()
    # All outside
    box_b = np.vstack([[1, 1, 4], [2, 2, 5]])
    intersect = box_intersect(box_a, box_b)
    assert not intersect

    # Repeat in 2D
    box_a = np.vstack([[-2, -2], [2, 2]])
    # One corner inside
    box_b = np.vstack([[0, 0], [4, 4]])
    intersect = box_intersect(box_a, box_b, return_array=True)
    np.testing.assert_array_almost_equal(intersect, np.vstack([[0, 0], [2, 2]]))
    assert (np.diff(intersect, axis=0) > 0).all()
