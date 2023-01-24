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

import pytest

from geoh5py.shared.utils import (
    iterable,
    iterable_message,
    mask_by_extent,
    overwrite_kwargs,
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
        match=re.escape("Input 'extent' must be an array-like of shape(2, 3)."),
    ):
        mask_by_extent(points, corners)

    with pytest.raises(
        ValueError,
        match=re.escape("Input 'locations' must be an array-like of shape(*, 3)."),
    ):
        mask_by_extent(points, corners[:2])

    assert not mask_by_extent([points], corners[:2]), "Point should have been outside."


def test_overwrite_kwargs():

    kwargs = {"test": 8}
    kwargs_to_overwrite = {"test": 9}

    new_kwargs = overwrite_kwargs(kwargs, kwargs_to_overwrite)

    assert new_kwargs["test"] == kwargs_to_overwrite["test"]
