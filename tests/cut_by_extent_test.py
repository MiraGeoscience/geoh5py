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

import numpy as np
import pytest
from pydantic import ValidationError

from geoh5py.shared.cut_by_extent import Plane


@pytest.mark.parametrize(
    "origin,u,v,match",
    [
        # bad shape
        (
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            r"Expected shape \(3,\), got \(2,\)",
        ),
        # non-normalized
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            r"Vector is not normalized\.",
        ),
        # non-orthogonal
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            r"u_vector and v_vector are not orthogonal\.",
        ),
    ],
)
def test_plane_validate_uv_errors(origin, u, v, match):
    """Ensure Plane model_validator raises for invalid origin/u/v inputs with message match."""
    with pytest.raises(ValidationError, match=match):
        Plane(origin=origin, u_vector=u, v_vector=v)


def test_extent_from_vertices_and_box_input_validation_extent():
    """
    extent_from_vertices_and_box should raise when `extent` is not a numpy.ndarray
    but has shape (2, 3), per the input check.
    """
    plane = Plane(
        origin=np.array([0.0, 0.0, 0.0]),
        u_vector=np.array([1.0, 0.0, 0.0]),
        v_vector=np.array([0.0, 1.0, 0.0]),
    )

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    extent = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # Invalid `extent` input: plain list (not ndarray)
    bad = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(
        ValueError, match=r"vertices must be a numpy array of shape \(4, 3\)\."
    ):
        plane.extent_from_vertices_and_box(
            planar_object=None,  # type: ignore
            vertices=bad,
            extent=extent,
        )

    with pytest.raises(
        ValueError, match=r"extent must be a numpy array of shape \(2, 3\)\."
    ):
        plane.extent_from_vertices_and_box(
            planar_object=None,  # type: ignore
            vertices=vertices,
            extent=bad,
        )

    with pytest.raises(ValueError, match="Planar object must have"):
        plane.extent_from_vertices_and_box(
            planar_object=None,  # type: ignore
            vertices=vertices,
            extent=extent,
        )
