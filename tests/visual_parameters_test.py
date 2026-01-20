# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

import logging

import numpy as np
import pytest

from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace


def test_visual_parameters(tmp_path, caplog):
    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12
    h5file_path = tmp_path / r"testTextData.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = Points.create(
            workspace,
            vertices=np.random.randn(n_data, 3),
            name=name,
            allow_move=False,
        )
        assert isinstance(points.visual_parameters, type(None))

        viz_params = points.add_default_visual_parameters()

        # Test color setter and round-trip transform
        new_colour = np.random.randint(0, 255, (3,)).tolist()
        viz_params.colour = new_colour

        assert points.visual_parameters.colour == new_colour

        # Test copying object with visual parameters
        copy = points.copy(name="CopyOfPoints")

        assert copy.visual_parameters.colour == viz_params.colour

        # Repeat with known color
        points.visual_parameters.colour = [255, 0, 0]  # Should be red

    with Workspace(h5file_path) as workspace:
        points = workspace.get_entity(name)[0]
        assert points.visual_parameters.colour == [255, 0, 0]

        viz_params = points.visual_parameters
        viz_params_b = viz_params.copy()

        points.visual_parameters = viz_params_b

        assert points.visual_parameters.uid == viz_params_b.uid
        assert viz_params not in points.children

        with caplog.at_level(logging.WARNING):
            points.add_data(
                {
                    "Visual Parameters": {
                        "name": "Visual Parameters",
                        "association": "OBJECT",
                        "primitive_type": "TEXT",
                    }
                }
            )
        assert any("Visual Parameters should not" in m for m in caplog.messages)

        with pytest.raises(UserWarning, match="Visual parameters already exist"):
            points.add_default_visual_parameters()

        with pytest.raises(TypeError, match="Input 'visual_parameters'"):
            points.visual_parameters = 1

        with pytest.raises(
            TypeError, match="Input 'colour' values must be a list of 3 or 4 integers"
        ):
            points.visual_parameters.colour = [255, 0]  # Wrong length


@pytest.mark.parametrize(
    "object_type,object_name,extra_params",
    [
        (Points, "TestPoints", {}),
        (Curve, "TestCurve", {}),
        (Grid2D, "TestGrid2D", {}),
        (
            Surface,
            "TestSurface",
            {"cells": np.array([[0, 1, 5], [1, 6, 5], [1, 2, 6]])},
        ),
    ],
)
def test_colour_setter(tmp_path, object_type, object_name, extra_params):
    """Test the colour setter with different object types."""
    h5file_path = tmp_path / f"test_{object_name}_colour.geoh5"
    colour = [0, 255, 0]  # Green

    # Create vertices
    if object_type == Surface:
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        x, y = x.ravel(), y.ravel()
        z = np.random.randn(x.shape[0])
        vertices = np.c_[x, y, z]
    else:
        vertices = np.random.randn(10, 3)

    with Workspace.create(h5file_path) as workspace:
        obj = object_type.create(
            workspace,
            vertices=vertices,
            name=object_name,
            **extra_params,
        )

        viz_params = obj.add_default_visual_parameters()
        viz_params.colour = colour

        assert obj.visual_parameters.colour == colour

    # Verify persistence
    with Workspace(h5file_path) as workspace:
        obj = workspace.get_entity(object_name)[0]
        assert obj.visual_parameters.colour == colour
