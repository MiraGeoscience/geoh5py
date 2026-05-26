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

import logging

import numpy as np
import pytest

from geoh5py.objects import Points, Surface
from geoh5py.workspace import Workspace


def test_colour_data(tmp_path, caplog):
    # Generate a 2D array
    h5file_path = tmp_path / r"testColour.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = Points.create(workspace, vertices=np.random.rand(3, 3))

        values = np.array(
            [(10, 50, 90), (60, 100, 40), (210, 150, 120)], dtype=np.uint8
        )

        result0 = np.array(
            [(10, 50, 90, 255), (60, 100, 40, 255), (210, 150, 120, 255)],
            dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)],
        )

        # create a colour object
        points.add_data({"colour1": {"values": values}})

    # reopen
    with Workspace(h5file_path) as workspace:
        points = workspace.get_entity("Points")[0]
        colour_data = points.get_data("colour1")[0]

        assert all(colour_data.values == result0)

        # create values1: a float np array
        values1 = np.array(
            [[np.nan, 0.5, 0.4], [0.6, 1.0, 0.4]],
        )

        colour_data.values = values1

        result1 = np.array(
            [(90, 90, 90, 0), (255, 255, 0, 255), (90, 90, 90, 0)],
            dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)],
        )

        assert all(colour_data.values == result1)

        # create values2, a structured float array rgb
        values2 = np.array(
            [
                (np.nan, 0.5, 0.9, 0.2),
                (0.6, 1.0, 0.4, 0.1),
                (2.1, 1.5, 1.2, 0.6),
                (2.1, 1.5, 0.4, 0.9),
            ],
            dtype=[
                ("r", np.float32),
                ("g", np.float32),
                ("b", np.float32),
                ("a", np.float32),
            ],
        )

        with caplog.at_level(logging.WARNING):
            colour_data.values = values2

        assert "Input 'values' of shape" in caplog.text

        result2 = np.array(
            [(90, 90, 90, 0), (72, 127, 0, 0), (255, 255, 255, 159)],
            dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)],
        )

        assert all(colour_data.values == result2)

        data_object = points.add_data(
            {
                "colour2": {
                    "values": values1,
                    "association": "Object",
                    "type": "Colour",
                }
            }
        )

        result3 = np.array(
            [(90, 90, 90, 0), (255, 255, 0, 255)],
            dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)],
        )

        assert all(data_object.values == result3)


def test_colour_errors():
    with Workspace() as workspace:
        values = np.array(
            [(10, 50, 90), (60, 100, 40), (210, 150, 120)],
            dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8)],
        )

        from geoh5py.shared.utils import array_is_colour

        surface = Surface.create(workspace)

        with pytest.raises(TypeError, match="Parent 'Surface' is not allowed"):
            surface.add_data(
                {
                    "colour1": {
                        "values": values,
                    }
                }
            )

        points = Points.create(workspace, vertices=np.random.rand(3, 3))

        data_colour = points.add_data(
            {
                "colour1": {
                    "values": values,
                }
            }
        )

        with pytest.raises(TypeError, match="Values must be a numpy array"):
            data_colour.values = "test"

        with pytest.raises(ValueError, match="Values must be a 2D numpy array"):
            data_colour.values = np.array([1, 2, 3])

        no_colour = np.array(
            [(10, 50, 90), (60, 100, 40), (210, 150, 120)],
            dtype=[("a", np.uint8), ("b", np.uint8), ("c", np.uint8)],
        )

        with pytest.raises(
            ValueError, match="Values must be a 2D numpy array containing RGB bands"
        ):
            data_colour.values = no_colour

        with pytest.raises(
            NotImplementedError, match="Only add_data values of type FLOAT,"
        ):
            points.add_data({"colour_false": {"values": no_colour}})
