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


# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.objects import Grid2D
from geoh5py.shared.conversion.base import CellObjectConversion
from geoh5py.workspace import Workspace


def test_create_grid_2d_data(tmp_path):
    # Create a workspace
    h5file_path = tmp_path / r"test2Grid.geoh5"

    with Workspace.create(h5file_path) as workspace_context:
        # test base converter
        n_x, n_y, name = 10, 15, "test"
        grid = Grid2D.create(
            workspace_context,
            origin=[0, 0, 0],
            u_cell_size=20.0,
            v_cell_size=30.0,
            u_count=n_x,
            v_count=n_y,
            name=name,
            allow_move=False,
        )

        converter = CellObjectConversion

        values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
        grid.add_data(
            data={"DataValues": {"values": values.flatten(), "association": "CELL"}}
        )

        points = converter.to_points(grid)

        with pytest.raises(TypeError, match="Input entity for `GridObject` conversion"):
            converter.to_points(points)
