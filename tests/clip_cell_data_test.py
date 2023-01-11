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

import numpy as np
import pytest

from geoh5py.objects import Grid2D
from geoh5py.workspace import Workspace


def test_copy_extent_grid_2d(tmp_path):
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    workspace = Workspace(h5file_path)

    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        name=name,
        allow_move=False,
    )

    data = grid.add_data({"rando": {"values": values.flatten()}})

    with pytest.warns(
        UserWarning,
        match=f"Method 'clip_by_extent' for entity {Grid2D} not fully implemented.",
    ):
        new_grid = grid.copy_from_extent(
            np.r_[np.c_[-100, -100, 0], np.c_[200, 200, 0]]
        )

    assert new_grid.n_cells == grid.n_cells
    assert new_grid.children[0].values.shape == data.values.shape
