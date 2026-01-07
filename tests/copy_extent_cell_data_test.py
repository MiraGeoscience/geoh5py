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

import numpy as np
import pytest

from geoh5py.objects import DrapeModel, Grid2D
from geoh5py.workspace import Workspace

from .drape_model_test import create_drape_parameters


def test_copy_extent_grid_2d(tmp_path):
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    x_val, y_val = np.meshgrid(np.linspace(0, 909, n_x), np.linspace(100, 1500, n_y))
    values = x_val + y_val
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    workspace = Workspace.create(h5file_path)

    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        rotation=30,
        name=name,
        allow_move=False,
    )

    data = grid.add_data({"rando": {"values": values.flatten()}})

    with pytest.raises(TypeError, match="Expected a numpy array of extent values."):
        grid.copy_from_extent(data)

    assert (
        grid.copy_from_extent(np.r_[np.c_[-2000.0, -2000.0], np.c_[-1000.0, -1000.0]])
        is None
    )

    new_grid = grid.copy_from_extent(np.r_[np.c_[50, 50], np.c_[200, 200]])

    assert new_grid.n_cells == 30

    data_intersect = np.intersect1d(data.values, new_grid.children[0].values)

    assert data_intersect.size == 22
    assert (data_intersect.min() == 504) & (data_intersect.max() == 1309)

    # Repeat with inverse flag
    new_grid = grid.copy_from_extent(
        np.r_[np.c_[50, 50], np.c_[200, 200]], inverse=True
    )
    assert new_grid.n_cells == grid.n_cells
    assert np.isnan(new_grid.children[0].values).sum() == len(data_intersect)

    ind = (grid.centroids[:, 0] > 75) & (grid.centroids[:, 1] < 60)
    mask = data.mask_by_extent(np.vstack([[75, -100], [1000, 60]]))

    assert np.all(mask == ind), "Error masking data by extent."


def test_crop_image_rotated_dip(tmp_path):
    """
    Crop an image based on the rotation and dip of the grid.
    """
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    x_val, y_val = np.meshgrid(np.linspace(0, 909, n_x), np.linspace(100, 1500, n_y))
    values = x_val + y_val
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    workspace = Workspace.create(h5file_path)

    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        rotation=30,
        name=name,
        allow_move=False,
        dip=28,
    )

    grid.add_data({"rando": {"values": values.flatten()}})

    new_grid = grid.copy_from_extent(np.r_[np.c_[20, 20, 20], np.c_[200, 200, 40]])

    assert new_grid.rotation == 30
    assert new_grid.dip == 28
    assert new_grid.n_cells == 16
    assert new_grid.u_count == 8
    assert new_grid.v_count == 2


def test_crop_grid2d_rotated_dip_not_null_origin(tmp_path):
    """
    Crop an image based on the rotation and dip of the grid.
    """
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    x_val, y_val = np.meshgrid(np.linspace(0, 909, n_x), np.linspace(100, 1500, n_y))
    values = x_val + y_val
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    workspace = Workspace.create(h5file_path)
    origin = [10, 20, 30]

    grid = Grid2D.create(
        workspace,
        origin=origin,
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        rotation=30,
        name=name,
        allow_move=False,
        dip=28,
    )

    grid.add_data({"rando": {"values": values.flatten()}})

    new_grid2 = grid.copy_from_extent(
        np.r_[np.c_[20, 20, 20], np.c_[200, 200, 40]] + np.array(origin)
    )

    assert new_grid2.n_cells == 16
    assert new_grid2.u_count == 8
    assert new_grid2.v_count == 2
    assert (~np.isnan(new_grid2.children[0].values)).sum() == 15

    # Generate a 2D array
    n_x, n_y = 10, 15
    x_val, y_val = np.meshgrid(np.linspace(0, 909, n_x), np.linspace(100, 1500, n_y))
    values = x_val + y_val

    # doint the inverse
    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        rotation=30,
        name=f"invert{name}",
        allow_move=False,
        dip=28,
    )

    grid.add_data({"rando": {"values": values.flatten()}})

    new_grid = grid.copy_from_extent(
        np.r_[np.c_[20, 20, 20], np.c_[200, 200, 40]], inverse=True
    )

    assert new_grid.n_cells == grid.n_cells
    assert np.isnan(new_grid.children[0].values).sum() == 15
