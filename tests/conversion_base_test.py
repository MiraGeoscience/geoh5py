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


# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.objects import Grid2D
from geoh5py.shared.conversion.base import CellObject, ConversionBase
from geoh5py.workspace import Workspace


def test_create_grid_2d_data(tmp_path):

    # Create a workspace
    h5file_path = tmp_path / r"test2Grid.geoh5"
    workspace = Workspace(h5file_path)

    with workspace.open("r+") as workspace_context:
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

        converter = ConversionBase(entity=grid)

        with pytest.raises(ValueError, match="Workspace has not been defined"):
            _ = converter.workspace_output

        with pytest.raises(TypeError, match="Name must be a string"):
            converter.name = 0

        converter.change_workspace_parent(workspace=workspace_context)

        with pytest.raises(ValueError, match="Output has not been created"):
            converter.copy_properties()

        # test cell object

        with pytest.raises(TypeError, match="Input entity for `GridObject` conversion"):
            _ = CellObject(None)

        values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
        grid.add_data(data={"DataValues": {"values": values, "association": "CELL"}})
        cell_converter = CellObject(grid)

        with pytest.raises(ValueError, match="Output has not been created"):
            cell_converter.copy_properties()

        with pytest.raises(ValueError, match="Output has not been created"):
            cell_converter.copy_child_properties(association="VERTEX")

        cell_converter.to_points()
        cell_converter.to_points(parent=grid.parent)
