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

from geoh5py.objects import GeoImage, Grid2D
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_grid_2d_data(tmp_path):

    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    workspace = Workspace(h5file_path)

    with workspace.open("r+") as workspace_context:
        grid = Grid2D.create(workspace_context)

        with pytest.raises(AttributeError, match="The Grid2D has no geographic"):
            grid.get_tag()

        grid.u_cell_size = 20.0
        grid.v_cell_size = 20.0

        with pytest.raises(AttributeError, match="The Grid2D has no number of cells"):
            grid.get_tag()

        workspace_context.remove_entity(grid)

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

        with pytest.raises(TypeError, match="The type of the keys"):
            grid.to_geoimage(123)

        with pytest.raises(KeyError, match=" is not in the data: "):
            grid.to_geoimage("DataValues")

        data = grid.add_data({"DataValues": {"values": values}})
        grid.rotation = 45.0

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        rec_obj = new_workspace.get_entity(name)[0]

        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(grid, rec_obj)
        compare_entities(data, rec_data)

        with pytest.raises(IndexError, match="Only 1, 3, or 4 layers can be selected"):
            grid.to_geoimage(["DataValues", "DataValues"])

        with pytest.raises(UserWarning, match="Cannot assign tag for rotated Grid2D."):
            geoimage = grid.to_geoimage(["DataValues"])

        grid.rotation = 0.0
        geoimage = grid.to_geoimage(["DataValues"])
        geoimage.save_as("geotiff.tiff", path=tmp_path)
        assert isinstance(geoimage, GeoImage)

        geoimage = grid.to_geoimage(["DataValues", "DataValues", "DataValues"])

        geoimage = grid.to_geoimage(
            ["DataValues", "DataValues", "DataValues", "DataValues"]
        )
