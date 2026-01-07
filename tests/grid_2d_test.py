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

import re

import numpy as np
import pytest

from geoh5py.objects import GeoImage, Grid2D
from geoh5py.shared.conversion import Grid2DConversion
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_attribute_setters():
    with Workspace() as workspace_context:
        with pytest.raises(TypeError, match="Attribute 'v_cell_size' must be"):
            Grid2D.create(
                workspace_context,
                origin=[0, 0, 0],
                u_cell_size=20.0,
                v_cell_size="abc",
                u_count=10,
                v_count=15,
                vertical=True,
            )

        with pytest.raises(TypeError, match="Dip angle must be a float"):
            Grid2D.create(
                workspace_context,
                origin=[0, 0, 0],
                u_cell_size=20.0,
                v_cell_size=10.0,
                u_count=10,
                v_count=15,
                dip="90",
            )

        grid = Grid2D.create(workspace_context)

        with pytest.raises(TypeError, match="Attribute 'last_focus'"):
            grid.last_focus = 666

        with pytest.raises(TypeError, match="Input 'entity_type'"):
            grid.validate_entity_type("bidon")


def test_create_grid_2d_data(tmp_path):
    name = "MyTestGrid2D"

    # Generate a 2D array
    h5file_path = tmp_path / r"test2Grid.geoh5"

    with Workspace.create(h5file_path) as workspace_context:
        grid = Grid2D.create(workspace_context)

        converter = Grid2DConversion
        for axis in ["u", "v"]:
            assert len(getattr(grid, f"cell_center_{axis}", None)) == 1

            with pytest.raises(
                TypeError,
                match=re.escape(f"Attribute '{axis}_cell_size' must be type(float)."),
            ):
                setattr(grid, f"{axis}_cell_size", "rando")

        assert grid.n_cells == 1
        assert grid.shape == (1, 1)

        grid.origin = [0, 0, 0]
        grid.u_cell_size = 20.0
        grid.v_cell_size = 30.0
        grid.name = name

        assert converter.grid_to_tag(grid)[33550] == (20.0, 30.0, 0.0)


def test_copy_from_extent():
    with Workspace() as workspace_context:
        grid = Grid2D.create(
            workspace_context,
            origin=[0, 0, 0],
            u_cell_size=np.r_[20.0],
            v_cell_size=np.r_[30.0],
            u_count=10,
            v_count=15,
            vertical=True,
        )
        assert grid.dip == 90.0

        with pytest.raises(TypeError, match="Expected a 2D numpy array"):
            grid.copy_from_extent(np.ones((3, 3)))


def test_grid2d_to_geoimage(tmp_path):
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
    h5file_path = tmp_path / r"test2Grid.geoh5"

    # Create a workspace
    converter = Grid2DConversion
    with Workspace.create(h5file_path) as workspace_context:
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

        assert isinstance(grid.origin, np.ndarray)

        with pytest.raises(TypeError, match="'The keys must be pass as a list"):
            grid.to_geoimage(("test", 3))

        with pytest.raises(KeyError, match=" you entered does not exists."):
            grid.to_geoimage("DataValues")

        with pytest.raises(
            IndexError, match="'int' values pass as key can't be larger"
        ):
            grid.to_geoimage(1000)

        data = grid.add_data({"DataValues": {"values": values.flatten()}})
        grid.rotation = 45.0

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        rec_obj = new_workspace.get_entity(name)[0]

        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(grid, rec_obj)
        compare_entities(data, rec_data)

        with pytest.raises(IndexError, match="Only 1, 3, or 4 layers can be selected"):
            grid.to_geoimage(["DataValues", "DataValues"])

        grid.rotation = 0.0
        geoimage = grid.to_geoimage(["DataValues"])
        geoimage.save_as("geotiff.tiff", path=tmp_path)
        assert isinstance(geoimage, GeoImage)

        _ = grid.to_geoimage(0)
        _ = grid.to_geoimage(data.uid)
        _ = grid.to_geoimage(data)

        assert grid.to_geoimage(["DataValues", "DataValues", "DataValues"])

        assert grid.to_geoimage(
            ["DataValues", "DataValues", "DataValues", "DataValues"]
        )

        with pytest.raises(AttributeError, match="No data is selected."):
            converter.convert_to_pillow("test")

        with pytest.raises(TypeError, match="The dtype of the keys must be"):
            converter.key_to_data(grid, [0, 1])
