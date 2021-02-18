#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import Grid2D
from geoh5py.shared import Entity, EntityType
from geoh5py.workspace import Workspace


def compare_objects(object_a, object_b):
    for attr in object_a.__dict__.keys():
        if attr in ["_workspace", "_children"]:
            continue
        if isinstance(getattr(object_a, attr[1:]), (Entity, EntityType)):
            compare_objects(getattr(object_a, attr[1:]), getattr(object_b, attr[1:]))
        else:
            assert np.all(
                getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
            ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"


def test_create_grid_2d_data():

    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    d_x, d_y = 20.0, 30.0
    origin = [0, 0, 0]
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"test2Grid.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        grid = Grid2D.create(
            workspace,
            origin=origin,
            u_cell_size=d_x,
            v_cell_size=d_y,
            u_count=n_x,
            v_count=n_y,
            name=name,
            allow_move=False,
        )

        data = grid.add_data({"DataValues": {"values": values}})
        grid.rotation = 45.0

        workspace.finalize()

        # Read the data back in from a fresh workspace
        workspace = Workspace(h5file_path)

        rec_obj = workspace.get_entity(name)[0]

        rec_data = workspace.get_entity("DataValues")[0]

        compare_objects(grid, rec_obj)
        compare_objects(data, rec_data)
