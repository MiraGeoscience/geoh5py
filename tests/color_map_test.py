#  Copyright (c) 2022 Mira Geoscience Ltd.
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
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_color_map():

    name = "Grid2D_Colormap"

    # Generate a 2D array
    n_x, n_y = 10, 15
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"test_color_map.geoh5"

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

        data = grid.add_data({"DataValues": {"values": values}})

        n_c = 10
        rgba = np.vstack(
            [
                np.linspace(values.min(), values.max(), n_c),  # Values
                np.linspace(0, 255, n_c),  # Red
                np.linspace(255, 0, n_c),  # Green
                np.linspace(125, 15, n_c),  # Blue,
                np.ones(n_c) * 255,  # Alpha,
            ]
        )
        data.entity_type.color_map.values = rgba
        workspace.finalize()

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        rec_obj = new_workspace.get_entity(name)[0]

        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(grid, rec_obj)
        compare_entities(data, rec_data)
