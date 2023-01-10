#  Copyright (c) 2023 Mira Geoscience Ltd Ltd.
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

from geoh5py.objects import BlockModel
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_block_model_data(tmp_path):

    name = "MyTestBlockModel"
    h5file_path = tmp_path / r"block_model.geoh5"
    # Generate a 3D array
    n_x, n_y, n_z = 8, 9, 10

    nodal_x = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_x * 1.5 ** np.arange(3)[::-1],
                np.ones(n_x) * np.pi / n_x,
                np.pi / n_x * 1.5 ** np.arange(4),
            ]
        ),
    ]
    nodal_y = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_y * 1.5 ** np.arange(5)[::-1],
                np.ones(n_y) * np.pi / n_y,
                np.pi / n_y * 1.5 ** np.arange(6),
            ]
        ),
    ]
    nodal_z = -np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_z * 1.5 ** np.arange(7)[::-1],
                np.ones(n_z) * np.pi / n_z,
                np.pi / n_z * 1.5 ** np.arange(8),
            ]
        ),
    ]

    with Workspace(h5file_path) as workspace:
        grid = BlockModel.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_delimiters=nodal_x,
            v_cell_delimiters=nodal_y,
            z_cell_delimiters=nodal_z,
            name=name,
            rotation=30,
            allow_move=False,
        )

        data = grid.add_data(
            {
                "DataValues": {
                    "association": "CELL",
                    "values": (
                        np.cos(grid.centroids[:, 0])
                        * np.cos(grid.centroids[:, 1])
                        * np.cos(grid.centroids[:, 2])
                    ),
                }
            }
        )

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        rec_obj = new_workspace.get_entity(name)[0]
        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(grid, rec_obj)
        compare_entities(data, rec_data)
