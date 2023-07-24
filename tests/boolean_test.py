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
# import pytest

import numpy as np

from geoh5py.objects.grid2d import Grid2D
from geoh5py.workspace import Workspace


def test_data_boolean(tmp_path):
    h5file_path = tmp_path / r"testbool.geoh5"

    with Workspace.create(h5file_path) as workspace_context:
        grid = Grid2D.create(
            workspace_context,
            origin=[0, 0, 0],
            u_cell_size=20.0,
            v_cell_size=30.0,
            u_count=10,
            v_count=10,
            name="masking",
            allow_move=False,
        )

        values = np.zeros(grid.shape, dtype=bool)
        values[3:-3, 3:-3] = 1
        values = values.astype(bool)

        grid.add_data(
            {
                "my_boolean": {
                    "association": "CELL",
                    "values": values,
                }
            }
        )
