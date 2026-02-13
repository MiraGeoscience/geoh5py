# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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

# import pytest

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.data import PrimitiveTypeEnum
from geoh5py.objects.grid2d import Grid2D
from geoh5py.workspace import Workspace


def test_data_boolean(tmp_path):
    h5file_path = tmp_path / r"testbool.geoh5"
    h5file_path2 = tmp_path / r"testbool2.geoh5"

    with Workspace.create(h5file_path) as workspace_context:
        with Workspace.create(h5file_path2) as workspace_context2:
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

            _, non_bool = grid.add_data(
                {
                    "my_boolean": {
                        "association": "CELL",
                        "values": values.flatten(),
                    },
                    "non-bool": {
                        "association": "CELL",
                        "values": values.astype(float),
                    },
                }
            )

            values = np.ones(grid.shape)
            values[3:-3, 3:-3] = 0
            values[:1, :1] = np.nan

            grid.add_data(
                {
                    "my_boolean2": {
                        "association": "CELL",
                        "values": values.flatten(),
                        "entity_type": grid.get_data("my_boolean")[0].entity_type,
                    }
                }
            )

            assert (
                grid.get_data("my_boolean")[0].entity_type.primitive_type
                == PrimitiveTypeEnum.BOOLEAN
            )

            grid2 = grid.copy(parent=workspace_context2)

            # save the grid in a new workspace
            data2 = grid2.get_data("my_boolean")[0]

            assert all(data2.values == grid.get_data("my_boolean")[0].values)

            assert data2.entity_type.primitive_type == PrimitiveTypeEnum.BOOLEAN

            with pytest.raises(
                ValueError,
                match="Values provided by my_boolean are not containing only 0 or 1",
            ):
                data2.values = np.array([1.1, 0.2, 1.1])

            with pytest.raises(
                TypeError, match="Input 'entity_type' with primitive_type"
            ):
                data2.entity_type = non_bool.entity_type

            with pytest.raises(ValueError, match="Values provided by "):
                data2.values = np.array([0, 2, 1])

            with pytest.raises(TypeError, match="Input 'values' must be a numpy array"):
                data2.values = "bidon"

    with Workspace(h5file_path, mode="r") as workspace:
        grid2 = workspace.get_entity("masking")[0]
        data2 = grid2.get_data("my_boolean")[0]
        assert all(data2.values == grid.get_data("my_boolean")[0].values)
        assert data2.entity_type.primitive_type == PrimitiveTypeEnum.BOOLEAN
