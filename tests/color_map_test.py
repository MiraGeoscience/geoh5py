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


from __future__ import annotations

import numpy as np
import pytest

from geoh5py.data.color_map import ColorMap
from geoh5py.objects import Grid2D
from geoh5py.shared.utils import compare_entities
from geoh5py.shared.validators import ShapeValidationError
from geoh5py.workspace import Workspace


def test_create_color_map(tmp_path):
    name = "Grid2D_Colormap"

    # Generate a 2D array
    n_x, n_y = 10, 15
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))

    h5file_path = tmp_path / r"test_color_map.geoh5"

    standalone = ColorMap()
    assert standalone.values.shape[1] == 0

    # Create a workspace
    workspace = Workspace.create(h5file_path)
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

    data = grid.add_data({"DataValues": {"values": values.flatten()}})

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

    with pytest.raises(TypeError, match="Attribute 'color_map' must be of type"):
        data.entity_type.color_map = 1234

    with pytest.raises(ShapeValidationError) as error:
        data.entity_type.color_map = rgba

    assert ShapeValidationError.message("values", "(5, 10)", "(*, 5)") == str(
        error.value
    )

    data.entity_type.color_map = rgba.T

    with pytest.raises(TypeError, match="Input 'values' of ColorMap must be of type"):
        data.entity_type.color_map.values = "abc"

    with pytest.raises(
        ValueError, match="Input 'values' must contain fields with types"
    ):
        data.entity_type.color_map.values = np.core.records.fromarrays(
            rgba.T, names=("a", "b", "c", "d", "f")
        )

    data.entity_type.color_map.name = "my colours"
    workspace.close()

    # Read the data back in from a fresh workspace
    new_workspace = Workspace(h5file_path)
    rec_data = new_workspace.get_entity("DataValues")[0]
    compare_entities(
        rec_data.entity_type.color_map, data.entity_type.color_map, ignore=["_parent"]
    )

    new_workspace = Workspace.create(tmp_path / r"test_color_map_copy.geoh5")

    workspace.open(mode="r")
    data.copy(parent=new_workspace)

    rec_data = new_workspace.get_entity("DataValues")[0]

    assert np.all(
        getattr(rec_data.entity_type.color_map, key)
        == getattr(data.entity_type.color_map, key)
        for key in ["name", "values"]
    ), "Issue copying the ColorMap."

    assert len(rec_data.entity_type.color_map) == 10
