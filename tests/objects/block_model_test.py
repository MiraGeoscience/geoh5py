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

from geoh5py.objects import BlockModel
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_negative_cell_delimiters_centroids(tmp_path):
    workspace = Workspace.create(tmp_path / "test.geoh5")
    block_model = BlockModel.create(
        workspace,
        name="test",
        u_cell_delimiters=np.array([-1, 0, 1]),
        v_cell_delimiters=np.array([-1, 0, 1]),
        z_cell_delimiters=np.array([-3, -2, -1]),
        origin=np.r_[0, 0, 0],
    )
    assert np.allclose(
        block_model.centroids[:, 0],
        np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]),
    )
    assert np.allclose(
        block_model.centroids[:, 1],
        np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    assert np.allclose(
        block_model.centroids[:, 2],
        np.array([-2.5, -1.5, -2.5, -1.5, -2.5, -1.5, -2.5, -1.5]),
    )


def test_create_block_model_data(tmp_path):
    name = "MyTestBlockModel"
    h5file_path = tmp_path / r"block_model.geoh5"
    # Generate a 3D array

    nodal_x = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / 8 * 1.5 ** np.arange(3)[::-1],
                np.ones(8) * np.pi / 8,
                np.pi / 8 * 1.5 ** np.arange(4),
            ]
        ),
    ]
    nodal_y = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / 9 * 1.5 ** np.arange(5)[::-1],
                np.ones(9) * np.pi / 9,
                np.pi / 9 * 1.5 ** np.arange(6),
            ]
        ),
    ]
    nodal_z = -np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / 10 * 1.5 ** np.arange(7)[::-1],
                np.ones(10) * np.pi / 10,
                np.pi / 10 * 1.5 ** np.arange(8),
            ]
        ),
    ]

    with Workspace.create(h5file_path) as workspace:
        with pytest.raises(
            TypeError, match=r"Attribute 'u_cell_delimiters' must be a numpy array\."
        ):
            BlockModel.create(workspace, u_cell_delimiters="abc")

        with pytest.raises(ValueError, match="must be a 1D array of floats"):
            BlockModel.create(workspace, u_cell_delimiters=np.ones((2, 2)))

        grid = BlockModel.create(
            workspace,
            u_cell_delimiters=nodal_x,
            v_cell_delimiters=nodal_y,
            z_cell_delimiters=nodal_z,
            name=name,
            rotation=30.0,
            allow_move=False,
        )
        assert grid.mask_by_extent(np.vstack([[-100, -100], [-1, -1]])) is None

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

        assert grid.mask_by_extent(np.vstack([[-100, -100], [-1, -1]])) is None
        # Read the data back in from a fresh workspace
        with Workspace(h5file_path) as new_workspace:
            rec_obj = new_workspace.get_entity(name)[0]
            rec_data = new_workspace.get_entity("DataValues")[0]
            compare_entities(grid, rec_obj)
            compare_entities(data, rec_data)

        with pytest.raises(TypeError, match="Mask must be a numpy array of shape"):
            grid.copy(mask="abc")

        # mask = np.ones(grid.n_cells, dtype=bool)
        # mask[-2:] = False

        grid_copy = grid.copy(rotation=0.0)

        mask = grid_copy.mask_by_extent(np.vstack([[-100, -100], [1, 100]]))

        grid_copy_copy = grid_copy.copy_from_extent(np.vstack([[-100, -100], [1, 100]]))

        assert np.all(~np.isnan(grid_copy_copy.children[0].values) == mask)
        assert mask.sum() == np.prod(grid.shape[1:])

        grid_copy_copy = grid_copy.copy(cell_mask="abc", mask=mask)
        assert grid_copy.n_cells == grid.n_cells
        assert np.all(~np.isnan(grid_copy_copy.children[0].values) == mask)
        assert np.all(
            grid_copy.mask_by_extent(np.vstack([[-100, -100], [1, 100]]), inverse=True)
            == ~mask
        )


def test_block_extents(tmp_path):
    origin = np.random.randn(3)
    rotation = np.random.randint(0, 90, 1)

    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        mesh = BlockModel.create(
            workspace,
            name="test",
            u_cell_delimiters=np.array([-0.5, 0.0, 0.5]),
            v_cell_delimiters=np.array([-0.5, 0.0, 0.5]),
            z_cell_delimiters=np.array([-0.5, 0.0, 0.5]),
            origin=origin,
            rotation=rotation,
        )

    angle = np.deg2rad(rotation)
    delta = np.sin(angle) + np.cos(angle)
    assert np.allclose(
        mesh.extent,
        np.c_[
            origin - np.r_[delta, delta, 1] * 0.5,
            origin + np.r_[delta, delta, 1] * 0.5,
        ].T,
    )


def test_shaped_data_values():
    with Workspace() as workspace:
        block_model = BlockModel.create(
            workspace,
            u_cell_delimiters=np.array([0.0, 1.0, 2.0]),
            v_cell_delimiters=np.array([0.0, 1.0, 2.0]),
            z_cell_delimiters=np.array([0.0, -1.0, -2.0]),
        )
        values = np.arange(block_model.n_cells, dtype=float)
        block_model.add_data({"DataValues": {"association": "CELL", "values": values}})
        shaped = block_model.shaped_data_values("DataValues")
        assert shaped.shape == block_model.shape
        assert np.allclose(shaped.ravel(), values)
        assert np.allclose(shaped, block_model.shaped_data_values(values))

        block_model.add_data({"DataValues2": {"association": "CELL", "values": shaped}})

        data1 = block_model.get_data("DataValues")[0].values
        data2 = block_model.get_data("DataValues2")[0].values
        assert np.allclose(data1, data2)
