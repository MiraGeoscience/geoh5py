# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import DrapeModel
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


# pylint: disable=too-many-locals


def create_drape_parameters():
    """
    Utility function to generate basic drape model
    """
    n_col, n_row = 64, 32
    j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))
    bottom = -np.sin(j / n_col * np.pi) * np.abs(np.cos(4 * i / n_col * np.pi)) - 0.1

    x = np.sin(2 * np.arange(n_col) / n_col * np.pi)
    y = np.cos(2 * np.arange(n_col) / n_col * np.pi)
    top = bottom.flatten()[::n_row] + 0.1

    layers = np.c_[i.flatten(), j.flatten(), bottom.flatten()]
    prisms = np.c_[
        x, y, top, np.arange(0, i.flatten().shape[0], n_row), np.tile(n_row, n_col)
    ]
    return layers, prisms


def test_create_drape_model(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        layers, prisms = create_drape_parameters()

        with pytest.raises(TypeError, match="Attribute 'layers' must be"):
            DrapeModel.create(workspace, layers="abc")

        with pytest.raises(ValueError, match="Array of 'layers' must be of shape"):
            DrapeModel.create(workspace, layers=(0, 0))

        with pytest.raises(TypeError, match="Attribute 'prisms' must be"):
            DrapeModel.create(workspace, layers=layers, prisms="abc")

        with pytest.raises(ValueError, match="Array of 'prisms' must be of shape"):
            DrapeModel.create(workspace, layers=layers, prisms=(0, 0))

        drape = DrapeModel.create(workspace, layers=layers, prisms=prisms)

        with pytest.raises(AttributeError):
            layers[-32:, 0] = 64
            drape.layers = layers

        drape.add_data(
            {
                "indices": {
                    "values": np.arange(drape.n_cells).astype(np.int32),
                    "association": "CELL",
                }
            }
        )

        with Workspace.create(tmp_path / "tester.geoh5") as new_workspace:
            drape.copy(parent=new_workspace)

        with Workspace(tmp_path / "tester.geoh5") as new_workspace:
            rec_drape = new_workspace.objects[0]
            compare_entities(drape, rec_drape, ignore=["_parent"])


def test_centroids(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        layers, prisms = create_drape_parameters()

        drape = DrapeModel.create(workspace, layers=layers, prisms=prisms)

        assert drape.centroids.shape[0] == len(layers)

        rand_ind = np.random.randint(0, len(layers), 10)
        np.testing.assert_array_almost_equal(
            drape.centroids[rand_ind, 2],
            layers[rand_ind, 2] + drape.z_cell_size[rand_ind] / 2,
        )


def test_copy_extent(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        layers, prisms = create_drape_parameters()

        drape = DrapeModel.create(workspace, layers=layers, prisms=prisms)
        vals = drape.add_data({"indices": {"values": np.random.randn(drape.n_cells)}})
        new_drape = drape.copy_from_extent([[-2, -2], [-0.2, 0]])

        assert len(new_drape.layers) == 448
        np.testing.assert_array_almost_equal(
            new_drape.children[0].values, vals.values[(1568 - 448) : 1568]
        )
