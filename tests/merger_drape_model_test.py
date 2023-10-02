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

import numpy as np

from geoh5py.objects import DrapeModel
from geoh5py.shared.merging import DrapeModelMerger
from geoh5py.workspace import Workspace


def create_drape_model(workspace: Workspace, alpha: float = 0.0):
    n_col, n_row = 64, 32

    j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))
    bottom = -np.sin(j / n_col * np.pi) * np.abs(np.cos(4 * i / n_col * np.pi)) - 0.1

    x = np.sin(2 * np.arange(n_col) / n_col * np.pi)
    y = np.cos(2 * np.arange(n_col) / n_col * np.pi) + alpha
    top = bottom.flatten()[::n_row] + 0.1
    drape = DrapeModel.create(workspace)

    layers = np.c_[i.flatten(), j.flatten(), bottom.flatten()]
    drape.layers = layers

    prisms = np.c_[
        x, y, top, np.arange(0, i.flatten().shape[0], n_row), np.tile(n_row, n_col)
    ]
    drape.prisms = prisms

    drape.add_data(
        {
            "indices": {
                "values": (i * n_row + j).flatten().astype(np.int32),
                "association": "CELL",
            }
        }
    )

    return drape


def test_merge_drape_model(tmp_path):
    h5file_path = tmp_path / "drapedmodel.geoh5"
    with Workspace.create(h5file_path) as workspace:
        drape_models = []
        count = 0
        for i in range(10):
            drape_model = create_drape_model(workspace, alpha=i * 2.5)
            count += drape_model.n_cells
            drape_models.append(drape_model)

        drape_model_merged = DrapeModelMerger.merge_objects(
            workspace, drape_models, name="merged"
        )

        assert drape_model_merged.n_cells == count + 18
