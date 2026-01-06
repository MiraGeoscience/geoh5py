# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

from geoh5py.objects import DrapeModel, Points
from geoh5py.shared.merging import DrapeModelMerger
from geoh5py.workspace import Workspace


def create_drape_model(workspace: Workspace, alpha: float = 0.0):
    n_col, n_row = 64, 32

    j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))
    bottom = -np.sin(j / n_col * np.pi) * np.abs(np.cos(4 * i / n_col * np.pi)) - 0.1

    x = np.sin(2 * np.arange(n_col) / n_col * np.pi)
    y = np.cos(2 * np.arange(n_col) / n_col * np.pi) + alpha
    top = bottom.flatten()[::n_row] + 0.1

    layers = np.c_[i.flatten(), j.flatten(), bottom.flatten()]
    prisms = np.c_[
        x, y, top, np.arange(0, i.flatten().shape[0], n_row), np.tile(n_row, n_col)
    ]
    drape = DrapeModel.create(workspace, prisms=prisms, layers=layers)

    drape.add_data(
        {
            "indices": {
                "values": (i * n_row + j).flatten().astype(np.int32),
                "association": "CELL",
            }
        }
    )

    return drape


def test_merge_drape_model(tmp_path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / "drapedmodel.geoh5"
    with Workspace.create(h5file_path) as workspace_init:
        drape_models = []
        test_values = []
        test_prisms = []
        test_layers = []

        count_n_cells = 0

        # prepare the output
        for i in range(10):
            drape_model = create_drape_model(workspace_init, alpha=i * 2.5)
            count_n_cells += drape_model.n_cells
            drape_models.append(drape_model)

            test_values.append(drape_model.get_data("indices")[0].values)
            test_prisms.append(drape_model.prisms)
            test_layers.append(drape_model.layers)

    h5file_path_2 = tmp_path / "drapedmodel2.geoh5"
    with Workspace.create(h5file_path_2) as workspace:
        drape_model_merged = DrapeModelMerger.merge_objects(
            workspace, drape_models, name="merged", children=[Points.create(workspace)]
        )

        # test the output
        count_values = 0
        count_prism = 0
        out_values = drape_model_merged.get_data("indices")[0].values
        out_prisms = drape_model_merged.prisms
        out_layers = drape_model_merged.layers
        for i in range(10):
            np.testing.assert_almost_equal(
                out_values[count_values : count_values + test_values[i].shape[0]],
                test_values[i],
            )
            np.testing.assert_almost_equal(
                out_prisms[count_prism : count_prism + test_prisms[i].shape[0], :3],
                test_prisms[i][:, :3],
            )
            np.testing.assert_almost_equal(
                out_prisms[count_prism : count_prism + test_prisms[i].shape[0], -1],
                test_prisms[i][:, -1],
            )

            np.testing.assert_almost_equal(
                out_layers[count_values : count_values + test_layers[i].shape[0], 1:],
                test_layers[i][:, 1:],
            )
            count_values += test_values[i].shape[0] + 2
            count_prism += test_prisms[i].shape[0] + 2

        assert drape_model_merged.n_cells == count_n_cells + 18


def test_merge_drape_model_attribute_error(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"
    drape_models = []
    with Workspace.create(h5file_path) as workspace:
        for i in range(10):
            drape_model = create_drape_model(workspace, alpha=i * 2.5)
            drape_models.append(drape_model)

        drape_models[1] = DrapeModel.create(
            workspace, layers=(0, 0, -1), prisms=(0, 0, 0, 0, 0)
        )

        with pytest.raises(
            ValueError, match="All DrapeModel entities must have at least 2 prisms"
        ):
            _ = DrapeModelMerger.merge_objects(workspace, drape_models)
