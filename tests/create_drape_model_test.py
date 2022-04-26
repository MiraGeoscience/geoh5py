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

import os

import numpy as np

#
from geoh5py.objects import DrapeModel
from geoh5py.workspace import Workspace


def test_create_drape_model(tmp_path):
    # pass
    h5file_path = os.path.join(tmp_path, "drapedmodel.geoh5")
    workspace = Workspace(h5file_path)
    #
    # drape_model = workspace.get_entity("draped_models_line_id_1")[0]
    n_col, n_row = 64, 32
    j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))
    bottom = -np.sin(j / n_col * np.pi) * np.abs(np.cos(4 * i / n_col * np.pi)) - 0.1

    x = np.sin(2 * np.arange(n_col) / n_col * np.pi)
    y = np.cos(2 * np.arange(n_col) / n_col * np.pi)
    top = bottom.flatten()[::n_row] + 0.1
    layers = np.tile(n_row, n_col)
    drape = DrapeModel.create(
        workspace,
        prisms=np.c_[x, y, top, np.arange(0, i.flatten().shape[0], n_row), layers],
        layers=np.c_[i.flatten(), j.flatten(), bottom.flatten()],
    )
    drape.add_data(
        {"indices": {"values": (i * n_row + j).flatten(), "association": "CELL"}}
    )

    # print(drape_model.centroids)
    # new_workspace = Workspace("tester.geoh5")
    # drape_model.copy(parent=new_workspace)
    #
    # new_drape = new_workspace.get_entity("A LINE02__")[0]
    # compare_entities(drape_model, new_drape, ignore=["_parent"])
    # Generate a 2D array
    # n_x, n_y = 10, 15
    # values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
    #
    # with tempfile.TemporaryDirectory() as tempdir:
    #     h5file_path = Path(tempdir) / r"test2Grid.geoh5"
    #
    #     # Create a workspace
    #     workspace = Workspace(h5file_path)
    #
    #     grid = Grid2D.create(
    #         workspace,
    #         origin=[0, 0, 0],
    #         u_cell_size=20.0,
    #         v_cell_size=30.0,
    #         u_count=n_x,
    #         v_count=n_y,
    #         name=name,
    #         allow_move=False,
    #     )
    #
    #     data = grid.add_data({"DataValues": {"values": values}})
    #     grid.rotation = 45.0
    #
    #     workspace.finalize()
    #
    #     # Read the data back in from a fresh workspace
    #     new_workspace = Workspace(h5file_path)
    #
    #     rec_obj = new_workspace.get_entity(name)[0]
    #
    #     rec_data = new_workspace.get_entity("DataValues")[0]
    #
    #     compare_entities(grid, rec_obj)
    #     compare_entities(data, rec_data)
