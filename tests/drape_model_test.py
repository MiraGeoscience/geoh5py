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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import DrapeModel
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_drape_model(tmp_path: Path):
    h5file_path = tmp_path / "drapedmodel.geoh5"
    with Workspace().save(h5file_path) as workspace:
        #
        # drape_model = workspace.get_entity("draped_models_line_id_1")[0]
        n_col, n_row = 64, 32
        j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))
        bottom = (
            -np.sin(j / n_col * np.pi) * np.abs(np.cos(4 * i / n_col * np.pi)) - 0.1
        )

        x = np.sin(2 * np.arange(n_col) / n_col * np.pi)
        y = np.cos(2 * np.arange(n_col) / n_col * np.pi)
        top = bottom.flatten()[::n_row] + 0.1
        drape = DrapeModel.create(workspace)

        with pytest.raises(AttributeError) as error:
            getattr(drape, "centroids")

        assert "Attribute 'layers'" in str(error)

        drape.layers = np.c_[i.flatten(), j.flatten(), bottom.flatten()]

        with pytest.raises(AttributeError) as error:
            getattr(drape, "centroids")

        assert "Attribute 'prisms'" in str(error)

        drape.prisms = np.c_[
            x, y, top, np.arange(0, i.flatten().shape[0], n_row), np.tile(n_row, n_col)
        ]

        drape.add_data(
            {
                "indices": {
                    "values": (i * n_row + j).flatten().astype(np.int32),
                    "association": "CELL",
                }
            }
        )

        with Workspace().save(tmp_path / "tester.geoh5") as new_workspace:
            drape.copy(parent=new_workspace)

        with Workspace(tmp_path / "tester.geoh5") as new_workspace:
            rec_drape = new_workspace.objects[0]
            compare_entities(drape, rec_drape, ignore=["_parent"])
