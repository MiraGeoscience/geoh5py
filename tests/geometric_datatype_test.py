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

from geoh5py.data import data_type, geometric_data
from geoh5py.objects import Curve, Points
from geoh5py.workspace import Workspace


@pytest.mark.parametrize("obj_type", [Points, Curve])
def test_xyz_dataype(tmp_path: Path, obj_type):
    h5file_path = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = obj_type.create(workspace, vertices=np.random.randn(10, 3))

        for name in ["X", "Y", "Z"]:
            class_name = f"GeometricData{name}Type"
            dtype = getattr(data_type, class_name)
            dynamic_id = {
                value: key for key, value in data_type.DYNAMIC_CLASS_IDS.items()
            }
            data = points.add_data(
                {
                    name: {
                        "association": "VERTEX",
                        "name": name,
                        "primitive_type": "GEOMETRIC",
                        "dynamic_implementation_id": dynamic_id[dtype],
                    }
                }
            )

            assert isinstance(data, geometric_data.GeometricDataConstants)
            assert (
                data.entity_type.uid
                == getattr(data_type, class_name).default_type_uid()
            )

        copy_pts = points.copy()

        assert len(copy_pts.children) == 0

        assert data.copy(parent=copy_pts) is None

    ws = Workspace(h5file_path)
    assert all(
        isinstance(data, geometric_data.GeometricDataConstants) for data in ws.data
    )
