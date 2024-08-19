#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from geoh5py.data import data_type, geometric_data
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_xyz_dataype(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = Points.create(workspace, vertices=np.random.randn(10, 3))

        for axis in "XYZ":
            data = points.add_data(
                {
                    axis: {
                        "association": "VERTEX",
                        "entity_type": {"name": axis, "primitive_type": "GEOMETRIC"},
                    }
                }
            )

            assert isinstance(data, geometric_data.GeometricDataConstants)
            assert (
                data.entity_type.uid
                == getattr(data_type, f"GeometricData{axis}").default_type_uid()
            )


def test_dynamic_data():
    ws = Workspace(r"C:\Users\dominiquef\Desktop\Tests\colortable.geoh5")
    print(ws.objects[0].get_data("FlinFlon_geology"))
