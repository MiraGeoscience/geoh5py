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

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_no_data_values(tmp_path):
    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)
    float_values = np.random.randn(n_data)
    float_values[3:5] = np.nan

    int_values = np.random.randint(n_data, size=n_data).astype(float)
    int_values[2:5] = np.nan

    all_nan = np.ones(n_data)
    h5file_path = tmp_path / r"testProject.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = Points.create(workspace, vertices=xyz)
        data_objs = points.add_data(
            {
                "DataFloatValues": {"association": "VERTEX", "values": float_values},
                "DataIntValues": {
                    "values": int_values,
                    "type": "FLOAT",
                },
                "NoValues": {"association": "VERTEX"},
                "AllNanValues": {"association": "VERTEX", "values": all_nan},
            }
        )
        data_objs[-1].values = None  # Reset all values to nan

        # Read the data back in from a fresh workspace
        with Workspace(h5file_path) as new_workspace:
            for data in data_objs:
                rec_data = new_workspace.get_entity(data.name)[0]

                if data.values is None:
                    assert rec_data.values is None, "Data 'values' saved should None"
                else:
                    assert all(np.isnan(rec_data.values) == np.isnan(data.values)), (
                        "Mismatch between input and recovered data values"
                    )
