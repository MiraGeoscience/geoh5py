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

import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_no_data_values():

    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)
    float_values = np.random.randn(n_data)
    float_values[3:5] = np.nan

    int_values = np.random.randint(n_data, size=n_data).astype(float)
    int_values[2:5] = np.nan

    all_nan = np.ones(n_data)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testProject.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)
        points = Points.create(workspace, vertices=xyz)
        data_objs = points.add_data(
            {
                "DataFloatValues": {"association": "VERTEX", "values": float_values},
                "DataIntValues": {
                    "values": int_values,
                    "type": "INTEGER",
                },
                "NoValues": {"association": "VERTEX"},
                "AllNanValues": {"association": "VERTEX", "values": all_nan},
            }
        )
        data_objs[-1].values = None  # Reset all values to nan

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        for data in data_objs:
            rec_data = new_workspace.get_entity(data.name)[0]

            if data.values is None:
                assert rec_data.values is None, "Data 'values' saved should None"
            else:
                assert all(
                    np.isnan(rec_data.values) == np.isnan(data.values)
                ), "Mismatch between input and recovered data values"
