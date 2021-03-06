#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import random
import string
import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_reference_data():

    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12
    values = np.random.randint(1, high=8, size=n_data)
    refs = np.unique(values)
    value_map = {}
    for ref in refs:
        value_map[ref] = "".join(
            random.choice(string.ascii_lowercase) for i in range(8)
        )

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testPoints.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        points = Points.create(
            workspace, vertices=np.random.randn(n_data, 3), name=name, allow_move=False
        )

        data = points.add_data(
            {
                "DataValues": {
                    "type": "referenced",
                    "values": values,
                    "value_map": value_map,
                }
            }
        )

        workspace = Workspace(h5file_path)
        rec_obj = workspace.get_entity(name)[0]
        rec_data = workspace.get_entity("DataValues")[0]

        compare_entities(points, rec_obj)
        compare_entities(data, rec_data)
