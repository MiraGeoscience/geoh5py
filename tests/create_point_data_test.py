#  Copyright (c) 2020 Mira Geoscience Ltd.
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
from geoh5py.shared import Entity, EntityType
from geoh5py.workspace import Workspace


def test_create_point_data():

    name = "MyTestPointset"
    new_name = "TestName"

    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)
    values = np.random.randn(n_data)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testPoints.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        points = Points.create(workspace, vertices=xyz, name=name, allow_move=False)

        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )

        # Change some data attributes for testing
        data.allow_delete = False
        data.allow_move = True
        data.allow_rename = False
        data.name = new_name

        workspace.finalize()

        # Read the data back in from a fresh workspace
        workspace = Workspace(h5file_path)

        rec_obj = workspace.get_entity(name)[0]
        rec_data = workspace.get_entity(new_name)[0]

        def compare_objects(object_a, object_b):
            for attr in object_a.__dict__.keys():
                if attr in ["_workspace", "_children"]:
                    continue
                if isinstance(getattr(object_a, attr[1:]), (Entity, EntityType)):
                    compare_objects(
                        getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                    )
                else:
                    assert np.all(
                        getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                    ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"

        compare_objects(points, rec_obj)
        compare_objects(data, rec_data)
