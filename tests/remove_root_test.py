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

import tempfile
from pathlib import Path

import numpy as np
from h5py import File

from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_remove_root():

    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testProject.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)
        points = Points.create(workspace, vertices=xyz)
        data = points.add_data(
            {
                "DataValues": {
                    "association": "VERTEX",
                    "values": np.random.randn(n_data),
                },
                "DataValues2": {
                    "association": "VERTEX",
                    "values": np.random.randn(n_data),
                },
            }
        )

        group_name = "SomeGroup"
        data_group = points.add_data_to_group(data, group_name)

        workspace.finalize()

        # Remove the root
        with File(h5file_path, "r+") as project:
            base = list(project.keys())[0]
            del project[base]["Root"]
            del project[base]["Groups"]
            del project[base]["Types"]["Group types"]

        # Read the data back in from a fresh workspace
        new_workspace = Workspace(h5file_path)

        rec_points = new_workspace.get_entity(points.name)[0]
        rec_group = rec_points.find_or_create_property_group(name=group_name)
        rec_data = new_workspace.get_entity(data[0].name)[0]

        compare_entities(
            points,
            rec_points,
            ignore=["_parent", "_existing_h5_entity", "_property_groups"],
        )
        compare_entities(data[0], rec_data, ignore=["_parent", "_existing_h5_entity"])
        compare_entities(
            data_group, rec_group, ignore=["_parent", "_existing_h5_entity"]
        )
