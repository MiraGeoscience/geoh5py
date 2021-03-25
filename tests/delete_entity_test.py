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

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_delete_entities():

    n_data = 12
    xyz = np.random.randn(n_data, 3)
    values = np.random.randn(n_data)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testPoints.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        group = ContainerGroup.create(workspace)

        points = Points.create(workspace, vertices=xyz, parent=group)

        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )

        points.delete()

        assert (
            (len(group.children) == 0)
            and (points.uid not in list(workspace.objects.keys()))
            and (data.uid not in list(workspace.data.keys()))
        ), "Object and data were not fully removed from the workspace"

        group.delete()

        assert group.uid not in list(
            workspace.groups.keys()
        ), "Group object not fully remove from the workspace"
